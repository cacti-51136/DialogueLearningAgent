"""FastAPI / SSE 适配层（doc/04 §3）。

薄适配：复用核心引擎 ``DialogueEngine``，把对话能力暴露为 HTTP 接口；核心包零强依赖，
FastAPI 仅作为外壳（与 UI 同构）。支持：

- ``GET  /health`` ``GET /ready``           存活/就绪探针
- ``POST /v1/sessions`` ``GET /v1/sessions`` 会话生命周期
- ``POST /v1/sessions/{id}/messages``        发送消息；``Accept: text/event-stream`` 走 SSE 流式
- ``GET  /v1/sessions/{id}/messages``        历史消息
- ``GET  /v1/sessions/{id}/weights``         当前权重（?layer=L2）
- ``GET  /v1/sessions/{id}/weights/history`` 权重演化历史
- ``GET  /v1/scenarios``                     场景模板列表
- ``GET  /v1/tools``                         已注册工具
- ``POST /v1/sessions/{id}/tools/{name}``    按需调用工具（doc/08 recall_memory 等）

安全：CORS 显式来源白名单（生产禁用 *）；统一安全响应头；错误体统一结构，绝不返回堆栈。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

from dla.config.loader import get_keyword_lib  # noqa: E402
from dla.config.scenario_loader import list_scenarios  # noqa: E402
from dla.config.settings import get_settings  # noqa: E402
from dla.core.events import (  # noqa: E402
    ChainStepEvent,
    DoneEvent,
    ErrorEvent,
    PersonaChangeEvent,
    TokenEvent,
    WeightUpdateEvent,
)
from dla.llm.openai_compat import make_llm_client  # noqa: E402
from dla.orchestration.engine import DialogueEngine  # noqa: E402
from dla.storage.migrator import migrate  # noqa: E402
from dla.storage.repositories import SQLiteRepo  # noqa: E402
from dla.storage.sqlite import get_connection  # noqa: E402
from dla.tools import build_builtin_registry  # noqa: E402


# ---- 请求体 ----
class CreateSessionReq(BaseModel):
    mode: str = "fixed"
    scenario_id: Optional[str] = None
    sid: Optional[str] = None


class MessageReq(BaseModel):
    text: str


class ToolCallReq(BaseModel):
    args: Dict[str, Any] = {}


# ---- 引擎构建（与 UI 同构）----
def _build_engine() -> DialogueEngine:
    settings = get_settings()
    lib = get_keyword_lib()
    llm = make_llm_client(
        settings.llm_api_key, settings.llm_base_url, settings.llm_model,
        settings.llm_timeout_seconds, settings.llm_max_retries,
    )
    repo = None
    try:
        # API 在请求线程经仓储读写，连接需允许跨线程（访问已串行化，doc/04 §4）
        conn = get_connection(settings.db_path, check_same_thread=False)
        migrate(conn, "migrations")
        repo = SQLiteRepo(conn)
    except Exception:  # noqa: BLE001 - 无 DB 也能跑（内存模式）
        repo = None
    engine = DialogueEngine(settings, lib, llm, repo)
    if settings.tools_enabled:
        try:
            engine.tool_registry = build_builtin_registry()
        except Exception:  # noqa: BLE001
            engine.tool_registry = None
    return engine


def _error(code: str, message: str, status: int = 400) -> JSONResponse:
    """统一错误体结构（doc/04 §3.1）：绝不返回堆栈与内部路径。"""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": ""}},
    )


def _event_to_sse(ev: Any) -> Optional[Dict[str, Any]]:
    """把引擎事件映射为 SSE 事件（doc/04 §3.1）。"""
    if isinstance(ev, TokenEvent):
        return {"event": "token", "data": {"text": ev.text}}
    if isinstance(ev, WeightUpdateEvent):
        return {
            "event": "weights",
            "data": {"l1": ev.snapshot.l1, "l2": ev.snapshot.l2, "l3": ev.snapshot.l3, "turn": ev.snapshot.turn},
        }
    if isinstance(ev, PersonaChangeEvent):
        return {"event": "persona_change", "data": {"delta": ev.delta, "action": ev.action}}
    if isinstance(ev, ChainStepEvent):
        return {"event": "chain", "data": {"step": ev.step, "detail": ev.detail}}
    if isinstance(ev, DoneEvent):
        return {
            "event": "done",
            "data": {
                "turn": ev.turn,
                "final_text": ev.final_text,
                "summary": ev.summary,
                "rep_hit": ev.rep_hit,
                "candidate_count": ev.candidate_count,
            },
        }
    if isinstance(ev, ErrorEvent):
        return {"event": "error", "data": {"message": ev.message}}
    return None


def create_app(settings=None, engine=None) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or _build_engine()

    app = FastAPI(title="DialogueLearningAgent API", version="0.1.0")

    # CORS：仅显式白名单（doc/04 §3.2）
    origins = [o.strip() for o in (settings.api_cors_origins or "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    # ---- 探针 ----
    @app.get("/health")
    async def health():
        return {"status": "ok", "time": time.time()}

    @app.get("/ready")
    async def ready():
        ok = engine.repo is not None and getattr(engine, "lib", None) is not None
        if not ok:
            return _error("NOT_READY", "DB 或词库未就绪", status=503)
        return {"status": "ready"}

    # ---- 场景 ----
    @app.get("/v1/scenarios")
    async def scenarios():
        try:
            return [{"id": s.id, "name": s.name} for s in list_scenarios(engine.settings.scenario_dir)]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    # ---- 会话 ----
    @app.post("/v1/sessions")
    async def create_session(req: CreateSessionReq):
        try:
            engine.start_session(mode=req.mode, scenario_id=req.scenario_id, sid=req.sid)
        except Exception as e:  # noqa: BLE001
            return _error("SESSION_CREATE_FAILED", str(e), status=500)
        return {"sid": engine._active, "greeting": "（会话已创建）"}

    @app.get("/v1/sessions")
    async def list_sessions():
        return {"sessions": engine.list_sessions()}

    @app.get("/v1/sessions/{sid}/messages")
    async def get_messages(sid: str, limit: int = 100):
        if engine.repo is None:
            return {"messages": []}
        try:
            return {"messages": engine.repo.list_messages(sid, limit)}
        except Exception:  # noqa: BLE001
            return {"messages": []}

    @app.get("/v1/sessions/{sid}/weights")
    async def get_weights(sid: str, layer: str = "L2"):
        if engine.repo is not None:
            snap = engine.repo.get_latest_snapshot(sid)
            if snap is not None:
                weights = {"L1": snap.l1, "L2": snap.l2, "L3": snap.l3}.get(layer, snap.l2)
                return {"session_id": sid, "layer": layer, "weights": weights}
        return {"session_id": sid, "layer": layer, "weights": {}}

    @app.get("/v1/sessions/{sid}/weights/history")
    async def get_weights_history(sid: str, limit: int = 200):
        if engine.repo is None:
            return {"history": []}
        return {"history": engine.repo.list_snapshots(sid, limit)}

    # ---- 工具（doc/08）----
    @app.get("/v1/tools")
    async def list_tools():
        return {"tools": engine.list_tools()}

    @app.post("/v1/sessions/{sid}/tools/{name}")
    async def call_tool(sid: str, name: str, req: ToolCallReq):
        engine.switch_session(sid)
        result = engine.call_tool(name, req.args, session_id=sid)
        return {
            "ok": result.ok,
            "content": result.content,
            "metadata": result.metadata,
            "error": result.error,
        }

    # ---- 消息发送（流式 / 非流式）----
    def _run_turn(sid: str, text: str):
        """在独立线程跑同步流式入口，避免阻塞事件循环。"""
        engine.switch_session(sid)
        return list(engine.stream_reply_sync(text))

    @app.post("/v1/sessions/{sid}/messages")
    async def post_message(sid: str, req: MessageReq, request: Request):
        if not req.text or not req.text.strip():
            return _error("BAD_REQUEST", "text 不能为空")

        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept:

            async def event_gen():
                loop = asyncio.get_event_loop()
                queue: "asyncio.Queue" = asyncio.Queue()

                def producer():
                    try:
                        for ev in _run_turn(sid, req.text):
                            asyncio.run_coroutine_threadsafe(queue.put(ev), loop).result()
                    except Exception as e:  # noqa: BLE001
                        asyncio.run_coroutine_threadsafe(queue.put(ErrorEvent(message=str(e))), loop).result()
                    finally:
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

                loop.run_in_executor(None, producer)
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    sse = _event_to_sse(item)
                    if sse is not None:
                        yield sse

            return EventSourceResponse(event_gen())

        # 非流式：收集完整结果
        try:
            events = await asyncio.get_event_loop().run_in_executor(None, _run_turn, sid, req.text)
        except Exception as e:  # noqa: BLE001
            return _error("ENGINE_ERROR", str(e), status=500)

        final_text = summary = ""
        turn = rep_hit = 0
        for ev in events:
            if isinstance(ev, DoneEvent):
                final_text = ev.final_text
                summary = ev.summary
                turn = ev.turn
                rep_hit = ev.rep_hit
        return {"sid": sid, "turn": turn, "text": final_text, "summary": summary, "rep_hit": bool(rep_hit)}

    app.state.engine = engine
    return app


def main() -> None:
    settings = get_settings()
    import uvicorn

    uvicorn.run(create_app(), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
