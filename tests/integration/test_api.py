"""API 层集成测试（doc/04 §3 FastAPI/SSE）。

用 fastapi TestClient 验证全部端点；SSE 走流式分支，断言事件帧正确。
"""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient  # noqa: E402

from dla.config.loader import get_keyword_lib  # noqa: E402
from dla.config.settings import Settings  # noqa: E402
from dla.llm.openai_compat import make_llm_client  # noqa: E402
from dla.orchestration.engine import DialogueEngine  # noqa: E402
from dla.storage.migrator import migrate  # noqa: E402
from dla.storage.repositories import SQLiteRepo  # noqa: E402
from dla.storage.sqlite import get_connection  # noqa: E402
from dla.tools import build_builtin_registry  # noqa: E402
from apps.api.main import create_app  # noqa: E402


def _engine(tmp_path):
    s = Settings()
    s.memory_retrieve_trigger = "always"
    s.memory_retrieve_sim_threshold = 0.0
    lib = get_keyword_lib()
    llm = make_llm_client("", "https://api.openai.com/v1", "gpt-4o-mini")
    db = str(tmp_path / "api.db")
    conn = get_connection(db, check_same_thread=False)
    migrate(conn, "migrations")
    repo = SQLiteRepo(conn)
    e = DialogueEngine(s, lib, llm, repo)
    e.tool_registry = build_builtin_registry()
    return e


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(engine=_engine(tmp_path)))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_session_and_message(client):
    r = client.post("/v1/sessions", json={"sid": "s1"})
    assert r.status_code == 200
    sid = r.json()["sid"]
    r = client.post(f"/v1/sessions/{sid}/messages", json={"text": "你好"})
    assert r.status_code == 200
    assert r.json()["text"]


def test_sse_stream(client):
    client.post("/v1/sessions", json={"sid": "s2"})
    with client.stream("POST", "/v1/sessions/s2/messages", json={"text": "你好"},
                       headers={"accept": "text/event-stream"}) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "event: token" in text
    assert "event: done" in text


def test_tools_list_and_call(client):
    client.post("/v1/sessions", json={"sid": "s3"})
    client.post("/v1/sessions/s3/messages", json={"text": "我母语是粤语。"})

    r = client.get("/v1/tools")
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["tools"]]
    assert "recall_memory" in names

    r = client.post("/v1/sessions/s3/tools/recall_memory", json={"args": {"query": "我的母语"}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "粤语" in body["content"]


def test_weights_and_history(client):
    client.post("/v1/sessions", json={"sid": "s4"})
    client.post("/v1/sessions/s4/messages", json={"text": "你好"})
    r = client.get("/v1/sessions/s4/weights?layer=L2")
    assert r.status_code == 200
    assert "weights" in r.json()
    r = client.get("/v1/sessions/s4/weights/history")
    assert r.status_code == 200
    assert "history" in r.json()


def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
