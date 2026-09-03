"""DLA 命令行入口（doc/04 交互层 / doc/01 选型 typer→本期用标准库 argparse 实现等价功能）。

子命令：
- chat   ：对话（--mode fixed|auto|free / --scenario / --describe / --message 单次 / --explain / --debug）
- scenario ：list / show / validate / export
- keyword ：map list / map reset（kw_agent_map，doc/03 §2.15）
- ctx    ：status / compact（上下文压缩日志，doc/11）
- bench  ：离线剧本回归（FakeLLM，验证权重演化）

无 API key 时自动退回 FakeLLM，保证开箱即跑（doc/01 §3 / openai_compat）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# 确保项目根在 sys.path（以 `python apps/cli/main.py` 直接运行时）
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from src.dla.config.loader import get_keyword_lib  # noqa: E402
from src.dla.config.scenario_loader import (  # noqa: E402
    list_scenarios,
    load_scenario_by_id,
)
from src.dla.config.settings import get_settings  # noqa: E402
from src.dla.llm.openai_compat import make_llm_client  # noqa: E402
from src.dla.orchestration.engine import DialogueEngine  # noqa: E402
from src.dla.storage.migrator import migrate  # noqa: E402
from src.dla.storage.repositories import SQLiteRepo  # noqa: E402
from src.dla.storage.sqlite import get_connection  # noqa: E402
from src.dla.tools.loader import discover_all  # noqa: E402
from src.dla.tools.registry import ToolRegistry  # noqa: E402
from src.dla.tools.watcher import HotReloadWatcher  # noqa: E402


def _build_registry(settings) -> ToolRegistry:
    """扫描插件目录 + entry_points 构建工具注册表（doc/08）。"""
    registry = ToolRegistry()
    if settings.tools_enabled:
        discovered = discover_all()
        registry.load_from(discovered)
    return registry


def _build_engine(args, with_db: bool = True, force_fake: bool = False):
    settings = get_settings()
    lib = get_keyword_lib()
    if force_fake:
        # 离线回归（bench）强制 FakeLLM，保证确定性、不触碰真实 API
        from src.dla.llm.openai_compat import FakeLLMClient

        llm = FakeLLMClient(model=settings.llm_model)
    else:
        llm = make_llm_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_model,
            settings.llm_timeout_seconds, settings.llm_max_retries,
        )
    repo = None
    if with_db and not getattr(args, "no_db", False):
        # check_same_thread=False：工具执行器（doc/08 executor）在独立线程运行工具，
        # 而工具（如 recall_memory）会经 ctx.repo/ctx.memory 访问本连接。若为 True，
        # 每次工具调用都会抛 "SQLite objects created in a thread..."，并被静默吞掉。
        # 与 apps/api、apps/ui 保持一致。
        conn = get_connection(settings.db_path, check_same_thread=False)
        migrate(conn, "migrations")
        repo = SQLiteRepo(conn)
    registry = _build_registry(settings)
    engine = DialogueEngine(settings, lib, llm, repo, tool_registry=registry)
    return settings, lib, llm, repo, engine, registry


def _maybe_start_watcher(settings, registry):
    """若配置了自动热更新，启动监听线程（doc/08 §3.2）。返回 watcher 或 None。"""
    mode = settings.tools_auto_reload
    if mode not in ("watch", "shadow"):
        return None
    plugin_dir = settings.tools_plugin_dir
    if not os.path.isabs(plugin_dir):
        plugin_dir = os.path.join(_PROJ_ROOT, plugin_dir)
    watcher = HotReloadWatcher(
        registry, plugin_dir, mode=mode,
        on_reload=lambda r: None,
    )
    watcher.start()
    return watcher


def _print_weights(snapshot) -> None:
    print("  [L1]", {k: round(v, 2) for k, v in snapshot.l1.items()})
    print("  [L2]", {k: round(v, 2) for k, v in snapshot.l2.items()})
    print("  [L3]", {k: round(v, 2) for k, v in snapshot.l3.items()})


def _print_chain(chain) -> None:
    print("  ── 思维链 ──")
    for frame, data in chain:
        print(f"  ▸ {frame}: {json.dumps(data, ensure_ascii=False)}")


# ----------------------------- chat -----------------------------
def cmd_chat(args) -> int:
    settings, lib, llm, repo, engine, registry = _build_engine(args, with_db=not args.no_db)
    watcher = _maybe_start_watcher(settings, registry)
    mode = args.mode or settings.mode_scenario
    scenario = args.scenario or settings.scenario_default
    greeting = engine.start_session(mode=mode, scenario_id=scenario, describe=args.describe)
    print(f"[会话开始] 模式={mode} 场景={scenario}")
    print("Agent:", greeting)

    if args.message:
            reply, meta = engine.send(args.message)
            print("Agent:", reply)
            if args.explain:
                _print_weights(meta["snapshot"])
            if args.debug:
                _print_chain(meta["debug_chain"])
            if watcher is not None:
                watcher.stop()
            return 0

    print("（输入 /exit 结束）")
    while True:
        try:
            u = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if u in ("/exit", "/quit"):
            break
        if not u:
            continue
        reply, meta = engine.send(u)
        print("Agent:", reply)
        if args.explain:
            _print_weights(meta["snapshot"])
        if args.debug:
            _print_chain(meta["debug_chain"])
    if watcher is not None:
        watcher.stop()
    return 0


# --------------------------- scenario ---------------------------
def cmd_scenario(args) -> int:
    settings = get_settings()
    if args.sub == "list":
        for s in list_scenarios(settings.scenario_dir):
            print(f"{s.id:16} {s.name}  modes={','.join(s.modes)}")
        return 0
    if args.sub == "show":
        s = load_scenario_by_id(settings.scenario_dir, args.id)
        print(f"id={s.id} name={s.name}")
        print(f"greeting={s.greeting}")
        print("l1:", s.l1)
        print("l2_preset:", s.l2_preset)
        print("l3_baseline:", s.l3_baseline)
        if s.safe_mode:
            print("safe_mode:", s.safe_mode)
        return 0
    if args.sub == "validate":
        problems = 0
        lib = get_keyword_lib()
        for s in list_scenarios(settings.scenario_dir):
            for k in list(s.l1) + list(s.l2_preset):
                if not lib.lexicon.is_known(k):
                    print(f"[WARN] 场景 {s.id} 引用未知关键词: {k}")
                    problems += 1
            for k in s.l3_baseline:
                if not lib.lexicon.is_known(k):
                    print(f"[WARN] 场景 {s.id} l3_baseline 未知关键词: {k}")
                    problems += 1
            # 情绪/脾性不得 preset（doc/09 / doc/02 §11.9）
            for k in s.l2_preset:
                if k.startswith("user_mood.") or k.startswith("user_temper."):
                    print(f"[WARN] 场景 {s.id} l2_preset 含情绪/脾性词(应靠对话涌现): {k}")
                    problems += 1
        print(f"校验完成：{problems} 个问题" if problems else "校验通过：无问题")
        return 1 if problems else 0
    if args.sub == "export":
        out = {}
        for s in list_scenarios(settings.scenario_dir):
            out[s.id] = {"name": s.name, "l1": s.l1, "l2_preset": s.l2_preset, "l3_baseline": s.l3_baseline}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    print("未知 scenario 子命令"); return 2


# --------------------------- keyword ---------------------------
def cmd_keyword(args) -> int:
    settings = get_settings()
    conn = get_connection(settings.db_path)
    migrate(conn, "migrations")
    repo = SQLiteRepo(conn)
    if args.sub == "map" and args.action == "list":
        cur = conn.cursor()
        cur.execute("SELECT src_keyword, dst_keyword, direction, delta, observed_count FROM kw_agent_map ORDER BY delta DESC")
        rows = cur.fetchall()
        if not rows:
            print("（kw_agent_map 为空，随对话涌现累积，doc/03 §2.15）")
        for r in rows:
            print(f"{r['src_keyword']} → {r['dst_keyword']} [{r['direction']}] δ={r['delta']:.3f} n={r['observed_count']}")
        return 0
    if args.sub == "map" and args.action == "reset":
        repo.kwmap_reset()
        print("kw_agent_map 已清空")
        return 0
    print("未知 keyword 子命令"); return 2


# ----------------------------- ctx -----------------------------
def cmd_ctx(args) -> int:
    settings = get_settings()
    conn = get_connection(settings.db_path)
    migrate(conn, "migrations")
    cur = conn.cursor()
    if args.sub == "status":
        cur.execute("SELECT COUNT(*) FROM context_compact_log")
        n = cur.fetchone()[0]
        print(f"上下文压缩日志条数: {n}")
        print(f"compact_ratio 阈值: {settings.ctx_compact_ratio}  hard: {settings.ctx_hard_ratio}  auto: {settings.ctx_auto_compact}")
        # 按档位汇总（doc/11 §8.1 可观测）
        cur.execute("SELECT trigger_level, COUNT(*) FROM context_compact_log GROUP BY trigger_level ORDER BY 2 DESC")
        by_level = cur.fetchall()
        if by_level:
            print("档位分布: " + "  ".join(f"{lv}={c}" for lv, c in by_level))
        return 0
    if args.sub == "compact":
        if getattr(args, "force", False):
            # 真正触发一次压缩（doc/11 §8.1）：构建引擎并强制 compact 当前会话
            _s, _l, _r, _repo, engine, _reg = _build_engine(args, with_db=True)
            engine.start_session()
            res = engine.force_compact()
            if res is None:
                print("（无法执行压缩：无活动会话）")
                return 1
            if not res["triggered"]:
                print(f"未达阈值，无需压缩（ratio_before={res['ratio_before']:.3f} < {settings.ctx_compact_ratio}）")
            else:
                print(f"已强制压缩：ratio {res['ratio_before']:.3f}→{res['ratio_after']:.3f}")
                print(f"动作: {res['actions']}")
            return 0
        cur.execute(
            "SELECT turn, trigger_level, ratio_before, ratio_after, tokens_before, tokens_after, actions_json"
            " FROM context_compact_log ORDER BY id DESC LIMIT 20"
        )
        rows = cur.fetchall()
        if not rows:
            print("（暂无压缩记录）")
        for r in rows:
            print(
                f"turn={r['turn']} [{r['trigger_level']}] "
                f"ratio {r['ratio_before']:.3f}→{r['ratio_after']:.3f} "
                f"tokens {r['tokens_before']}→{r['tokens_after']} "
                f"actions={r['actions_json']}"
            )
        return 0
    print("未知 ctx 子命令"); return 2


# ----------------------------- ops -----------------------------
def cmd_ops(args) -> int:
    """词库操作审计：列出 scene_ops / agent_ops 的即时增删改记录（doc/06 §4.2/§4.4）。

    固定模式全程跳过运行期增删改；其余模式的操作（含被护栏拒绝的）均落 lexicon_ops。
    """
    settings = get_settings()
    conn = get_connection(settings.db_path)
    migrate(conn, "migrations")
    cur = conn.cursor()
    if args.sub == "list":
        limit = int(getattr(args, "limit", 20) or 20)
        sid = getattr(args, "session", None)
        if sid:
            cur.execute(
                "SELECT turn, op_type, layer, target_key, applied, llm_reason"
                " FROM lexicon_ops WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (sid, limit),
            )
        else:
            cur.execute(
                "SELECT session_id, turn, op_type, layer, target_key, applied"
                " FROM lexicon_ops ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
        if not rows:
            print("（暂无词库操作记录）")
            return 0
        for r in rows:
            state = "生效" if r["applied"] else "拒绝"
            if sid:
                detail = f"  {r['llm_reason'] or ''}".rstrip()
                print(f"turn={r['turn']:>3} [{r['op_type']:6}] {r['layer']} {r['target_key']} [{state}]{detail}")
            else:
                print(f"{r['session_id']} turn={r['turn']:>3} [{r['op_type']:6}] {r['layer']} {r['target_key']} [{state}]")
        return 0
    print("未知 ops 子命令"); return 2


# ----------------------------- tool -----------------------------
def cmd_tool(args) -> int:
    settings = get_settings()
    registry = _build_registry(settings)
    if args.sub == "list":
        if not registry.snapshot():
            print("（未加载任何工具插件）")
            return 0
        for t in registry.all():
            state = "禁用" if not registry.is_enabled(t.name) else "启用"
            ro = "只读" if t.is_readonly else "有副作用"
            print(f"{t.name:24} v{getattr(t, 'version', '?')}  [{state}/{ro}]  {t.description[:42]}")
        return 0
    if args.sub == "describe":
        t = registry.get(args.name)
        if t is None:
            print(f"未找到工具: {args.name}")
            return 1
        print(f"name: {t.name}")
        print(f"version: {getattr(t, 'version', '?')}")
        print(f"readonly: {t.is_readonly}  enabled: {registry.is_enabled(t.name)}")
        print(f"description: {t.description}")
        print(f"parameters: {json.dumps(t.parameters, ensure_ascii=False)}")
        return 0
    if args.sub == "reload":
        result = registry.reload(args.name, shadow=args.shadow)
        if result["ok"]:
            if result["shadow"]:
                print(f"已载入影子快照（观测中），用 `dla tool promote {args.name or ''}` 确认切换")
            else:
                print(f"重载成功：生效工具 {result['count']} 个")
        else:
            print(f"重载失败（已回滚）：{result['error']}")
            return 1
        return 0
    if args.sub == "promote":
        result = registry.promote_shadow()
        if result["ok"]:
            print(f"影子快照已提升为生效快照，共 {result['count']} 个工具")
        else:
            print(f"提升失败：{result['error']}")
            return 1
        return 0
    if args.sub in ("enable", "disable"):
        on = args.sub == "enable"
        ok = registry.set_enabled(args.name, on)
        if not ok:
            print(f"未找到工具: {args.name}")
            return 1
        print(f"{args.name} 已{'启用' if on else '禁用'}")
        return 0
    print("未知 tool 子命令"); return 2


# ---------------------------- bench ----------------------------
def cmd_bench(args) -> int:
    # 强制 FakeLLM：bench 是确定性离线回归，不依赖/不触碰真实 API
    settings, lib, llm, repo, engine, _registry = _build_engine(args, with_db=False, force_fake=True)
    engine.start_session(mode="fixed", scenario_id="oral_practice")
    script = [
        "你好，我想练口语。",
        "这个语法太难了，我完全不会。",
        "我有点受挫，感觉学不会。",
        "还是不太懂，能不能简单点。",
        "好吧，我试试看。",
    ]
    print("剧本回归（FakeLLM，验证 L3 随对话演化）：")
    traj: List[float] = []
    prev_emp = None
    for i, u in enumerate(script, 1):
        reply, meta = engine.send(u)
        l3 = meta["snapshot"].l3
        emp = l3.get("empathy", 0.0)
        traj.append(emp)
        arrow = "" if prev_emp is None else ("▲" if emp > prev_emp else ("▼" if emp < prev_emp else "＝"))
        print(f"[{i}] empathy={emp:.3f} {arrow} | agent={reply[:30]}")
        prev_emp = emp

    # 断言：受挫信号（脚本 2/3 句）应使 empathy 高于无信号基线（脚本 1 句）
    baseline_emp = traj[0]
    peak_emp = max(traj)
    rose = peak_emp > baseline_emp + 1e-6
    print("-" * 40)
    print(f"基线 empathy={baseline_emp:.3f}  峰值 empathy={peak_emp:.3f}  "
          f"{'✅ 受挫后共情度上升' if rose else '❌ 未观测到上升'}")
    print("bench 完成。")
    return 0 if rose else 1


# --------------------------- parser ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dla", description="DialogueLearningAgent CLI")
    sub = p.add_subparsers(dest="cmd")

    pc = sub.add_parser("chat", help="对话")
    pc.add_argument("--mode", choices=["fixed", "auto", "free"], default=None)
    pc.add_argument("--scenario", default=None)
    pc.add_argument("--describe", default=None, help="free 模式场景描述")
    pc.add_argument("--message", default=None, help="单次对话消息（不进入交互）")
    pc.add_argument("--explain", action="store_true", help="打印三层权重")
    pc.add_argument("--debug", action="store_true", help="打印思维链帧")
    pc.add_argument("--no-db", action="store_true", help="不持久化到 SQLite")

    ps = sub.add_parser("scenario", help="场景模板")
    pss = ps.add_subparsers(dest="sub")
    pss.add_parser("list")
    psh = pss.add_parser("show"); psh.add_argument("id")
    pss.add_parser("validate")
    pss.add_parser("export")

    pk = sub.add_parser("keyword", help="关键词/映射")
    pks = pk.add_subparsers(dest="sub")
    pkm = pks.add_parser("map"); pkm.add_argument("action", choices=["list", "reset"])

    px = sub.add_parser("ctx", help="上下文压缩")
    pxs = px.add_subparsers(dest="sub")
    pxs.add_parser("status")
    pxc = pxs.add_parser("compact")
    pxc.add_argument("--force", action="store_true", help="真正触发一次压缩（doc/11）")

    pt = sub.add_parser("tool", help="工具插件管理（doc/08）")
    pts = pt.add_subparsers(dest="sub")
    pts.add_parser("list")
    ptd = pts.add_parser("describe"); ptd.add_argument("name")
    ptr = pts.add_parser("reload"); ptr.add_argument("name", nargs="?", default=None); ptr.add_argument("--shadow", action="store_true", help="载入影子快照（观测中），需在同进程 promote 才可生效；CLI 单发命令默认直接重载")
    pts.add_parser("promote")
    pte = pts.add_parser("enable"); pte.add_argument("name")
    ptd2 = pts.add_parser("disable"); ptd2.add_argument("name")

    po = sub.add_parser("ops", help="词库操作审计（scene_ops/agent_ops，doc/06 §4.2/§4.4）")
    pos = po.add_subparsers(dest="sub")
    pol = pos.add_parser("list")
    pol.add_argument("--session", default=None, help="限定会话 id（缺省列出全部会话）")
    pol.add_argument("--limit", type=int, default=20, help="最多展示条数")

    pb = sub.add_parser("bench", help="离线剧本回归")
    pb.add_argument("--no-db", action="store_true")

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "chat":
        return cmd_chat(args)
    if args.cmd == "scenario":
        return cmd_scenario(args)
    if args.cmd == "keyword":
        return cmd_keyword(args)
    if args.cmd == "ctx":
        return cmd_ctx(args)
    if args.cmd == "tool":
        return cmd_tool(args)
    if args.cmd == "ops":
        return cmd_ops(args)
    if args.cmd == "bench":
        return cmd_bench(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
