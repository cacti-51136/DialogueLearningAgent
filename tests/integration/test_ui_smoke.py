"""PyQt UI 冒烟测试（doc/04 §4）。

仅验证「模块可导入 + MainWindow 可构造 + 离线引擎可接入」；交互逻辑由
``test_engine_stream`` 通过事件流覆盖。需在安装了 PyQt6 的环境运行（offscreen 平台）。
"""

import os
import sys
import tempfile

# 必须在导入任何 dla / PyQt 模块前设置，避免测试在仓库里落库
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["DLA_DB__PATH"] = os.path.join(tempfile.gettempdir(), "dla_ui_smoke.db")

sys.path.insert(0, "src")

pytest = __import__("pytest")
pytest.importorskip("PyQt6")

# 强制以当前 env（含临时 DB 路径）重载 settings，避免被同进程内其他测试缓存的默认 settings 影响
from dla.config.settings import get_settings  # noqa: E402

get_settings(reload=True)

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from apps.ui.main import MainWindow  # noqa: E402


def test_ui_module_imports():
    # 导入成功本身即验证 PyQt 依赖与路径处理正确
    assert MainWindow is not None


def test_main_window_constructs():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    assert win.engine is not None
    assert win.weight_tree is not None
    assert win.session_list.count() >= 1  # 首屏已建一个会话
    # 关闭事件不应抛异常
    win.close()


def test_ui_runs_a_turn_offscreen():
    """真实发一条消息、等流式回合结束，断言对话区拿到干净回复（无 <turn_summary> 泄露）。

    该用例守护两个曾出现的回归：
    - SQLite 连接跨线程（主线程建连 / worker 线程写库）；
    - 流式在 <turn_summary> 开标签处提前 break，导致闭标签未收集、标签泄露进回复。
    """
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    result = {"text": None, "err": None}

    def on_done(ev):
        result["text"] = ev.final_text
        app.quit()

    def on_failed(msg):
        result["err"] = msg
        app.quit()

    win.input.setPlainText("这个语法太难了，我完全不会，好烦。")
    win._on_send()
    win._worker.finished_turn.connect(on_done)
    win._worker.failed.connect(on_failed)
    QTimer.singleShot(10000, app.quit)  # 安全闸：10s 内未结束则强制退出，避免挂起
    app.exec()

    assert result["err"] is None, f"worker 出错：{result['err']}"
    assert result["text"], "未产出最终回复"
    assert "<turn_summary>" not in result["text"], "回复中泄露了 <turn_summary> 标签"
    chat = win.chat_view.toPlainText()
    assert "🤖" in chat, "对话区未呈现 bot 回复"
    win.close()

