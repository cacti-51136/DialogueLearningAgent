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
