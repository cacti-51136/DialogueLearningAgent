"""PyQt6 桌面界面入口（doc/04 §4）。

运行：``python apps/ui/main.py`` 或安装后 ``dla-ui``。

依赖：``pip install pyqt6``（可选 extra ``ui``）。核心引擎仍零强依赖，PyQt 仅作为外壳。
"""

from __future__ import annotations

import os
import sys
import time

# 允许以脚本方式直接运行（python apps/ui/main.py）
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from PyQt6.QtCore import QThread, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

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


class StreamWorker(QThread):
    """在独立线程跑引擎流式入口，通过 Signal 把事件推回主线程（doc/04 §4.2）。"""

    token_received = pyqtSignal(str)
    weights_updated = pyqtSignal(object)
    persona_changed = pyqtSignal(object)
    chain_step = pyqtSignal(object)
    finished_turn = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, engine: DialogueEngine, text: str) -> None:
        super().__init__()
        self._engine = engine
        self._text = text
        self._stop = False

    def run(self) -> None:
        try:
            for ev in self._engine.stream_reply_sync(self._text):
                if self._stop:
                    break
                if isinstance(ev, TokenEvent):
                    self.token_received.emit(ev.text)
                elif isinstance(ev, WeightUpdateEvent):
                    self.weights_updated.emit(ev.snapshot)
                elif isinstance(ev, PersonaChangeEvent):
                    self.persona_changed.emit(ev)
                elif isinstance(ev, ChainStepEvent):
                    self.chain_step.emit(ev)
                elif isinstance(ev, DoneEvent):
                    self.finished_turn.emit(ev)
                elif isinstance(ev, ErrorEvent):
                    self.failed.emit(ev.message)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不让线程静默死亡
            self.failed.emit(str(exc))

    def stop(self) -> None:
        self._stop = True


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DialogueLearningAgent")
        self.resize(1180, 720)

        self.engine = self._build_engine()
        self._active_sid: str = ""
        self._worker: StreamWorker | None = None
        self._live_buffer = ""

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(30)  # 防抖：合并高频 token 刷新（doc/04 §4.2）
        self._flush_timer.timeout.connect(self._flush_live)

        self._build_ui()
        self._init_session()

    # ---- 构建 ----
    def _build_engine(self) -> DialogueEngine:
        settings = get_settings()
        lib = get_keyword_lib()
        llm = make_llm_client(
            settings.llm_api_key, settings.llm_base_url, settings.llm_model,
            settings.llm_timeout_seconds, settings.llm_max_retries,
        )
        repo = None
        try:
            # UI 在 worker 线程经仓储写库，连接需允许跨线程使用（访问已串行化，doc/04 §4）
            conn = get_connection(settings.db_path, check_same_thread=False)
            migrate(conn, "migrations")
            repo = SQLiteRepo(conn)
        except Exception:  # noqa: BLE001 - 无 DB 也能跑（内存模式）
            repo = None
        return DialogueEngine(settings, lib, llm, repo)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        # 顶部工具栏：模式 / 场景 / 新建会话
        top = QHBoxLayout()
        top.addWidget(QLabel("模式"))
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["fixed", "auto", "free"])
        top.addWidget(self.mode_cb)
        top.addWidget(QLabel("场景"))
        self.scene_cb = QComboBox()
        try:
            for sc in list_scenarios(self.engine.settings.scenario_dir):
                self.scene_cb.addItem(sc.name, sc.id)
        except Exception:  # noqa: BLE001
            pass
        top.addWidget(self.scene_cb)
        self.new_btn = QPushButton("新建会话")
        self.new_btn.clicked.connect(self._on_new_session)
        top.addWidget(self.new_btn)
        top.addStretch(1)
        self.status_lbl = QLabel("就绪")
        top.addWidget(self.status_lbl)
        root_layout.addLayout(top)

        # 三栏
        split = QSplitter()
        root_layout.addWidget(split, 1)

        # 左：会话列表
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("会话"))
        self.session_list = QListWidget()
        self.session_list.currentTextChanged.connect(self._on_switch_session)
        left_layout.addWidget(self.session_list)
        split.addWidget(left)

        # 中：对话区
        mid = QWidget()
        mid_layout = QVBoxLayout(mid)
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        mid_layout.addWidget(self.chat_view, 1)
        self.live_view = QPlainTextEdit()
        self.live_view.setReadOnly(True)
        self.live_view.setMaximumHeight(120)
        mid_layout.addWidget(self.live_view)
        input_row = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setMaximumHeight(60)
        self.input.textChanged.connect(lambda: None)
        input_row.addWidget(self.input, 1)
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)
        mid_layout.addLayout(input_row)
        split.addWidget(mid)

        # 右：权重面板 + 思维链
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("实时权重（灵魂面板）"))
        self.weight_tree = QTreeWidget()
        self.weight_tree.setHeaderLabels(["关键词", "权重"])
        self.weight_tree.setColumnCount(2)
        right_layout.addWidget(self.weight_tree, 2)
        self.chain_cb = QPushButton("显示思维链")
        self.chain_cb.setCheckable(True)
        right_layout.addWidget(self.chain_cb)
        self.chain_view = QTextEdit()
        self.chain_view.setReadOnly(True)
        self.chain_view.setVisible(False)
        self.chain_cb.toggled.connect(lambda v: self.chain_view.setVisible(v))
        right_layout.addWidget(self.chain_view, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)

    def _init_session(self) -> None:
        greeting = self._new_session(self.mode_cb.currentText(), self._current_scenario_id())
        self.chat_view.append(greeting)
        self.status_lbl.setText("已就绪（离线 FakeLLM，无 key 也能跑）" if not self.engine.settings.llm_api_key else "已就绪")

    # ---- 会话管理 ----
    def _current_scenario_id(self) -> str:
        idx = self.scene_cb.currentIndex()
        return self.scene_cb.itemData(idx) or self.engine.settings.scenario_default

    def _new_session(self, mode: str, scenario_id: str) -> str:
        sid = f"ui-{int(time.time() * 1000)}"
        greeting = self.engine.start_session(mode=mode, scenario_id=scenario_id, sid=sid)
        self._active_sid = sid
        self.session_list.addItem(sid)
        self.session_list.setCurrentRow(self.session_list.count() - 1)
        return greeting

    def _on_new_session(self) -> None:
        greeting = self._new_session(self.mode_cb.currentText(), self._current_scenario_id())
        self.chat_view.append(f"\n——— 新会话 ———\n{greeting}\n")

    def _on_switch_session(self, sid: str) -> None:
        if sid and sid != self._active_sid:
            self.engine.switch_session(sid)
            self._active_sid = sid

    # ---- 对话 ----
    def _on_send(self) -> None:
        if self._worker is not None:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.chat_view.append(f"👤 {text}\n")
        self.input.clear()
        self._live_buffer = ""
        self.live_view.clear()
        self.send_btn.setEnabled(False)
        self._worker = StreamWorker(self.engine, text)
        self._worker.token_received.connect(self._on_token)
        self._worker.weights_updated.connect(self._refresh_weights)
        self._worker.persona_changed.connect(self._on_persona)
        self._worker.chain_step.connect(self._on_chain)
        self._worker.finished_turn.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_token(self, text: str) -> None:
        self._live_buffer += text
        self._flush_timer.start()  # 已在跑则重启计时 → 防抖

    def _flush_live(self) -> None:
        self.live_view.setPlainText(f"🤖 {self._live_buffer}")

    def _on_done(self, ev: DoneEvent) -> None:
        self._flush_timer.stop()
        self.chat_view.append(f"🤖 {ev.final_text}\n")
        self.live_view.clear()
        self._live_buffer = ""
        tag = []
        if ev.notify:
            tag.append("人格切换")
        if ev.rep_hit:
            tag.append("重复护栏触发")
        self.status_lbl.setText(f"第 {ev.turn} 轮 · " + (" · ".join(tag) if tag else "稳定"))

    def _on_persona(self, ev: PersonaChangeEvent) -> None:
        self.chat_view.append(f"（系统：我调整了一下节奏，Δ={ev.delta:.2f}）\n")

    def _on_chain(self, ev: ChainStepEvent) -> None:
        if self.chain_view.isVisible():
            self.chain_view.append(f"[{ev.step}] {ev.detail}\n")

    def _on_failed(self, msg: str) -> None:
        self.chat_view.append(f"（出错：{msg}）\n")

    def _on_worker_finished(self) -> None:
        self._worker = None
        self.send_btn.setEnabled(True)

    # ---- 权重面板（灵魂）----
    def _refresh_weights(self, snapshot) -> None:
        self.weight_tree.clear()
        layers = [
            ("L1 功能场景", snapshot.l1),
            ("L2 用户肖像", snapshot.l2),
            ("L3 Agent 人格", snapshot.l3),
        ]
        for label, weights in layers:
            if not weights:
                continue
            pitem = QTreeWidgetItem([label])
            self.weight_tree.addTopLevelItem(pitem)
            for k, v in sorted(weights.items(), key=lambda kv: -kv[1]):
                name = self._kw_name(k)
                child = QTreeWidgetItem([name, f"{v:.2f}"])
                pitem.addChild(child)
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(v * 100))
                bar.setFormat(f"{v:.2f}")
                self.weight_tree.setItemWidget(child, 1, bar)
            pitem.setExpanded(True)
        self.weight_tree.resizeColumnToContents(0)

    def _kw_name(self, key: str) -> str:
        kw = self.engine.lib.lexicon.get(key)
        return kw.name if kw else key

    # ---- 关闭：先停 worker，超时强退（doc/04 §4.5）----
    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
