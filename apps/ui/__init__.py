"""PyQt 桌面界面（doc/04 §4）。

三栏布局：会话列表 / 对话区 / 实时权重面板。通过 ``StreamWorker(QThread)`` 迭代
``DialogueEngine.stream_reply_sync`` 产出的 ``TurnEvent``，严格遵守「worker 只发信号、
主线程更新 UI」的 Qt 铁律（doc/04 §4.2）。
"""

from __future__ import annotations
