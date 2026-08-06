from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from layout_components import FillFadeStack, ViewportLockedScrollArea


def _process(app: QApplication, count: int = 8) -> None:
    for _ in range(count):
        app.processEvents()


def test_hidden_page_cannot_push_current_page_outside_window() -> None:
    app = QApplication.instance() or QApplication([])

    root = QWidget()
    root.resize(1450, 820)
    row = QHBoxLayout(root)
    row.setContentsMargins(10, 0, 10, 10)
    row.setSpacing(12)

    sidebar = QFrame()
    sidebar.setFixedWidth(176)
    stack = FillFadeStack()

    scroll = ViewportLockedScrollArea()
    page = QWidget()
    page.setMinimumSize(0, 0)
    page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.MinimumExpanding)
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(6, 6, 8, 8)
    deliberately_wide_card = QFrame()
    deliberately_wide_card.setMinimumWidth(1900)
    deliberately_wide_card.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    page_layout.addWidget(deliberately_wide_card)
    scroll.setWidget(page)

    hidden_page = QWidget()
    hidden_page.setMinimumWidth(2500)

    stack.addWidget(scroll)
    stack.addWidget(hidden_page)
    stack.setCurrentIndex(0)
    row.addWidget(sidebar, 0)
    row.addWidget(stack, 1)
    row.setStretch(0, 0)
    row.setStretch(1, 1)

    root.show()
    _process(app)
    stack.fit_current()
    scroll.sync_page_width()
    _process(app)

    margins = row.contentsMargins()
    expected = (
        root.contentsRect().width()
        - margins.left()
        - margins.right()
        - sidebar.width()
        - row.spacing()
    )

    assert stack.x() == margins.left() + sidebar.width() + row.spacing()
    assert stack.width() == expected
    assert stack.geometry().right() <= root.contentsRect().right() - margins.right()
    assert scroll.geometry() == stack.rect()
    assert page.x() == 0
    assert page.width() == scroll.viewport().width()
    assert scroll.horizontalScrollBar().value() == 0

    root.close()
