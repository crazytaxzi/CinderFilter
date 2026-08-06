from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from ui_components import FadeStack


class FillFadeStack(FadeStack):
    """A stacked page host that ignores hidden-page width hints and fills its slot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.currentChanged.connect(self._schedule_fit)

    def addWidget(self, widget: QWidget) -> int:  # noqa: N802 - Qt API
        widget.setMinimumSize(0, 0)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        index = super().addWidget(widget)
        self._schedule_fit()
        return index

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        current = self.currentWidget()
        if current is None:
            return QSize(0, 0)
        hint = current.sizeHint()
        return QSize(0, max(0, hint.height()))

    def _schedule_fit(self, *_args) -> None:
        QTimer.singleShot(0, self.fit_current)

    def fit_current(self) -> None:
        current = self.currentWidget()
        if current is None:
            return
        current.setMinimumWidth(0)
        current.setMaximumWidth(max(1, self.width()))
        current.setGeometry(self.rect())
        current.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.fit_current()


class ViewportLockedScrollArea(QScrollArea):
    """A vertical scroll area whose page always equals the viewport width."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.verticalScrollBar().rangeChanged.connect(self._schedule_width_sync)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt API
        widget.setMinimumSize(0, 0)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.MinimumExpanding)
        super().setWidget(widget)
        self._schedule_width_sync()

    def _schedule_width_sync(self, *_args) -> None:
        QTimer.singleShot(0, self.sync_page_width)

    def sync_page_width(self) -> None:
        page = self.widget()
        if page is None:
            return
        target = max(1, self.viewport().width())
        page.setMinimumWidth(target)
        page.setMaximumWidth(target)
        page.resize(target, max(page.height(), page.sizeHint().height()))
        page.move(0, 0)
        self.horizontalScrollBar().setValue(0)
        page.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.sync_page_width()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._schedule_width_sync()


class ReflowGrid(QWidget):
    """A width-flexible grid that changes columns instead of overflowing."""

    def __init__(
        self,
        widgets: list[QWidget],
        breakpoints: tuple[tuple[int, int], ...],
        spacing: int = 12,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._widgets = widgets
        self._breakpoints = tuple(sorted(breakpoints, reverse=True))
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(spacing)
        self._grid.setVerticalSpacing(spacing)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)

        for widget in self._widgets:
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16_777_215)
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            for combo in widget.findChildren(QComboBox):
                combo.setMinimumWidth(0)
                combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            for button in widget.findChildren(QPushButton):
                button.setMinimumWidth(0)

        QTimer.singleShot(0, self._reflow)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def _column_count(self) -> int:
        width = max(0, self.contentsRect().width())
        for minimum_width, columns in self._breakpoints:
            if width >= minimum_width:
                return columns
        return 1

    def _reflow(self) -> None:
        columns = self._column_count()
        if columns == self._columns:
            return
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        for column in range(8):
            self._grid.setColumnStretch(column, 0)
        for column in range(max(columns, 1)):
            self._grid.setColumnStretch(column, 1)
        for index, widget in enumerate(self._widgets):
            row, column = divmod(index, columns)
            self._grid.addWidget(widget, row, column)
        self._grid.invalidate()
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._reflow()


class RouteReflow(QWidget):
    """Two flexible route cards with a compact center signal indicator."""

    def __init__(
        self,
        input_block: QWidget,
        flow: QWidget,
        output_block: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._input = input_block
        self._flow = flow
        self._output = output_block
        self._wide: bool | None = None
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(8)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)

        for block in (self._input, self._output):
            block.setMinimumWidth(0)
            block.setMaximumWidth(16_777_215)
            block.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._flow.setMinimumWidth(70)
        self._flow.setMaximumWidth(120)
        self._flow.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        QTimer.singleShot(0, self._reflow)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def _reflow(self) -> None:
        wide = self.contentsRect().width() >= 760
        if wide == self._wide:
            return
        self._wide = wide
        while self._grid.count():
            self._grid.takeAt(0)
        for column in range(3):
            self._grid.setColumnStretch(column, 0)
        if wide:
            self._grid.addWidget(self._input, 0, 0)
            self._grid.addWidget(self._flow, 0, 1)
            self._grid.addWidget(self._output, 0, 2)
            self._grid.setColumnStretch(0, 1)
            self._grid.setColumnStretch(2, 1)
        else:
            self._grid.addWidget(self._input, 0, 0)
            self._grid.addWidget(self._flow, 1, 0, alignment=Qt.AlignCenter)
            self._grid.addWidget(self._output, 2, 0)
            self._grid.setColumnStretch(0, 1)
        self._grid.invalidate()
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._reflow()
