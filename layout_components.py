from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QSize, Qt, QPropertyAnimation, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsOpacityEffect,
    QGridLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)


MAX_WIDGET = 16_777_215


class FillFadeStack(QStackedWidget):
    """Stack host with page-local fades and no stack-wide graphics effect."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(MAX_WIDGET, MAX_WIDGET)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.currentChanged.connect(self._schedule_fit)
        self._transition: QPropertyAnimation | None = None
        self._transition_effect: QGraphicsOpacityEffect | None = None
        self._transition_page: QWidget | None = None
        self._target_index = -1

    def addWidget(self, widget: QWidget) -> int:  # noqa: N802 - Qt API
        widget.setMinimumSize(0, 0)
        widget.setMaximumSize(MAX_WIDGET, MAX_WIDGET)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        index = super().addWidget(widget)
        self._schedule_fit()
        return index

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def _schedule_fit(self, *_args) -> None:
        QTimer.singleShot(0, self.fit_current)

    def fit_current(self) -> None:
        current = self.currentWidget()
        if current is None:
            return
        current.setMinimumSize(0, 0)
        current.setMaximumSize(MAX_WIDGET, MAX_WIDGET)
        current.setGeometry(self.contentsRect())
        current.updateGeometry()

    def _detach_transition_effect(self) -> None:
        animation = self._transition
        self._transition = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()

        page = self._transition_page
        effect = self._transition_effect
        self._transition_page = None
        self._transition_effect = None
        if page is not None and effect is not None and page.graphicsEffect() is effect:
            page.setGraphicsEffect(None)

    def _animate_page(
        self,
        page: QWidget,
        start: float,
        end: float,
        finished,
    ) -> None:
        self._detach_transition_effect()
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(start)
        page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(105)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.finished.connect(finished)
        self._transition_page = page
        self._transition_effect = effect
        self._transition = animation
        animation.start()

    def fadeTo(self, index: int) -> None:  # noqa: N802 - existing app API
        if index < 0 or index >= self.count() or index == self.currentIndex():
            return
        current = self.currentWidget()
        if current is None:
            self.setCurrentIndex(index)
            self.fit_current()
            return

        self._target_index = index
        self._animate_page(current, 1.0, 0.0, self._finish_fade_out)

    def _finish_fade_out(self) -> None:
        target = self._target_index
        self._detach_transition_effect()
        if target < 0 or target >= self.count():
            return
        self.setCurrentIndex(target)
        self.fit_current()
        incoming = self.currentWidget()
        if incoming is None:
            return
        self._animate_page(incoming, 0.0, 1.0, self._finish_fade_in)

    def _finish_fade_in(self) -> None:
        self._detach_transition_effect()
        self.fit_current()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.fit_current()


class ViewportLockedScrollArea(QScrollArea):
    """Vertical-only scroll area whose page is always anchored at top-left."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(MAX_WIDGET, MAX_WIDGET)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.verticalScrollBar().rangeChanged.connect(self._schedule_geometry_sync)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt API
        widget.setMinimumSize(0, 0)
        widget.setMaximumSize(MAX_WIDGET, MAX_WIDGET)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        super().setWidget(widget)
        self._schedule_geometry_sync()

    def _schedule_geometry_sync(self, *_args) -> None:
        QTimer.singleShot(0, self.sync_page_geometry)

    def sync_page_geometry(self) -> None:
        page = self.widget()
        if page is None:
            return

        viewport = self.viewport()
        target_width = max(1, viewport.width())

        page.setMinimumWidth(0)
        page.setMaximumWidth(MAX_WIDGET)
        page.resize(target_width, max(1, page.sizeHint().height()))
        if page.layout() is not None:
            page.layout().activate()

        target_height = max(viewport.height(), page.sizeHint().height())
        page.resize(target_width, max(1, target_height))

        self.horizontalScrollBar().setRange(0, 0)
        self.horizontalScrollBar().setValue(0)
        if self.verticalScrollBar().value() == 0:
            page.move(0, 0)
        else:
            page.move(0, -self.verticalScrollBar().value())
        page.updateGeometry()
        viewport.update()

    def sync_page_width(self) -> None:
        self.sync_page_geometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.sync_page_geometry()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._schedule_geometry_sync()


class ReflowGrid(QWidget):
    """Width-flexible grid that changes columns instead of overflowing."""

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
        self.setMaximumWidth(MAX_WIDGET)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        for widget in self._widgets:
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(MAX_WIDGET)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            for combo in widget.findChildren(QComboBox):
                combo.setMinimumWidth(0)
                combo.setMaximumWidth(MAX_WIDGET)
                combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            for button in widget.findChildren(QPushButton):
                button.setMinimumWidth(0)
                button.setMaximumWidth(MAX_WIDGET)

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
        self.setMaximumWidth(MAX_WIDGET)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        for block in (self._input, self._output):
            block.setMinimumWidth(0)
            block.setMaximumWidth(MAX_WIDGET)
            block.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
