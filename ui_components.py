from __future__ import annotations

import math
import random

from PySide6.QtCore import QEasingCurve, QPointF, Property, QPropertyAnimation, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

PURPLE = "#B34DFF"
PURPLE_SOFT = "#7937B0"
GREEN = "#79FF4D"
GREEN_SOFT = "#3E9D2C"
ORANGE = "#FF7A33"
TEXT = "#F4F6FC"
MUTED = "#939CAD"
BG = "#070910"
PANEL = "#0D111A"
PANEL_2 = "#111723"
BORDER = "#252D3D"
DANGER = "#FF5E68"


def add_glow(widget: QWidget, color: str, blur: float = 28, alpha: int = 80) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    shade = QColor(color)
    shade.setAlpha(alpha)
    effect.setColor(shade)
    effect.setBlurRadius(blur)
    effect.setOffset(0, 0)
    widget.setGraphicsEffect(effect)


def secondary_button_style(accent: str = PURPLE) -> str:
    return f"""
    QPushButton {{ color:{TEXT}; background:#0D131E; border:1px solid {accent};
        border-radius:10px; padding:9px 14px; font-size:13px; font-weight:700; }}
    QPushButton:hover {{ background:#171022; }}
    QPushButton:pressed {{ background:#211232; }}
    QPushButton:disabled {{ color:#626A79; border-color:#303747; background:#0A0D13; }}
    """


def slider_style(accent: str = PURPLE) -> str:
    return f"""
    QSlider::groove:horizontal {{ background:#1A2230; height:7px; border-radius:3px; }}
    QSlider::sub-page:horizontal {{ background:{accent}; border-radius:3px; }}
    QSlider::add-page:horizontal {{ background:#1A2230; border-radius:3px; }}
    QSlider::handle:horizontal {{ background:#F5F7FC; border:2px solid {accent}; width:17px;
        margin:-6px 0; border-radius:8px; }}
    """


class CinderBackdrop(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._clock = 0.0
        rng = random.Random(7047)
        self._embers = [[rng.random(), rng.random(), rng.uniform(0.001, 0.004), rng.uniform(1.0, 3.2)] for _ in range(34)]
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(50)
        self._timer = timer

    def _tick(self) -> None:
        self._clock += 0.05
        for ember in self._embers:
            ember[1] -= ember[2]
            ember[0] += math.sin(self._clock * 0.7 + ember[3]) * 0.0003
            if ember[1] < -0.05:
                ember[1] = 1.05
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG))
        purple = QRadialGradient(self.width() * 0.30, 0, self.width() * 0.58)
        purple.setColorAt(0, QColor(135, 35, 198, 62))
        purple.setColorAt(0.6, QColor(52, 16, 82, 18))
        purple.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), purple)
        green = QRadialGradient(self.width(), self.height() * 0.45, self.width() * 0.48)
        green.setColorAt(0, QColor(72, 178, 48, 24))
        green.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), green)
        for x, y, _speed, size in self._embers:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 105, 43, int(70 + 95 * (1 - max(0, min(1, y))))))
            painter.drawEllipse(QPointF(x * self.width(), y * self.height()), size, size)


class BackdropFrame(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.backdrop = CinderBackdrop(self)
        self.backdrop.lower()

    def resizeEvent(self, event) -> None:
        self.backdrop.setGeometry(self.rect())
        super().resizeEvent(event)


class FlameLogo(QWidget):
    def __init__(self, size: int = 58, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r = QRectF(7, 4, self.width() - 14, self.height() - 9)
        path = QPainterPath()
        path.moveTo(r.center().x(), r.bottom())
        path.cubicTo(r.left() - 2, r.bottom() - 13, r.left() + 3, r.center().y(), r.center().x() - 7, r.top() + 10)
        path.cubicTo(r.center().x() + 2, r.top() + 18, r.right() - 4, r.top() + 4, r.right() - 2, r.center().y())
        path.cubicTo(r.right(), r.bottom() - 11, r.center().x() + 10, r.bottom() - 4, r.center().x(), r.bottom())
        gradient = QLinearGradient(r.topLeft(), r.bottomRight())
        gradient.setColorAt(0, QColor(198, 82, 255))
        gradient.setColorAt(0.58, QColor(110, 36, 179))
        gradient.setColorAt(1, QColor(43, 14, 73))
        painter.setPen(QPen(QColor(219, 130, 255, 190), 1.2))
        painter.setBrush(gradient)
        painter.drawPath(path)
        inner = QPainterPath()
        inner.moveTo(r.center().x(), r.bottom() - 7)
        inner.cubicTo(r.center().x() - 10, r.bottom() - 14, r.center().x() - 2, r.center().y() + 4, r.center().x() + 2, r.top() + 19)
        inner.cubicTo(r.center().x() + 13, r.center().y() + 7, r.center().x() + 8, r.bottom() - 8, r.center().x(), r.bottom() - 7)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(29, 8, 50, 225))
        painter.drawPath(inner)


class GlowCard(QFrame):
    def __init__(self, title: str = "", accent: str = PURPLE, parent=None) -> None:
        super().__init__(parent)
        self.accent = QColor(accent)
        self.setObjectName("GlowCard")
        self.setStyleSheet(f"QFrame#GlowCard {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 {PANEL_2},stop:1 {PANEL}); border:1px solid {BORDER}; border-radius:17px; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(11)
        if title:
            header = QHBoxLayout()
            icon = QLabel("◆")
            icon.setStyleSheet(f"color:{accent}; border:none; font-size:12px;")
            label = QLabel(title.upper())
            label.setStyleSheet(f"color:{TEXT}; border:none; font-size:16px; font-weight:800; letter-spacing:1px;")
            header.addWidget(icon)
            header.addWidget(label)
            header.addStretch(1)
            layout.addLayout(header)
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        layout.addLayout(self.body)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(self.accent)
        color.setAlpha(105)
        painter.setPen(QPen(color, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 17, 17)


class DarkCombo(QComboBox):
    def __init__(self, values=(), parent=None) -> None:
        super().__init__(parent)
        popup = QListView(self)
        popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        popup.setAttribute(Qt.WA_TranslucentBackground)
        self.setView(popup)
        self.addItems(values)
        self.setMinimumHeight(41)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
        QComboBox {{ color:{TEXT}; background:#090E16; border:1px solid {BORDER}; border-radius:10px;
            padding:8px 36px 8px 11px; font-size:13px; }}
        QComboBox:hover, QComboBox:focus {{ border-color:{PURPLE}; background:#0E1320; }}
        QComboBox::drop-down {{ border:none; width:32px; }}
        QComboBox::down-arrow {{ image:none; width:8px; height:8px; border-right:2px solid #D8DEEA;
            border-bottom:2px solid #D8DEEA; margin-right:12px; }}
        QComboBox QAbstractItemView {{ color:{TEXT}; background:#0C111B; border:1px solid {PURPLE_SOFT};
            border-radius:9px; outline:0; padding:5px; selection-background-color:#301347; }}
        QComboBox QAbstractItemView::item {{ min-height:32px; padding:4px 9px; border-radius:6px; }}
        """)


class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, checked: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(52, 28)
        self.setCursor(Qt.PointingHandCursor)
        self._checked = checked
        self._position = 1.0 if checked else 0.0
        self._animation = QPropertyAnimation(self, b"position", self)
        self._animation.setDuration(180)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value: bool) -> None:
        value = bool(value)
        self._checked = value
        self._animation.stop()
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(1.0 if value else 0.0)
        self._animation.start()
        self.toggled.emit(value)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.setChecked(not self._checked)
            event.accept()

    def getPosition(self) -> float:
        return self._position

    def setPosition(self, value: float) -> None:
        self._position = float(value)
        self.update()

    position = Property(float, getPosition, setPosition)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        off = QColor("#202838")
        on = QColor(GREEN)
        red = int(off.red() + (on.red() - off.red()) * self._position)
        green = int(off.green() + (on.green() - off.green()) * self._position)
        blue = int(off.blue() + (on.blue() - off.blue()) * self._position)
        painter.setPen(QPen(QColor("#3B465A"), 1))
        painter.setBrush(QColor(red, green, blue))
        painter.drawRoundedRect(QRectF(1, 1, 50, 26), 13, 13)
        x = 4 + self._position * 24
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#F8FAFF"))
        painter.drawEllipse(QRectF(x, 4, 20, 20))


class SegmentMeter(QWidget):
    def __init__(self, accent: str = GREEN, minimum: float = -60.0, maximum: float = 0.0, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(18)
        self._accent = QColor(accent)
        self._minimum = minimum
        self._maximum = maximum
        self._value = minimum

    def setValue(self, value: float) -> None:
        self._value = max(self._minimum, min(self._maximum, float(value)))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        count, gap = 30, 3
        width = (self.width() - gap * (count - 1)) / count
        ratio = (self._value - self._minimum) / max(self._maximum - self._minimum, 1e-6)
        active = int(round(ratio * count))
        for index in range(count):
            color = QColor(self._accent if index < active else QColor("#202735"))
            if index >= int(count * 0.82) and index < active:
                color = QColor("#FFB33F" if index < int(count * 0.93) else "#FF5D67")
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(index * (width + gap), 1, width, self.height() - 2), 2, 2)


class RingGauge(QWidget):
    def __init__(self, accent: str = PURPLE, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(68, 68)
        self._value = 0.0
        self._accent = QColor(accent)

    def setValue(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(7, 7, self.width() - 14, self.height() - 14)
        painter.setPen(QPen(QColor("#252D3D"), 7, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 0, 360 * 16)
        painter.setPen(QPen(self._accent, 7, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 90 * 16, int(-360 * 16 * self._value))
        painter.setPen(QColor(TEXT))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self._value * 100:.0f}%")


class StatusPill(QFrame):
    def __init__(self, label: str, accent: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background:#0D121B; border:1px solid {BORDER}; border-radius:13px;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(13, 8, 13, 8)
        self.name_label = QLabel(label)
        self.name_label.setStyleSheet(f"color:{accent}; border:none; font-weight:900; font-size:13px;")
        self.detail_label = QLabel("Detecting…")
        self.detail_label.setStyleSheet(f"color:{TEXT}; border:none; font-size:12px;")
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(f"color:{MUTED}; border:none; font-size:12px;")
        layout.addWidget(self.name_label)
        layout.addWidget(self.detail_label)
        layout.addStretch(1)
        layout.addWidget(self.value_label)

    def setStatus(self, detail: str, value: str) -> None:
        self.detail_label.setText(detail)
        self.value_label.setText(value)


class SegmentedControl(QWidget):
    changed = Signal(str)

    def __init__(self, values, current: str, parent=None) -> None:
        super().__init__(parent)
        self.buttons: dict[str, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        for value in values:
            button = QPushButton(value)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, v=value: self.setValue(v))
            self.buttons[value] = button
            layout.addWidget(button)
        self.setStyleSheet(f"""
        QPushButton {{ color:{MUTED}; background:#0A0F18; border:1px solid {BORDER}; border-radius:9px;
            padding:8px 12px; font-size:12px; }}
        QPushButton:hover {{ color:{TEXT}; }}
        QPushButton:checked {{ color:{TEXT}; background:#271139; border-color:{PURPLE}; }}
        """)
        self.setValue(current, emit=False)

    def value(self) -> str:
        return next((name for name, button in self.buttons.items() if button.isChecked()), "")

    def setValue(self, value: str, emit: bool = True) -> None:
        if value not in self.buttons:
            return
        for name, button in self.buttons.items():
            button.setChecked(name == value)
        if emit:
            self.changed.emit(value)


class FadeStack(QStackedWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._animation = QPropertyAnimation(self._effect, b"opacity", self)
        self._animation.setDuration(170)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)
        self._target = 0

    def fadeTo(self, index: int) -> None:
        if index == self.currentIndex():
            return
        self._target = index
        self._animation.stop()
        try:
            self._animation.finished.disconnect()
        except RuntimeError:
            pass
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.0)
        self._animation.finished.connect(self._swap)
        self._animation.start()

    def _swap(self) -> None:
        try:
            self._animation.finished.disconnect()
        except RuntimeError:
            pass
        self.setCurrentIndex(self._target)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.start()


class TitleBar(QFrame):
    def __init__(self, host: QMainWindow, parent=None) -> None:
        super().__init__(parent)
        self.host = host
        self._drag_offset = None
        self.setFixedHeight(92)
        self.setStyleSheet("background:rgba(5,7,12,245); border:none; border-top-left-radius:21px; border-top-right-radius:21px;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(22, 12, 15, 10)
        logo = FlameLogo(58)
        title_box = QVBoxLayout()
        title = QLabel("Cinder<span style='color:#B34DFF'>Filter</span>")
        title.setTextFormat(Qt.RichText)
        title.setStyleSheet(f"color:{TEXT}; border:none; font-size:28px; font-weight:900;")
        subtitle = QLabel("Clean voice. Zero compromise.")
        subtitle.setStyleSheet(f"color:{MUTED}; border:none; font-size:11px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addWidget(logo)
        layout.addSpacing(7)
        layout.addLayout(title_box)
        layout.addStretch(1)
        self.gpu_pill = StatusPill("GPU", GREEN)
        self.cpu_pill = StatusPill("CPU", PURPLE)
        self.gpu_pill.setMinimumWidth(245)
        self.cpu_pill.setMinimumWidth(225)
        layout.addWidget(self.gpu_pill)
        layout.addWidget(self.cpu_pill)
        self.start_button = QPushButton("▶   START FILTER")
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setMinimumSize(205, 48)
        self.start_button.setStyleSheet(f"""
        QPushButton {{ color:{TEXT}; background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #153311,stop:1 #0D1710);
            border:1px solid {GREEN}; border-radius:14px; font-size:15px; font-weight:900; padding:10px 18px; }}
        QPushButton:hover {{ background:#173B16; }}
        QPushButton:disabled {{ color:#677064; border-color:#344534; background:#0C120C; }}
        """)
        add_glow(self.start_button, GREEN, 24, 95)
        layout.addWidget(self.start_button)
        for symbol, callback in (("—", host.showMinimized), ("▢", self.toggleMax), ("✕", host.close)):
            button = QPushButton(symbol)
            button.setFixedSize(34, 32)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(f"QPushButton {{ color:{TEXT}; background:transparent; border:none; border-radius:8px; font-size:15px; }} QPushButton:hover {{ background:#1B202C; }}")
            button.clicked.connect(callback)
            layout.addWidget(button)

    def setRunning(self, running: bool) -> None:
        self.start_button.setText("■   STOP FILTER" if running else "▶   START FILTER")

    def toggleMax(self) -> None:
        self.host.showNormal() if self.host.isMaximized() else self.host.showMaximized()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.toggleMax()
            event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and not self.host.isMaximized():
            self._drag_offset = event.globalPosition().toPoint() - self.host.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton and not self.host.isMaximized():
            self.host.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        event.accept()


class NavButton(QPushButton):
    def __init__(self, label: str, symbol: str, parent=None) -> None:
        super().__init__(f"{symbol}    {label}", parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(48)
        self.setStyleSheet(f"""
        QPushButton {{ color:#C7CDD9; text-align:left; background:transparent; border:1px solid transparent;
            border-radius:10px; padding:10px 13px; font-size:13px; }}
        QPushButton:hover {{ color:{TEXT}; background:#101622; }}
        QPushButton:checked {{ color:{TEXT}; background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2A123E,stop:1 #101522);
            border-left:2px solid {PURPLE}; }}
        """)
