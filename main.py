import sys
from functools import partial

from PySide6.QtCore import QEasingCurve, QPoint, Property, QPropertyAnimation, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


ACCENT_PURPLE = "#B14CFF"
ACCENT_GREEN = "#78FF4D"
BG = "#090B12"
PANEL = "#101422"
PANEL_2 = "#0D111B"
TEXT = "#F4F7FF"
SUBTLE = "#98A2B3"
BORDER = "#232A3D"
TITLEBAR = "#06080E"


def glow(widget, color, blur=28, x=0, y=0, alpha=120):
    effect = QGraphicsDropShadowEffect(widget)
    c = QColor(color)
    c.setAlpha(alpha)
    effect.setColor(c)
    effect.setBlurRadius(blur)
    effect.setOffset(x, y)
    widget.setGraphicsEffect(effect)
    return effect


class NeonCard(QFrame):
    def __init__(self, title="", accent=ACCENT_PURPLE, parent=None):
        super().__init__(parent)
        self.accent = accent
        self.setObjectName("NeonCard")
        self.setStyleSheet(
            f"""
            QFrame#NeonCard {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 {PANEL}, stop:1 {PANEL_2});
                border: 1px solid {BORDER};
                border-radius: 20px;
            }}
            """
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(14)
        if title:
            top = QHBoxLayout()
            badge = QLabel("●")
            badge.setStyleSheet(f"color:{accent}; font-size:18px; border:none;")
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color:{TEXT}; font-size:22px; font-weight:700; border:none;")
            top.addWidget(badge)
            top.addWidget(lbl)
            top.addStretch(1)
            outer.addLayout(top)
        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        outer.addLayout(self.body)


class StatPill(QFrame):
    def __init__(self, name, detail, value, color):
        super().__init__()
        self.setStyleSheet(
            f"""
            background:{PANEL};
            border:1px solid {BORDER};
            border-radius:16px;
            """
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 12, 18, 12)
        dot = QLabel(name)
        dot.setStyleSheet(f"color:{color}; font-weight:800; font-size:18px;")
        det = QLabel(detail)
        det.setStyleSheet(f"color:{TEXT}; font-size:17px;")
        val = QLabel(value)
        val.setStyleSheet(f"color:{TEXT}; font-size:17px; font-weight:600;")
        lay.addWidget(dot)
        lay.addSpacing(8)
        lay.addWidget(det)
        lay.addStretch(1)
        lay.addWidget(val)


class ToggleSwitch(QCheckBox):
    def __init__(self, text=""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"""
            QCheckBox {{ color:{TEXT}; font-size:16px; spacing:12px; }}
            QCheckBox::indicator {{ width:54px; height:28px; }}
            QCheckBox::indicator:unchecked {{
                border-radius:14px; background:#1A2130; border:1px solid {BORDER};
            }}
            QCheckBox::indicator:checked {{
                border-radius:14px; background:{ACCENT_GREEN}; border:1px solid #7af35a;
            }}
            """
        )


class NeonCombo(QComboBox):
    def __init__(self, items=None):
        super().__init__()
        if items:
            self.addItems(items)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(46)
        self.setStyleSheet(
            f"""
            QComboBox {{
                background:{PANEL_2};
                color:{TEXT};
                border:1px solid {BORDER};
                border-radius:14px;
                padding:10px 42px 10px 14px;
                font-size:16px;
            }}
            QComboBox:hover {{ border:1px solid {ACCENT_PURPLE}; }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width:36px;
                border:none;
            }}
            QComboBox::down-arrow {{
                image:none;
                width:0; height:0;
                border-left:6px solid transparent;
                border-right:6px solid transparent;
                border-top:8px solid {TEXT};
                margin-right:14px;
            }}
            QComboBox QAbstractItemView {{
                background:{PANEL};
                color:{TEXT};
                selection-background-color:#291240;
                selection-color:{TEXT};
                border:1px solid {ACCENT_PURPLE};
                outline:none;
                padding:8px;
                border-radius:12px;
            }}
            QScrollBar:vertical {{
                background:{PANEL_2}; width:12px; border:none; border-radius:6px;
            }}
            QScrollBar::handle:vertical {{
                background:#3A445C; min-height:30px; border-radius:6px;
            }}
            """
        )


class MeterBar(QProgressBar):
    def __init__(self, accent=ACCENT_GREEN):
        super().__init__()
        self.setRange(0, 100)
        self.setValue(65)
        self.setTextVisible(False)
        self.setMinimumHeight(16)
        self.setStyleSheet(
            f"""
            QProgressBar {{
                background:#121826;
                border:1px solid {BORDER};
                border-radius:8px;
            }}
            QProgressBar::chunk {{
                border-radius:7px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {accent}, stop:1 {'#B14CFF' if accent == ACCENT_PURPLE else '#B7FF77'});
            }}
            """
        )


class SegmentedControl(QWidget):
    changed = Signal(str)

    def __init__(self, options, current):
        super().__init__()
        self.buttons = {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for option in options:
            btn = QPushButton(option)
            btn.setCheckable(True)
            btn.clicked.connect(partial(self.select, option))
            self.buttons[option] = btn
            lay.addWidget(btn)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background:{PANEL_2};
                color:{SUBTLE};
                border:1px solid {BORDER};
                border-radius:12px;
                padding:10px 18px;
                font-size:15px;
            }}
            QPushButton:checked {{
                color:{TEXT};
                border:1px solid {ACCENT_PURPLE};
                background:#251135;
            }}
            """
        )
        self.select(current)

    def select(self, option):
        for name, button in self.buttons.items():
            button.setChecked(name == option)
        self.changed.emit(option)


class NavButton(QPushButton):
    def __init__(self, text, active=False):
        super().__init__(text)
        self.setCheckable(True)
        self.setChecked(active)
        self.setMinimumHeight(52)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"""
            QPushButton {{
                text-align:left;
                color:{TEXT};
                background:transparent;
                border:1px solid transparent;
                border-radius:16px;
                padding:12px 16px;
                font-size:17px;
            }}
            QPushButton:hover {{ background:#121725; border:1px solid {BORDER}; }}
            QPushButton:checked {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #25103A, stop:1 #131927);
                border:1px solid {ACCENT_PURPLE};
            }}
            """
        )


class TitleBar(QFrame):
    def __init__(self, host_window: QMainWindow):
        super().__init__()
        if host_window is None:
            raise ValueError("TitleBar requires its owning QMainWindow")
        self.host_window = host_window
        self._drag_pos = None
        self.setObjectName("TitleBar")
        self.setStyleSheet(
            f"QFrame#TitleBar {{ background:{TITLEBAR}; border-top-left-radius:22px; border-top-right-radius:22px; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 12, 14, 10)

        flame = QLabel("◈")
        flame.setStyleSheet(f"color:{ACCENT_PURPLE}; font-size:24px; font-weight:900;")
        title = QLabel("CinderFilter")
        title.setStyleSheet(f"color:{TEXT}; font-size:20px; font-weight:800;")
        subtitle = QLabel("Clean voice. Zero compromise.")
        subtitle.setStyleSheet(f"color:{SUBTLE}; font-size:13px;")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        lay.addWidget(flame)
        lay.addSpacing(10)
        lay.addLayout(title_box)
        lay.addStretch(1)

        for label, fn in [
            ("—", self.host_window.showMinimized),
            ("▢", self.toggle_max_restore),
            ("✕", self.host_window.close),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(fn)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedSize(40, 34)
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background:transparent; color:{TEXT}; border:none; border-radius:10px; font-size:18px;
                }}
                QPushButton:hover {{ background:#1A1F2C; }}
                """
            )
            lay.addWidget(btn)

    def toggle_max_restore(self):
        if self.host_window.isMaximized():
            self.host_window.showNormal()
        else:
            self.host_window.showMaximized()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_max_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.host_window.isMaximized():
                event.accept()
                return
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.host_window.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_pos is not None
            and event.buttons() & Qt.LeftButton
            and not self.host_window.isMaximized()
        ):
            self.host_window.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()


class MainView(QWidget):
    def __init__(self, host_window: QMainWindow):
        super().__init__()
        self.host_window = host_window
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"background:{BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.titlebar = TitleBar(self.host_window)
        root.addWidget(self.titlebar)

        content = QHBoxLayout()
        content.setContentsMargins(10, 10, 10, 10)
        content.setSpacing(14)
        root.addLayout(content)

        content.addWidget(self.build_sidebar(), 0)
        main_container = QVBoxLayout()
        main_container.setSpacing(14)
        content.addLayout(main_container, 1)

        main_container.addLayout(self.build_header())
        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.stack.setStyleSheet("QStackedWidget { background: transparent; }")
        pages = [self.build_overview_page(), self.simple_page("Devices"), self.simple_page("Voice"), self.simple_page("Tuning"), self.simple_page("Advanced"), self.simple_page("Diagnostics"), self.simple_page("About")]
        for p in pages:
            self.stack.addWidget(p)
        main_container.addWidget(self.stack, 1)

        self.fade_effect = QGraphicsOpacityEffect(self.stack)
        self.fade_effect.setOpacity(1.0)
        self.stack.setGraphicsEffect(self.fade_effect)
        self.fade_anim = QPropertyAnimation(self.fade_effect, b"opacity", self)
        self.fade_anim.setDuration(220)
        self.fade_anim.setEasingCurve(QEasingCurve.InOutQuad)


    def build_sidebar(self):
        wrap = QFrame()
        wrap.setFixedWidth(240)
        wrap.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0B0E16, stop:1 #090B12); border:1px solid {BORDER}; border-radius:24px;"
        )
        glow(wrap, ACCENT_PURPLE, blur=40, alpha=45)
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(14, 18, 14, 18)
        lay.setSpacing(8)

        self.nav_buttons = []
        names = ["Overview", "Devices", "Voice", "Tuning", "Advanced", "Diagnostics", "About"]
        for i, name in enumerate(names):
            btn = NavButton(name, active=(i == 0))
            btn.clicked.connect(partial(self.change_page, i))
            self.nav_buttons.append(btn)
            lay.addWidget(btn)
        lay.addStretch(1)
        cinder = QLabel("Cinder stream theme\nframeless • animated • neon")
        cinder.setStyleSheet(f"color:{SUBTLE}; font-size:13px; border:none; line-height:1.3;")
        lay.addWidget(cinder)
        version = QLabel("vNext UI")
        version.setStyleSheet(f"color:{ACCENT_PURPLE}; font-size:12px; border:none;")
        lay.addWidget(version)
        return wrap

    def build_header(self):
        lay = QHBoxLayout()
        lay.setSpacing(12)
        title = QLabel("CinderFilter Control Deck")
        title.setStyleSheet(f"color:{TEXT}; font-size:32px; font-weight:800;")
        lay.addWidget(title)
        lay.addStretch(1)
        lay.addWidget(StatPill("GPU", "RTX 4070", "Ready", ACCENT_GREEN))
        lay.addWidget(StatPill("CPU", "Main thread", "12%", ACCENT_PURPLE))
        start = QPushButton("▶  START FILTER")
        start.setCursor(Qt.PointingHandCursor)
        start.setMinimumHeight(54)
        start.setStyleSheet(
            f"""
            QPushButton {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #173313, stop:1 #0F1C10);
                color:{TEXT};
                border:1px solid {ACCENT_GREEN};
                border-radius:18px;
                padding:12px 22px;
                font-size:18px;
                font-weight:800;
            }}
            QPushButton:hover {{ background:#163418; }}
            """
        )
        glow(start, ACCENT_GREEN, blur=24, alpha=110)
        lay.addWidget(start)
        return lay

    def simple_page(self, name):
        page = QWidget()
        lay = QVBoxLayout(page)
        card = NeonCard(name, ACCENT_PURPLE)
        text = QLabel(f"{name} page placeholder for backend controls.\nThe shell, theme, transitions, and dark-mode components are live and ready for backend wiring.")
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{TEXT}; font-size:18px;")
        card.body.addWidget(text)
        lay.addWidget(card)
        lay.addStretch(1)
        return page

    def build_overview_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(14)

        routing = NeonCard("Routing", ACCENT_PURPLE)
        top = QGridLayout()
        top.setHorizontalSpacing(18)
        top.setVerticalSpacing(12)
        top.addWidget(self.device_block("Input Device", ["Microphone (Shure SM7B)", "Logitech Pro X Wireless", "VoiceMeeter Output"], "-12.4 dB"), 0, 0)
        center = QLabel("⟷")
        center.setAlignment(Qt.AlignCenter)
        center.setStyleSheet(f"color:{ACCENT_PURPLE}; font-size:32px; font-weight:800;")
        top.addWidget(center, 0, 1)
        top.addWidget(self.device_block("Output Device", ["VoiceMeeter AUX Input", "CABLE Input", "Voicemod Input"], "-6.1 dB"), 0, 2)
        top.setColumnStretch(0, 1)
        top.setColumnStretch(2, 1)
        routing.body.addLayout(top)
        lay.addWidget(routing)

        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(self.main_noise_card())
        row.addWidget(self.voice_lock_card())
        row.addWidget(self.pitch_lock_card())
        lay.addLayout(row)

        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        bottom.addWidget(self.meter_card("Input Level", "-12.4 dB", ACCENT_GREEN))
        bottom.addWidget(self.meter_card("Output Level", "-6.1 dB", ACCENT_GREEN))
        bottom.addWidget(self.meter_card("Voice Match Confidence", "92%", ACCENT_PURPLE))
        bottom.addWidget(self.meter_card("Noise Reduction", "-28 dB", ACCENT_GREEN))
        lay.addLayout(bottom)

        diag = NeonCard("Diagnostics", ACCENT_PURPLE)
        grid = QGridLayout()
        grid.setHorizontalSpacing(22)
        items = [
            ("GPU Detected", "NVIDIA RTX 4070"),
            ("Backend", "CUDA"),
            ("RTF", "0.42x"),
            ("Queue Depth", "128 / 512"),
            ("Sample Rate", "48 kHz"),
            ("Block Size", "480 samples (10 ms)"),
        ]
        for idx, (k, v) in enumerate(items):
            item = QLabel(f"<span style='color:{SUBTLE}; font-size:13pt'>{k}</span><br><span style='color:{TEXT}; font-size:16pt; font-weight:700'>{v}</span>")
            grid.addWidget(item, 0, idx)
        diag.body.addLayout(grid)
        lay.addWidget(diag)
        lay.addStretch(1)
        return page

    def device_block(self, title, items, db):
        block = QFrame()
        block.setStyleSheet(f"background:{PANEL_2}; border:1px solid {BORDER}; border-radius:16px;")
        lay = QVBoxLayout(block)
        lay.setContentsMargins(14, 14, 14, 14)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color:{TEXT}; font-size:17px; font-weight:600;")
        lay.addWidget(title_lbl)
        combo = NeonCombo(items)
        lay.addWidget(combo)
        meter_row = QHBoxLayout()
        meter_row.addWidget(MeterBar(ACCENT_GREEN), 1)
        db_lbl = QLabel(db)
        db_lbl.setStyleSheet(f"color:{ACCENT_PURPLE}; font-size:15px; font-weight:700;")
        meter_row.addWidget(db_lbl)
        lay.addLayout(meter_row)
        return block

    def main_noise_card(self):
        card = NeonCard("Main Noise Reducer", ACCENT_GREEN)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card.body.addWidget(self.field_label("Noise Engine"))
        combo = NeonCombo(["CUDA (NVIDIA)", "Auto", "CPU"])
        card.body.addWidget(combo)
        card.body.addWidget(self.field_label("CUDA Latency Preset"))
        combo2 = NeonCombo(["Balanced (Recommended)", "Low Latency", "Quality"])
        card.body.addWidget(combo2)
        status = QLabel("CUDA engine active and ready\nUsing NVIDIA RTX 4070")
        status.setStyleSheet(f"background:#102015; color:{ACCENT_GREEN}; border:1px solid #26452A; border-radius:14px; padding:14px; font-size:15px; font-weight:600;")
        card.body.addWidget(status)
        return card

    def voice_lock_card(self):
        card = NeonCard("Voice Lock", ACCENT_PURPLE)
        card.body.addWidget(self.field_label("Voice Profile"))
        combo = NeonCombo(["My Voice Profile", "Streamer Profile", "Alt Profile"])
        card.body.addWidget(combo)
        card.body.addWidget(self.field_label("Other-Voice Reduction"))
        slider = QSlider(Qt.Horizontal)
        slider.setValue(78)
        slider.setStyleSheet(self.slider_style(ACCENT_PURPLE))
        card.body.addWidget(slider)
        seg = SegmentedControl(["Relaxed", "Balanced", "Strict"], "Balanced")
        card.body.addWidget(seg)
        conf = QLabel("Confidence 92%")
        conf.setStyleSheet(f"color:{TEXT}; font-size:16px; font-weight:700;")
        card.body.addWidget(conf)
        return card

    def pitch_lock_card(self):
        card = NeonCard("Pitch Lock", ACCENT_GREEN)
        card.body.addWidget(ToggleSwitch("Pitch distinction"))
        fail = ToggleSwitch("Fail-Closed (Block on failure)")
        fail.setChecked(True)
        card.body.addWidget(fail)
        card.body.addWidget(self.field_label("Pitch Margin"))
        slider = QSlider(Qt.Horizontal)
        slider.setValue(40)
        slider.setStyleSheet(self.slider_style(ACCENT_GREEN))
        card.body.addWidget(slider)
        status = QLabel("Pitch lock engaged")
        status.setStyleSheet(f"background:#102015; color:{ACCENT_GREEN}; border:1px solid #26452A; border-radius:14px; padding:14px; font-size:15px; font-weight:700;")
        card.body.addWidget(status)
        return card

    def slider_style(self, color):
        return f"""
            QSlider::groove:horizontal {{ background:#1A2230; height:8px; border-radius:4px; }}
            QSlider::sub-page:horizontal {{ background:{color}; border-radius:4px; }}
            QSlider::add-page:horizontal {{ background:#1A2230; border-radius:4px; }}
            QSlider::handle:horizontal {{ background:#F3F7FF; border:2px solid {color}; width:18px; margin:-6px 0; border-radius:9px; }}
        """

    def meter_card(self, title, value, accent):
        card = NeonCard(title, accent)
        meter = MeterBar(accent)
        card.body.addWidget(meter)
        val = QLabel(value)
        val.setStyleSheet(f"color:{accent}; font-size:18px; font-weight:800;")
        card.body.addWidget(val)
        return card

    def field_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{SUBTLE}; font-size:14px; font-weight:600;")
        return lbl

    def change_page(self, index):
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self.fade_anim.stop()
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        try:
            self.fade_anim.finished.disconnect()
        except Exception:
            pass

        def switch_in():
            self.stack.setCurrentIndex(index)
            self.fade_anim.finished.disconnect()
            self.fade_anim.setStartValue(0.0)
            self.fade_anim.setEndValue(1.0)
            self.fade_anim.finished.connect(lambda: None)
            self.fade_anim.start()

        self.fade_anim.finished.connect(switch_in)
        self.fade_anim.start()


class CinderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(1680, 1040)
        self.central = QWidget()
        self.central.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self.central)
        outer.setContentsMargins(12, 12, 12, 12)
        self.shell = QFrame()
        self.shell.setStyleSheet(
            f"background:{BG}; border:1px solid {BORDER}; border-radius:26px;"
        )
        glow(self.shell, ACCENT_PURPLE, blur=50, alpha=50)
        shell_lay = QVBoxLayout(self.shell)
        shell_lay.setContentsMargins(0, 0, 0, 0)
        self.main_view = MainView(self)
        shell_lay.addWidget(self.main_view)
        outer.addWidget(self.shell)
        self.setCentralWidget(self.central)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    win = CinderWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
