"""One place that decides what the application looks like.

A small set of design tokens, one stylesheet generated from them, applied once
to the ``QApplication``. Widgets ask for a *role* (``muted``, ``danger``,
``heading``) rather than a colour, so a theme change is a token change and
nothing else; never give a widget an inline ``setStyleSheet`` with hardcoded
hex. ``ui.theme`` picks ``dark`` (default), ``light``, or ``system``, which
follows the desktop's own preference.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette, QPen
from qtpy.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

__all__ = ["THEMES", "apply_theme", "tokens", "role", "Heading", "current",
           "theme_icon", "app_icon", "banner_style", "compact_button", "set_role", "set_flag",
           "RADIUS"]

# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #
DARK = {
    "bg": "#15171C",          # window behind everything
    "surface": "#1C1F26",     # cards, group boxes, panels
    "raised": "#232730",      # inputs, table rows, hover
    "raised_hi": "#2B303A",   # pressed / stronger hover
    "border": "#2E333D",
    "border_hi": "#3C434F",
    "text": "#E8EAEE",
    "muted": "#9BA2AE",
    "faint": "#6C737F",
    "accent": "#4C8DFF",
    "accent_hi": "#6BA1FF",
    "accent_soft": "#22304A",  # tinted background for a selected row
    "on_accent": "#0B1220",
    "success": "#4FB477",
    "warning": "#E0A458",
    "danger": "#E5484D",
    "canvas": "#101216",      # behind the clip, deliberately darker than bg
}

LIGHT = {
    "bg": "#F4F5F8",
    "surface": "#FFFFFF",
    "raised": "#EDEFF3",
    "raised_hi": "#E2E6EC",
    "border": "#D9DDE4",
    "border_hi": "#C2C8D2",
    "text": "#1A1D23",
    "muted": "#5C6470",
    "faint": "#8A919C",
    "accent": "#2F6FE4",
    "accent_hi": "#1F5BC9",
    "accent_soft": "#E4EDFD",
    "on_accent": "#FFFFFF",
    "success": "#2F855A",
    "warning": "#B4700F",
    "danger": "#C4292E",
    "canvas": "#20242B",
}

THEMES = {"dark": DARK, "light": LIGHT}

RADIUS = 6
_active: dict = dict(DARK)
_active_name = "dark"


def tokens() -> dict:
    """The palette currently in force. Underscore keys are generated assets
    (arrow images), not colours; the stylesheet interpolates them in one pass.
    """
    return _active


def current() -> str:
    return _active_name


def role(name: str) -> str:
    """A token by name, for the few places that still need a literal colour."""
    return _active.get(name, _active["text"])


# --------------------------------------------------------------------------- #
# stylesheet
# --------------------------------------------------------------------------- #
def _stylesheet(t: dict) -> str:
    return f"""
* {{
    outline: none;
}}
QWidget {{
    background: {t['bg']};
    color: {t['text']};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {t['bg']}; }}

QLabel {{ background: transparent; }}
QLabel[role="muted"]   {{ color: {t['muted']}; }}
QLabel[role="faint"]   {{ color: {t['faint']}; }}
QLabel[role="danger"]  {{ color: {t['danger']}; }}
QLabel[role="warning"] {{ color: {t['warning']}; }}
QLabel[role="success"] {{ color: {t['success']}; }}
QLabel[role="heading"] {{
    color: {t['text']};
    font-size: 19px;
    font-weight: 600;
}}
QLabel[role="subheading"] {{
    color: {t['muted']};
    font-size: 13px;
}}

/* -- cards ------------------------------------------------------------- */
/* The title sits INSIDE the card. With the Qt default (subcontrol-origin:
   margin) it is drawn above the border, which on a filled card reads as a
   stray label floating over the page rather than as the card's header. */
QGroupBox {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {RADIUS + 2}px;
    margin-top: 0;
    padding: 34px 14px 14px 14px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: padding;
    subcontrol-position: top left;
    padding: 10px 14px 0 14px;
    color: {t['muted']};
    font-weight: 600;
}}
QGroupBox::indicator {{ width: 15px; height: 15px; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{   /* HLine / VLine */
    background: {t['border']};
    border: none;
    max-height: 1px;
}}

/* -- buttons ----------------------------------------------------------- */
QPushButton {{
    background: {t['raised']};
    color: {t['text']};
    border: 1px solid {t['border_hi']};
    border-radius: {RADIUS}px;
    padding: 7px 14px;
    font-weight: 500;
}}
QPushButton:hover  {{ background: {t['raised_hi']}; border-color: {t['accent']}; }}
QPushButton:pressed {{ background: {t['border']}; }}
QPushButton:disabled {{
    background: {t['surface']};
    color: {t['faint']};
    border-color: {t['border']};
}}
QPushButton:default, QPushButton[accent="true"] {{
    background: {t['accent']};
    color: {t['on_accent']};
    border-color: {t['accent']};
    font-weight: 600;
}}
QPushButton:default:hover, QPushButton[accent="true"]:hover {{
    background: {t['accent_hi']}; border-color: {t['accent_hi']};
}}
QPushButton[accent="true"]:disabled {{
    background: {t['surface']}; color: {t['faint']}; border-color: {t['border']};
}}
/* Small square buttons. The default 14px horizontal padding is wider than the
   whole control on a 26px button, so the label had no room at all and simply
   did not draw: the "?" help button and every "x" delete mark were blank. */
QPushButton[compact="true"] {{
    padding: 0;
    font-weight: 700;
}}
QPushButton[compact="true"][danger="true"] {{ color: {t['danger']}; }}
QPushButton[compact="true"][danger="true"]:hover {{
    background: {t['danger']}; color: {t['on_accent']}; border-color: {t['danger']};
}}

/* -- text inputs ------------------------------------------------------- */
QLineEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t['raised']};
    color: {t['text']};
    border: 1px solid {t['border_hi']};
    border-radius: {RADIUS}px;
    padding: 6px 8px;
    selection-background-color: {t['accent']};
    selection-color: {t['on_accent']};
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {t['accent']};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {t['surface']}; color: {t['faint']};
}}
QLineEdit[invalid="true"], QSpinBox[invalid="true"],
QDoubleSpinBox[invalid="true"], QComboBox[invalid="true"] {{
    border: 1px solid {t['warning']};
}}
/* Real images, not CSS border triangles. The border trick renders as two flat
   lines here because the subcontrol keeps a non-zero content box, and once any
   subcontrol is styled Qt stops asking the platform style to paint the native
   arrow - so leaving them alone does not work either. These are painted once
   per theme by _arrow_images(). */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 18px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {t['raised_hi']};
    border-radius: 3px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{t['_arrow_up']}");
    width: 9px; height: 6px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{t['_arrow_down']}");
    width: 9px; height: 6px;
}}
QSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    image: url("{t['_arrow_faint']}");
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: url("{t['_arrow_down']}");
    width: 9px; height: 6px;
    margin-right: 8px;
}}
QHeaderView::down-arrow {{ image: url("{t['_arrow_down']}"); width: 9px; height: 6px; }}
QHeaderView::up-arrow {{ image: url("{t['_arrow_up']}"); width: 9px; height: 6px; }}
QComboBox QAbstractItemView {{
    background: {t['raised']};
    color: {t['text']};
    border: 1px solid {t['border_hi']};
    border-radius: {RADIUS}px;
    selection-background-color: {t['accent']};
    selection-color: {t['on_accent']};
    padding: 4px;
}}

/* -- checkboxes -------------------------------------------------------- */
QCheckBox {{ background: transparent; spacing: 8px; padding: 2px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {t['border_hi']};
    border-radius: 4px;
    background: {t['raised']};
}}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked {{
    background: {t['accent']};
    border-color: {t['accent']};
}}
QCheckBox:disabled {{ color: {t['faint']}; }}

/* -- tabs -------------------------------------------------------------- */
QTabWidget::pane {{
    border: 1px solid {t['border']};
    border-radius: {RADIUS + 2}px;
    background: {t['surface']};
    top: -1px;
}}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    color: {t['muted']};
    border: 1px solid transparent;
    border-top-left-radius: {RADIUS}px;
    border-top-right-radius: {RADIUS}px;
    padding: 8px 16px;
    margin-right: 2px;
    font-weight: 500;
}}
QTabBar::tab:hover {{ color: {t['text']}; }}
QTabBar::tab:selected {{
    background: {t['surface']};
    color: {t['text']};
    border-color: {t['border']};
    border-bottom-color: {t['surface']};
    font-weight: 600;
}}
QTabBar::tab:disabled {{ color: {t['faint']}; }}

/* -- lists and tables -------------------------------------------------- */
QListWidget, QTableWidget, QTreeWidget {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {RADIUS}px;
    alternate-background-color: {t['raised']};
}}
QListWidget::item {{
    padding: 7px 8px;
    border-radius: {RADIUS - 2}px;
    color: {t['text']};
}}
QListWidget::item:hover {{ background: {t['raised']}; }}
QListWidget::item:selected {{ background: {t['accent_soft']}; color: {t['text']}; }}
QListWidget::item:disabled {{ color: {t['faint']}; }}
QTableWidget {{ gridline-color: {t['border']}; }}
/* Horizontal only. Vertical item padding is taken out of the cell rect, so a
   cell WIDGET (a combo in the skeleton table) gets squeezed below its own
   sizeHint and has its descenders clipped. */
QTableWidget::item {{ padding: 0 6px; }}
QTableWidget::item:selected {{ background: {t['accent_soft']}; color: {t['text']}; }}
QHeaderView::section {{
    background: {t['raised']};
    color: {t['muted']};
    border: none;
    border-bottom: 1px solid {t['border']};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {t['raised']}; border: none; }}

/* -- progress ---------------------------------------------------------- */
QProgressBar {{
    background: {t['raised']};
    border: 1px solid {t['border']};
    border-radius: {RADIUS}px;
    text-align: center;
    color: {t['muted']};
    height: 18px;
}}
QProgressBar::chunk {{
    background: {t['accent']};
    border-radius: {RADIUS - 1}px;
    margin: 1px;
}}

/* -- scrollbars -------------------------------------------------------- */
QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent; height: 11px; margin: 2px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {t['border_hi']};
    border-radius: 5px;
    min-height: 28px; min-width: 28px;
}}
QScrollBar::handle:hover {{ background: {t['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* -- misc -------------------------------------------------------------- */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {t['border']}; }}
QStatusBar {{
    background: {t['surface']};
    color: {t['muted']};
    border-top: 1px solid {t['border']};
}}
QStatusBar::item {{ border: none; }}
QToolTip {{
    background: {t['raised_hi']};
    color: {t['text']};
    border: 1px solid {t['border_hi']};
    border-radius: {RADIUS}px;
    padding: 6px 8px;
}}
QScrollArea {{ border: none; background: transparent; }}
QMenu {{
    background: {t['raised']};
    border: 1px solid {t['border_hi']};
    border-radius: {RADIUS}px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['accent']}; color: {t['on_accent']}; }}
"""


def _arrow_images(t: dict) -> None:
    """Paint the spin/combo arrows to PNGs, once per theme into a temp folder,
    and record their paths in the tokens: Qt's stylesheet ``image:`` takes only
    a URL, so an arrow that follows the palette has to be a file.
    """
    import tempfile

    from qtpy.QtCore import QPointF
    from qtpy.QtGui import QBrush, QPainter, QPixmap, QPolygonF

    folder = Path(tempfile.gettempdir()) / "trackanno-ui"
    folder.mkdir(parents=True, exist_ok=True)

    w, h, scale = 9, 6, 3          # drawn oversized so it stays crisp when scaled
    for name, colour, down in (
        ("_arrow_down", t["muted"], True),
        ("_arrow_up", t["muted"], False),
        ("_arrow_faint", t["faint"], True),
    ):
        path = folder / f"{name}_{colour.lstrip('#')}.png"
        if not path.exists():
            pixmap = QPixmap(w * scale, h * scale)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(colour)))
            top, bottom = 0.5, h * scale - 0.5
            points = (
                [QPointF(0, top), QPointF(w * scale, top), QPointF(w * scale / 2, bottom)]
                if down else
                [QPointF(0, bottom), QPointF(w * scale, bottom), QPointF(w * scale / 2, top)]
            )
            painter.drawPolygon(QPolygonF(points))
            painter.end()
            pixmap.save(str(path), "PNG")
        # QSS wants forward slashes even on Windows.
        t[name] = str(path).replace("\\", "/")


def _palette(t: dict) -> QPalette:
    """A matching QPalette, for the widgets that ignore stylesheets."""
    p = QPalette()
    bg, surface, text, muted = (QColor(t["bg"]), QColor(t["surface"]),
                                QColor(t["text"]), QColor(t["muted"]))
    p.setColor(QPalette.Window, bg)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, QColor(t["raised"]))
    p.setColor(QPalette.AlternateBase, surface)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor(t["raised"]))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.Highlight, QColor(t["accent"]))
    p.setColor(QPalette.HighlightedText, QColor(t["on_accent"]))
    p.setColor(QPalette.ToolTipBase, QColor(t["raised_hi"]))
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Mid, muted)
    p.setColor(QPalette.Dark, QColor(t["faint"]))
    p.setColor(QPalette.PlaceholderText, QColor(t["faint"]))
    for group in (QPalette.Disabled,):
        p.setColor(group, QPalette.WindowText, QColor(t["faint"]))
        p.setColor(group, QPalette.Text, QColor(t["faint"]))
        p.setColor(group, QPalette.ButtonText, QColor(t["faint"]))
    return p


def _prefers_dark(app: QApplication) -> bool:
    """Follow the desktop, falling back to dark if it will not say."""
    try:  # Qt 6.5+
        scheme = app.styleHints().colorScheme()
        if int(scheme) == 1:      # Qt.ColorScheme.Light
            return False
        if int(scheme) == 2:      # Qt.ColorScheme.Dark
            return True
    except Exception:
        pass
    window = app.palette().color(QPalette.Window)
    return window.lightness() < 128


def apply_theme(app: QApplication, name: str = "dark") -> str:
    """Apply a theme to the whole application. Returns the theme actually used."""
    global _active, _active_name

    if name == "system":
        name = "dark" if _prefers_dark(app) else "light"
    if name not in THEMES:
        name = "dark"
    _active, _active_name = dict(THEMES[name]), name
    _arrow_images(_active)      # must run before the stylesheet is built

    try:
        app.setStyle("Fusion")   # a predictable base for the stylesheet to sit on
    except Exception:
        pass
    app.setPalette(_palette(_active))
    app.setStyleSheet(_stylesheet(_active))

    font = app.font()
    for family in ("Segoe UI Variable Text", "Segoe UI", "Inter", "Ubuntu", "Helvetica Neue"):
        candidate = QFont(family)
        if candidate.exactMatch():
            font = candidate
            break
    font.setPointSize(max(9, font.pointSize() if font.pointSize() > 0 else 10))
    app.setFont(font)
    try:
        app.setWindowIcon(app_icon())
    except Exception:
        pass          # an icon is never worth failing to start over
    return name


# --------------------------------------------------------------------------- #
# small shared widgets
# --------------------------------------------------------------------------- #
class Heading(QWidget):
    """A page title and its one-line explanation, styled from the theme."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(3)

        self.title = QLabel(title)
        self.title.setProperty("role", "heading")
        layout.addWidget(self.title)

        self.subtitle = QLabel(subtitle)
        self.subtitle.setProperty("role", "subheading")
        self.subtitle.setWordWrap(True)
        self.subtitle.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle)


def theme_icon(size: int = 16) -> "QIcon":
    """A half-filled circle, painted rather than typed: a font glyph depends on
    the font carrying it and on it not being promoted to a colour emoji.
    """
    from qtpy.QtCore import QRectF
    from qtpy.QtGui import QIcon, QPainter, QPixmap

    t = tokens()
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    inset = 1.5
    rect = QRectF(inset, inset, size - 2 * inset, size - 2 * inset)
    painter.setPen(QPen(QColor(t["text"]), 1.4))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(rect)
    # Fill the half that stands for the theme you would switch TO.
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(t["text"]))
    painter.drawPie(rect, 90 * 16, 180 * 16)
    painter.end()
    return QIcon(pixmap)


def app_icon() -> "QIcon":
    """The application icon: the corner reticle, painted at every size rather
    than shipped as a file and scaled. The ground stays dark in both themes; a
    taskbar has its own colour scheme and does not follow ours.
    """
    from qtpy.QtCore import QLineF, QRectF
    from qtpy.QtGui import QIcon, QPainter, QPixmap

    from ..overlay import TARGET_RGB

    icon = QIcon()
    for size in (16, 32, 64, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Dark rounded ground, so the amber reads on any taskbar.
        pad = size * 0.06
        ground = QRectF(pad, pad, size - 2 * pad, size - 2 * pad)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#15171C"))
        painter.drawRoundedRect(ground, size * 0.22, size * 0.22)

        # Float coordinates: rounding to whole pixels at 16px moves a corner by
        # up to half an arm, and the stroke cannot go below a pixel, so the arms
        # are pushed out and shortened to keep the corners from closing to a ring.
        small = size <= 24
        inset = size * (0.23 if small else 0.29)
        arm = size * (0.14 if small else 0.16)
        pen = QPen(QColor(*TARGET_RGB), max(1.5, size * 0.072))
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        left = top = inset
        right = bottom = size - inset
        for (cx, cy), (dx, dy) in (
            ((left, top), (arm, arm)), ((right, top), (-arm, arm)),
            ((left, bottom), (arm, -arm)), ((right, bottom), (-arm, -arm)),
        ):
            painter.drawLine(QLineF(cx, cy, cx + dx, cy))
            painter.drawLine(QLineF(cx, cy, cx, cy + dy))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def compact_button(label: str = "", size: int = 28, tooltip: str = "",
                   danger: bool = False):
    """A small square button whose label actually fits. Use this rather than
    QPushButton + setFixedWidth: the themed padding is wider than the button and
    the text silently vanishes."""
    from qtpy.QtWidgets import QPushButton

    button = QPushButton(label)
    button.setFixedSize(size, size)
    button.setProperty("compact", "true")
    if danger:
        button.setProperty("danger", "true")
    if tooltip:
        button.setToolTip(tooltip)
    button.setFocusPolicy(Qt.NoFocus)
    return button


def banner_style(kind: str = "danger") -> str:
    """Inline style for the one-off notice strips (error, warning, info)."""
    t = tokens()
    colour = t.get(kind, t["danger"])
    return (
        f"background: {t['surface']}; color: {t['text']};"
        f"border: 1px solid {colour}; border-left: 3px solid {colour};"
        f"border-radius: {RADIUS}px; padding: 10px 12px;"
    )


def set_role(widget, name: str | None) -> None:
    """Tag a widget with a style role and make Qt re-evaluate the stylesheet."""
    widget.setProperty("role", name)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_flag(widget, name: str, value: bool) -> None:
    """Same, for a boolean property such as ``invalid`` or ``accent``."""
    widget.setProperty(name, "true" if value else None)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# Qt needs this import to exist for the palette group loop above.
_ = Qt
