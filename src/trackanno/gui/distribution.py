"""How the labels are actually spread, drawn rather than tabulated.

Shown on the Model page. Three things worth seeing before training: one class swallowing most of the
clips, so a classifier that always answers it scores well and means nothing; a
class below ``suggest.min_per_class``, dropped at training time; a class living
in a single movie, which grouped cross-validation cannot put in both train and
test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from .theme import tokens

__all__ = ["Tally", "GroupTally", "tally_annotations", "BarChart", "DistributionView",
           "read_annotation_rows"]

MULTI_SEP = "|"
UNLABELLED = "(not set)"


@dataclass
class Tally:
    """One bar."""

    name: str
    count: int = 0
    movies: set = field(default_factory=set)
    color: str = ""
    known: bool = True          # False: in the CSV but not in the config

    @property
    def n_movies(self) -> int:
        return len(self.movies)


@dataclass
class GroupTally:
    """One chart: every option of one category group."""

    name: str
    is_multi: bool
    bars: list = field(default_factory=list)
    n_rows: int = 0             # annotated clips considered for this group

    @property
    def total(self) -> int:
        return sum(b.count for b in self.bars)

    @property
    def largest(self) -> int:
        return max((b.count for b in self.bars), default=0)


def tally_annotations(rows, cfg) -> list:
    """[GroupTally] from CSV rows, in the order the config declares them.

    Every configured option gets a bar even at zero, and a value in the CSV the
    config no longer declares gets one marked unknown."""
    out = []
    for group in cfg.categories:
        tally = GroupTally(name=group.name, is_multi=group.is_multi)
        by_name = {}
        for option in group.options:
            bar = Tally(name=option.name, color=option.color)
            tally.bars.append(bar)
            by_name[option.name] = bar
        blank = Tally(name=UNLABELLED, color="", known=True)

        for row in rows:
            raw = str(row.get(group.name, "") or "").strip()
            movie = str(row.get("movie_id", "") or "")
            tally.n_rows += 1
            if not raw:
                blank.count += 1
                blank.movies.add(movie)
                continue
            values = [v.strip() for v in raw.split(MULTI_SEP)] if group.is_multi else [raw]
            for value in values:
                if not value:
                    continue
                bar = by_name.get(value)
                if bar is None:
                    bar = Tally(name=value, color="", known=False)
                    by_name[value] = bar
                    tally.bars.append(bar)
                bar.count += 1
                bar.movies.add(movie)

        # Only when non-zero, so a fully filled group carries no empty row.
        if blank.count:
            tally.bars.append(blank)
        out.append(tally)
    return out


class BarChart(QWidget):
    """Horizontal bars for one category group, sized from its own contents."""

    ROW = 26
    GAP = 5
    TOP = 8
    BOTTOM = 10

    def __init__(self, tally: GroupTally, min_per_class: int = 0, parent=None):
        super().__init__(parent)
        self.tally = tally
        self.min_per_class = int(min_per_class)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(360)

    def set_tally(self, tally: GroupTally, min_per_class: int = 0) -> None:
        self.tally = tally
        self.min_per_class = int(min_per_class)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        rows = max(1, len(self.tally.bars))
        return QSize(460, self.TOP + rows * (self.ROW + self.GAP) - self.GAP + self.BOTTOM)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # -- painting ----------------------------------------------------------- #
    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        bars = self.tally.bars
        if not bars:
            painter.setPen(QColor(t["faint"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "no options configured")
            return

        label_font = QFont(self.font())
        value_font = QFont(self.font())
        value_font.setWeight(QFont.DemiBold)
        metrics = painter.fontMetrics()

        # Column widths measured from the actual strings, not a fixed fraction.
        label_w = min(
            max(140, max(metrics.horizontalAdvance(b.name) for b in bars) + 26),
            max(150, int(self.width() * 0.42)),
        )
        value_texts = [self._value_text(b) for b in bars]
        value_w = max(metrics.horizontalAdvance(s) for s in value_texts) + 12

        track_x = label_w + 10
        track_w = max(30, self.width() - track_x - value_w - 6)
        scale = self.tally.largest or 1

        y = float(self.TOP)
        for bar, value_text in zip(bars, value_texts):
            # -- name, with a swatch in the option's own colour --------------
            painter.setFont(label_font)
            swatch = QRectF(2, y + (self.ROW - 11) / 2, 11, 11)
            colour = QColor(bar.color) if bar.color else QColor(t["faint"])
            if not colour.isValid():
                colour = QColor(t["faint"])
            painter.setPen(Qt.NoPen)
            painter.setBrush(colour)
            painter.drawRoundedRect(swatch, 3, 3)

            name_rect = QRectF(20, y, label_w - 24, self.ROW)
            painter.setPen(QColor(t["text"] if bar.known else t["warning"]))
            painter.drawText(
                name_rect, int(Qt.AlignVCenter | Qt.AlignLeft),
                metrics.elidedText(bar.name, Qt.ElideRight, int(name_rect.width())),
            )

            # -- track and bar ----------------------------------------------
            track = QRectF(track_x, y + (self.ROW - 16) / 2, track_w, 16)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t["raised"]))
            painter.drawRoundedRect(track, 4, 4)

            if bar.count > 0:
                width = max(3.0, track_w * bar.count / scale)
                fill = QRectF(track.x(), track.y(), width, track.height())
                painter.setBrush(colour)
                painter.drawRoundedRect(fill, 4, 4)

            # The line the Model page applies: left of it is dropped at training.
            if self.min_per_class > 1 and self.min_per_class < scale:
                x = track_x + track_w * self.min_per_class / scale
                pen = QPen(QColor(t["border_hi"]), 1, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(int(x), int(track.top() - 2), int(x), int(track.bottom() + 2))

            # -- count ------------------------------------------------------
            painter.setFont(value_font)
            short = bar.count < self.min_per_class and bar.name != UNLABELLED
            painter.setPen(QColor(t["warning"] if short else t["muted"]))
            painter.drawText(
                QRectF(self.width() - value_w, y, value_w - 4, self.ROW),
                int(Qt.AlignVCenter | Qt.AlignRight), value_text,
            )
            y += self.ROW + self.GAP
        painter.end()

    def _value_text(self, bar: Tally) -> str:
        total = self.tally.n_rows or 1
        text = f"{bar.count}   {100.0 * bar.count / total:.0f}%"
        # Only when it is a problem: a class in one movie cannot be scored by
        # grouped-by-movie cross-validation.
        if bar.count and bar.n_movies == 1 and bar.name != UNLABELLED:
            text += "   1 movie"
        return text


def summary_lines(tallies, cfg) -> list:
    """Plain-language warnings, or [] when the distribution is healthy."""
    out = []
    minimum = cfg.suggest.min_per_class
    for tally in tallies:
        if not tally.n_rows:
            continue
        real = [b for b in tally.bars if b.name != UNLABELLED]
        empty = [b.name for b in real if b.count == 0]
        thin = [b.name for b in real if 0 < b.count < minimum]
        single = [b.name for b in real if b.count and b.n_movies == 1]
        unknown = [b.name for b in real if not b.known]
        blank = next((b for b in tally.bars if b.name == UNLABELLED), None)

        if empty:
            out.append(f"{tally.name}: nothing labelled {_join(empty)} yet.")
        if thin:
            out.append(
                f"{tally.name}: {_join(thin)} below the {minimum}-label minimum, "
                "so they are dropped at training time."
            )
        if single:
            one = len(single) == 1
            out.append(
                f"{tally.name}: {_join(single)} only "
                f"{'appears' if one else 'appear'} in one movie, so grouped "
                f"cross-validation cannot score {'it' if one else 'them'}."
            )
        if unknown:
            out.append(
                f"{tally.name}: {_join(unknown)} {_is(unknown)} in the CSV but not "
                "in the config - an option was renamed or removed after labelling."
            )
        if blank is not None and blank.count and not tally.is_multi:
            out.append(
                f"{tally.name}: {blank.count} annotated clip(s) have no value for "
                "this group."
            )
        # Imbalance, judged on the labels themselves: counting the "(not set)"
        # bar would make an unlabelled group report a nonsense percentage.
        labelled = sum(b.count for b in real)
        top = max(real, key=lambda b: b.count, default=None)
        if labelled >= 20 and top is not None and top.count / labelled > 0.9:
            out.append(
                f"{tally.name}: {100.0 * top.count / labelled:.0f}% of labels are "
                f"'{top.name}', so a model that always answers that would look "
                "accurate."
            )
    return out


def _is(names) -> str:
    return "is" if len(list(names)) == 1 else "are"


def _join(names) -> str:
    names = list(names)
    if len(names) == 1:
        return f"'{names[0]}'"
    if len(names) == 2:
        return f"'{names[0]}' and '{names[1]}'"
    return ", ".join(f"'{n}'" for n in names[:-1]) + f" and '{names[-1]}'"


def read_annotation_rows(cfg):
    """(rows, problem). A broken CSV must report itself, not raise into Qt."""
    import csv

    path = cfg.paths.csv
    if not path.exists():
        return [], f"no annotation file yet at {path}"
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh)), ""
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return [], f"could not read {path.name}: {exc}"


class DistributionView(QWidget):
    """What the labels look like: a coverage line, warnings and one chart per group."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stale = True
        self._charts: dict[str, BarChart] = {}
        self._cfg = None
        self._status = None

        self.coverage = QLabel("")
        self.coverage.setWordWrap(True)
        self.coverage.setTextFormat(Qt.RichText)

        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.RichText)
        self.notes.setVisible(False)

        refresh = QPushButton("Refresh")
        refresh.setToolTip("Re-read the annotation CSV")
        refresh.clicked.connect(self.reload)

        head = QHBoxLayout()
        head.addWidget(self.coverage, 1)
        head.addWidget(refresh, 0, Qt.AlignTop)

        self.charts_box = QWidget()
        self.charts_layout = QVBoxLayout(self.charts_box)
        self.charts_layout.setContentsMargins(0, 0, 0, 0)
        self.charts_layout.setSpacing(12)
        self.charts_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.charts_box)

        layout = QVBoxLayout(self)
        layout.addLayout(head)
        layout.addWidget(self.notes)
        layout.addWidget(scroll, 1)

    def set_project(self, cfg, status) -> None:
        """Point at a (re)loaded project; charts are rebuilt on the next reload."""
        self._cfg, self._status = cfg, status
        self.stale = True
        # Groups may have been renamed or removed; old charts cannot be reused.
        self._clear_charts()

    def blocker(self) -> str:
        if self._cfg is None or self._status is None:
            return "no project open"
        if self._status.n_annotated == 0:
            return "no annotations yet - label some clips first"
        return ""

    def _clear_charts(self) -> None:
        for box in list(self._charts.values()):
            holder = box.parentWidget()
            while holder is not None and not isinstance(holder, QGroupBox):
                holder = holder.parentWidget()
            (holder or box).setParent(None)
            (holder or box).deleteLater()
        self._charts.clear()

    def reload(self) -> None:
        """Re-read the CSV and repaint."""
        self.stale = False
        blocker = self.blocker()
        if blocker:
            self._clear_charts()
            self.coverage.setText(f"<span style='color:{tokens()['muted']}'>{blocker}</span>")
            self.notes.setVisible(False)
            return

        cfg = self._cfg
        rows, problem = read_annotation_rows(cfg)
        if problem:
            self._clear_charts()
            self.coverage.setText(
                f"<span style='color:{tokens()['danger']}'>{problem}</span>"
            )
            self.notes.setVisible(False)
            return

        tallies = tally_annotations(rows, cfg)
        self._paint_summary(rows)

        minimum = cfg.suggest.min_per_class
        for tally in tallies:
            chart = self._charts.get(tally.name)
            if chart is None:
                chart = BarChart(tally, minimum)
                self._charts[tally.name] = chart
                title = tally.name + ("   (multi-select)" if tally.is_multi else "")
                box = QGroupBox(title)
                inner = QVBoxLayout(box)
                # Zero: the QGroupBox rule already reserves 34px for the title.
                inner.setContentsMargins(0, 0, 0, 0)
                inner.addWidget(chart)
                self.charts_layout.insertWidget(self.charts_layout.count() - 1, box)
            else:
                chart.set_tally(tally, minimum)

        lines = summary_lines(tallies, cfg)
        self.notes.setVisible(bool(lines))
        if lines:
            items = "".join(f"<li>{line}</li>" for line in lines)
            self.notes.setText(
                f"<div style='color:{tokens()['warning']}'>"
                f"<ul style='margin-left:-20px;'>{items}</ul></div>"
            )

    def _paint_summary(self, rows) -> None:
        status = self._status
        n = len(rows)
        total = status.n_clips if status else 0
        share = f" ({100.0 * n / total:.0f}% of {total} clips)" if total else ""
        movies = {str(r.get("movie_id", "") or "") for r in rows} - {""}
        people = {str(r.get("annotator", "") or "") for r in rows} - {""}
        uncertain = sum(1 for r in rows if str(r.get("uncertain", "")).strip() == "1")

        parts = [f"<b>{n}</b> clip(s) labelled{share}",
                 f"across <b>{len(movies)}</b> movie(s)"]
        if len(people) > 1:
            parts.append(f"by <b>{len(people)}</b> annotators")
        elif people:
            parts.append(f"by {sorted(people)[0]}")
        if uncertain:
            parts.append(f"<b>{uncertain}</b> marked uncertain")
        self.coverage.setText(
            "   |   ".join(parts)
            + f"<br><span style='color:{tokens()['faint']}'>The dashed line marks "
            f"suggest.min_per_class ({self._cfg.suggest.min_per_class}); classes "
            "left of it are dropped when the model is trained.</span>"
        )
