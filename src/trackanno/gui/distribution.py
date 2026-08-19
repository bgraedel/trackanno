"""How the labels are actually spread, drawn rather than tabulated.

Three things worth seeing on the way to Export: one class swallowing most of the
clips, so a classifier that always answers it scores well and means nothing; a
class below ``suggest.min_per_class``, silently dropped at training time; a class
living in a single movie, which grouped cross-validation cannot put in both train
and test, so its score is meaningless.

Painted with QPainter rather than a plotting library: no new dependency, it
follows the theme without a second palette, and it stays sharp at any DPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

from .theme import tokens

__all__ = ["Tally", "GroupTally", "tally_annotations", "BarChart"]

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

    Every configured option gets a bar even at zero: an empty class is the most
    important thing on the chart. Values in the CSV the config no longer declares
    get bars too, marked unknown, so a renamed option shows as orphaned."""
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
    """Horizontal bars for one category group, sized from its own contents: three
    options and twelve options should not get the same height."""

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
        # bar makes an entirely unlabelled group report a nonsense percentage.
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
