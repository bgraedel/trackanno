"""The whole pipeline in one window.

    trackanno

Pages down the left, in the order the work happens: pick a project,
configure it, run detection, annotate, train the suggester, look inside a label
for sub-behaviours, check the labels, export overlay movies. Each page states
its own prerequisites rather than failing when you press the button, so the
sidebar doubles as a progress readout.

The CLI entry points call the same ``prepare_all`` / ``train`` / ``render_all``
functions, so the two paths cannot drift apart.
"""

from .annotate import AnnotatePage
from .base import PAGE_HINTS, PAGES, Jobs, Page
from .configure import ConfigPage, ModelRegistryDialog, _install_shared_model
from .export import ExportPage, _reveal
from .model import ModelPage, _feature_table, _format_report, count_orphans, predict_plan_line
from .prepare import PreparePage, cfg_patterns
from .project import NewProjectDialog, ProjectPage
from .review import ReviewPage
from .window import Hub, install_excepthook, main

__all__ = [
    "AnnotatePage", "ConfigPage", "ExportPage", "Hub", "Jobs", "ModelPage",
    "ModelRegistryDialog", "NewProjectDialog", "PAGES", "PAGE_HINTS", "Page",
    "PreparePage", "ProjectPage", "ReviewPage", "cfg_patterns", "count_orphans",
    "install_excepthook", "main", "predict_plan_line",
    "_feature_table", "_format_report", "_install_shared_model", "_reveal",
]
