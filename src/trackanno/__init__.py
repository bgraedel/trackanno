"""trackanno - track-clip annotation pipeline.

``trackanno.gui.hub`` is the whole workflow in one window. The steps are also
separate entry points: ``prepare`` (tiled YOLO detection, tracking, clip cache),
``gui.app`` (the annotation screen), ``suggest`` (train the label suggester),
``predict`` (score every clip) and ``render`` (labelled overlay movies).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
