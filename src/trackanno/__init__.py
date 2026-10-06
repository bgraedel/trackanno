"""trackanno - track-clip annotation pipeline.

``trackanno.gui.hub`` is the whole workflow in one window. The steps are also
separate entry points: ``detect.prepare`` (tiled YOLO detection, tracking, clip
cache), ``learn.suggest`` (train the label suggester), ``infer.predict`` (score every clip) and ``view.render`` (labelled
overlay movies).

Modules are grouped by what they are for: ``video`` reads recordings and model
weights, ``detect`` finds and tracks animals, ``annotate`` holds the clip grid
and the human records, ``measure`` turns tracks into numbers, ``learn`` fits and
directs the model, ``infer`` scores and decodes, ``view`` draws. ``config``,
``project``, ``templates``, ``yamledit`` and ``progress`` stay at the top level
because everything imports them and they import nothing back.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
