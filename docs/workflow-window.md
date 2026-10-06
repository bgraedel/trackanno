# The workflow window

```bash
trackanno                      # reopens the last project
trackanno path/to/project
```

Steps down the left, in the order the work happens. Each carries its state
(`*` done, `>` next, `!` something missing) and a count.

| step | purpose |
|---|---|
| **1 Project** | pick the folder to work in |
| **2 Configure** | point it at the movies, choose the labels |
| **3 Detect & crop** | find the animals, cut them into clips |
| **4 Annotate** | watch each clip, press one key |
| **5 Model** | train on the labels, suggest the next one, see how they are spread |
| **6 Review** | check where the model put each boundary |
| **7 Export** | put the labels back onto the original movies |

Each page states its prerequisites up front and offers a **Next:** bar at the
bottom.

**Explanation lives under a `?`.** What is permanently on screen is a line: a
page's subtitle says what the page is for, a caption under a picture is the
fields that are true of *that* track and no others, and every one of them is
short enough to read past. The reasoning behind it is one button away and
closes again: what a bar is, why a held-out prediction is the only fair
comparison, what a bar's height means. A paragraph pinned above the work is height
taken from it on every visit, and it is read once.

**A project** is a folder holding `config.yaml`, `work/` (the clip cache) and
`annotations.csv`. *New project* writes a fully commented `config.yaml` with the
paths filled in. Its detection model row takes either a `.pt` on this machine or
**Choose shared model...**, which is fetched once the project exists and brings
the settings it was trained with (see
[Shared models](detection.md#shared-models)).

**Presets.** `config.yaml` is 500 lines and every one of them has a default, but
a handful decide what the project *is*. Those are set together, under **start
from** in *New project*:

| preset | detection | what it sets up |
|---|---|---|
| `annotator` | boxes | Clips and the annotation screen. No classifier. |
| `suggester` | pose | The above plus a CatBoost model trained on your labels, fitted on the trajectory, pose and waveform features, with the skeleton drawn on the clip and on the export. |
| `dense bouts` | pose | The above plus dense prediction: 25-frame windows stepped by 5, a min_bout decode into bouts, and boundary review. Annotation clips come down to 25 frames with it. |

Pose from the suggester up, because a model that cannot see the beat is not the
one worth training; `annotator` is the rung for a plain box detector. Each step
up is a handful of settings, and the Configure page sets every one of them by
hand, so a project can climb without being made again. What no edit makes free
is a change to what step 3 produced: `annotator` to `suggester` wants pose
weights and a second detection run, and `suggester` to `dense bouts` re-cuts the
clips at 25 frames, which is a different clip grid and so a different set of
things to have labelled.

`dense bouts` is two numbers, and they are not independent. A model is fitted at
`clips.frames` and refuses to score a grid of any other length, so the 25-frame
dense window is bought by cutting 25-frame annotation clips: `clips.frames` and
`clips.stride` come to 25, `min_clip_frames` to 20 (the same 80% of a window the
default asks for) and `clips.smooth_window` to 13, a crop average as long as the
clip being a fixed crop by another name. The dense window *is* `clips.frames`,
so changing the clip length moves it and the pair cannot come apart. Only `predict.window_stride: 5`
stands alone: five frames between window starts, which is what pins a change of
behaviour to five frames rather than to a window, at five times the windows to
score.

A preset lives in `trackanno/presets/*.yaml` and names only the keys it changes,
applied to the packaged default. Nothing copies the whole file, so a preset
cannot drift away from the defaults it was written against, and every comment in
`config.yaml` survives being overridden.

**Templates.** Tuning the detector, the crop geometry and the category buttons
is most of the setup for a dataset, and the next dataset usually wants the same
answers. *Save as template* on the Project page stores the open project's
settings under a name; *New project* then offers them under **start from**.

A template is a whole `config.yaml`, comments included, kept outside any project
so it outlives the one it came from. It carries everything except the three
paths that describe one project on one disk: `video_dir`, `work_dir` and `csv`
are reset, since a template pointing at last month's movie folder is worse than
one pointing nowhere. `paths.model` is kept, because the same weights are
normally reused, and the New Project form overrides it when a model is chosen.
`ui.annotator` is cleared, a template being a protocol rather than a person.

Saving copies the file rather than re-serialising it, so the comments survive,
and the result is parsed by the real loader before it is stored: a template that
would not load cannot be saved. They live under the state directory
(`%APPDATA%\trackanno\templates`, or `~/.config/trackanno/templates`), one
`.yaml` each, so a template can be handed to somebody else as a file.

From the API:

```python
from trackanno.project import create_project
from trackanno.templates import list_templates, save_template

create_project("new-project", video_dir="movies/", template="dense bouts")

save_template("Trypanosome 100 fps", "path/to/config.yaml")
create_project("another", video_dir="movies/", template="Trypanosome 100 fps")
```

`list_templates()` returns the built-in default, then the presets, then what has
been saved; `template` takes any of their names.

**Configure** has four tabs. *Basics* holds the fifteen settings that must be
decided (movie location, model, clip length); *Labels* is the button editor;
*Keypoints* is the pose table, active only for `task: pose`; *Advanced* holds
detector, tracker and cropping parameters. Settings marked `*` describe how the
clips were cut, so changing one requires re-running step 3, which the page
states.

**Settings autosave** ~0.7 s after typing stops, and navigating away flushes.
Values are rewritten in place, line by line, preserving every comment and the
file order, so the two thirds of `config.yaml` that document each knob survive.
Each autosave is parsed by the real loader before it replaces anything; an edit
that would not load is refused with the loader's own message and the file is
untouched. Paths that resolve to nothing are marked amber and annotated, since a
model on an offline share should not block unrelated edits.

**Theme.** `gui/theme.py` holds one palette and a stylesheet generated from it.
Widgets request a role (`muted`, `danger`, `heading`) rather than a colour.
`ui.theme` takes `dark` (default), `light` or `system`, and is applied when the
window opens.

**Long jobs run on a worker thread.** Detection reports movie, stage and frame
count; Cancel takes effect at the next frame.

---
