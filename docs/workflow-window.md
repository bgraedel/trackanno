# The workflow window

```bash
trackanno                      # reopens the last project
trackanno path/to/project
```

Pages down the left, in the order the work happens. Each shows its state (`*`
done, `>` next, `!` something missing) and a count. Each page states what it
needs before its button works, and a **Next:** bar says what to do after it.
Longer explanations sit behind the `?` beside each page title.

| page | purpose |
|---|---|
| **Project** | create, open or reopen a project; save it as a template and export that |
| **Configure** | settings, category buttons, keypoint names |
| **Detect & crop** | detection, tracking and the clip cache ([details](detection.md)) |
| **Annotate** | the annotation screen ([details](annotation.md)) |
| **Model** | train the suggester, label every frame, label distribution ([details](suggester.md)) |
| **Review** | decoded bouts beside the footage ([details](prediction.md#the-review-page)) |
| **Export** | overlay movies ([details](export.md)) |

Long jobs run on a worker thread with a progress bar and a Cancel button.

## New project and presets

*New project...* asks for a project name, where to create it, what to **start
from**, the video folder and the detection model. The model is a `.pt` on this
machine or **Choose shared model...** (see [Shared models](detection.md#shared-models)).

**Start from** offers the built-in default, three presets and any saved
templates.

| preset | detection | sets up |
|---|---|---|
| `annotator` | boxes | clips and the annotation screen; no suggester |
| `suggester` | pose | adds the CatBoost suggester on kinematic, pose and waveform features |
| `dense bouts` | pose | adds dense prediction: 25-frame clips, `predict.window_stride: 5`, `predict.min_bout_frames: 10` |

A preset (`src/trackanno/presets/*.yaml`) names only the keys it changes and is
applied to the packaged template, so every comment survives. Moving between
presets later is a matter of editing those keys. Changing `detection.task` or
`clips.frames` means running Detect & crop again, and a new clip length means a
new set of clips to label.

## Templates

**Template > Save as template...** on the Project page stores the open project's
`config.yaml`, comments included, under a name. It is parsed by the loader
first, so a template that would not load is refused. `paths.video_dir`,
`paths.work_dir`, `paths.csv` and `ui.annotator` are reset; `paths.model` is
kept. Templates are `.yaml` files in the `templates` folder of the state
directory.

To share one, save it, then **Template > Export** writes it to a file, and
**Import...** next to **start from** in the New project dialog adds it on
another computer. Import also takes another project's `config.yaml`. Export
lists the settings that name files on this computer (`paths.model` unless
`model.source` is set, and the `centerline` weights and code); Import lists
those that do not exist on the new one, to be set on the Configure page or in
`config.yaml`. A relative path counts as missing, since it points into the new
project.

```python
from trackanno.project import create_project
from trackanno.templates import export_template, list_templates, save_template

create_project("new-project", video_dir="movies/", template="dense bouts")
create_project("other-project", template="shared.yaml")   # a template file
save_template("Trypanosome 100 fps", "path/to/config.yaml")
export_template("Trypanosome 100 fps", "shared.yaml")
```

An unknown template name raises `ValueError`. `list_templates()` returns the
built-in default, the presets, then saved templates.

## The Configure page

Four tabs:

* **Basics**: paths, detection task and device, minimum track length, clip
  length, stride, window mode, crop follow and padding, playback fps, annotator.
* **Labels**: the category button editor.
* **Keypoints**: the pose keypoint table (`detection.task: pose` only).
* **Advanced**: the other form fields, for example confidence, tracker, border
  margin, crop size, intensity mapping, auto-advance, suggester and prediction
  settings.

Edits stay in the form until **Save**, which writes only the keys that changed
back into `config.yaml` in place, keeping every comment. **Revert** reloads the
file. Leaving the page with unsaved edits asks Save, Discard or Cancel. A save
is parsed by the loader first; an edit that would not load is refused with the
loader's message and the file is untouched.

Fields marked `*` change what Detect & crop produces, so they take effect after
it runs again. Keys outside the form ([list](configuration.md)) are edited in
`config.yaml` directly.

Picking a `.pt` reads its task and keypoint count; where they disagree with the
form, **Use these** applies them. On a pose project with no keypoint names,
**Set up keypoints** opens the Keypoints tab.

The colour theme is `ui.theme` in `config.yaml`; there is no toggle in the
window.
