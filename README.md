# Enhancing Cattle Health Monitoring: A Multi-Model CNN Approach for Disease Detection

**Anima** — a decision-support system that combines a trained convolutional
classifier, a **locally-hosted** multimodal language model and
retrieval-augmented generation into a six-agent diagnostic workflow, served by a
FastAPI backend and a Vue 3 chat frontend.

> **Runs entirely offline. No hosted model APIs, no API keys, no per-token
> cost.** The CNN runs in-process via TensorFlow, the language models via Ollama
> on `127.0.0.1:11434`, and retrieval via local TF-IDF. The only HTTP the
> backend makes is to localhost.

```
Masters-Project/
├── vision/              Model assets the application loads at runtime
│   ├── cattle_lsd_detector.keras   trained CNN (11 MB)
│   └── visual_index.npz            precedent index (4.6 MB)
├── backend/             FastAPI service — agents, RAG, API   (see backend/README.md)
├── frontend/            Vue 3 + Vite chat interface
├── Makefile             entrypoint — macOS, Linux, WSL, Git Bash
└── make.ps1 / make.cmd  entrypoint — native Windows
```

### What is deliberately not in this repository

The **training dataset** (`cow dataset/`, 262 MB of cattle photographs) and the
**training notebook** (`Processing.ipynb`) are not committed here. This
repository holds what is needed to *run* the system, not to retrain it.

The application is unaffected, because the two artefacts the runtime actually
needs — the trained model and the precedent index — are committed. The index
carries its own thumbnails, so the visual evidence panel and the reference
galleries work with no dataset present.

Three things do need the dataset, and each says so plainly rather than failing
oddly:

| Needs the dataset | Why |
|---|---|
| `make index` | Rebuilds the precedent index by re-embedding every image. Only needed if the CNN is retrained. |
| `make evaluate` | Held-out evaluation of the classifier. |
| 7 of the 127 tests | They feed real dataset images through the model, and **skip** when it is absent — `make test` still passes. |

To restore any of those, put the dataset back at
`vision/cow dataset/{healthycows,lumpycows}/`. Keep it out of git: it
is 262 MB, and it is training data rather than an application asset.

---

## Quick start

**macOS · Linux · WSL · Git Bash**

```bash
make setup    # installs everything, end to end
make dev      # backend on :8000 + frontend on :5173
```

**Windows (PowerShell or cmd — nothing to install first)**

```powershell
.\make.ps1 setup
.\make.ps1 dev
```

The Makefile needs GNU make and bash, which Windows does not ship, so
`make.ps1` is a full native equivalent with the **same target names** —
everything below reads the same on either platform, just swap `make x` for
`.\make.ps1 x`. From `cmd.exe` you can drop the extension entirely and type
`make setup`, which runs the `make.cmd` shim. See [Windows](#windows) for
details.

Then open <http://localhost:5173>. Vite proxies `/api` to the backend, so no
CORS configuration is needed in development.

`make setup` is the whole bootstrap. It installs **Python 3.9–3.12, Node 18+ and
Ollama** using whatever package manager the machine has, creates the virtualenv,
installs the Python and npm dependencies, starts the model server and downloads
the model weights. It is safe to re-run — anything already present is detected
and skipped, and it asks before installing system packages.

| Platform | Package manager used |
|---|---|
| macOS | Homebrew |
| Linux (incl. WSL) | apt · dnf · yum · pacman · zypper · apk · Homebrew; Ollama via its official installer |
| Windows (Git Bash / MSYS2) | winget · Chocolatey · Scoop |

If no supported package manager is found, setup prints the exact download links
rather than guessing. Already have the tools somewhere non-standard? Point make
at them: `make setup PYTHON=/path/to/python3.12`.

`make` on its own lists every target. Useful ones:

| Target | Purpose |
|---|---|
| `make doctor` | Diagnose the setup — Python, Node, Ollama, models, CNN, server |
| `make health` | Which components are live right now |
| `make test` | Run the 127-test suite (needs no models) |
| `make evaluate` | Held-out evaluation of the trained CNN *(needs the dataset)* |
| `make models` | Download or update the model weights |
| `make index` | Rebuild the precedent index *(needs the dataset; the built index is committed)* |
| `make serve-ollama` | Start the model server in the background |
| `make backend` / `make frontend` | Run one half only |
| `make setup-offline` | Install code dependencies only — no system packages, no downloads |
| `make stop` | Stop stray dev servers |

> The backend runs **without** the local LLM too — if Ollama isn't running or
> the weights aren't pulled, it falls back to CNN + retrieval-only mode. The UI
> works; reports are flagged `degraded`, confidence is capped, and the answer
> states plainly that the reasoning agents did not run.

### Hardware and runtime

Sized for consumer hardware: **one** 3.3 GB multimodal model (`gemma3:4b`)
serves every LLM agent. Using one model rather than a vision/text pair is the
single biggest performance decision — Ollama keeps one model resident, so a
split makes it reload gigabytes between agents.

Measured on a CPU-only Apple Silicon laptop, one full six-agent diagnosis takes
**~135 s** (a text-only question, ~23 s). Nearly all of that is LLM inference —
the CNN takes 0.5 s and retrieval 0.02 s — which is why the UI streams
per-agent progress rather than blocking. A GPU cuts this dramatically.

Tighter on memory? `LSD_VISION_MODEL=moondream` + `LSD_TEXT_MODEL=llama3.2:3b`
in `backend/.env`, then `make models`. All options are listed there with
verified sizes.

---

## The agent workflow

| # | Agent | Type | Role |
|---|---|---|---|
| 1 | Intake triage | local VLM (`llava:7b`) | Are these cattle? Is the image usable? Stops the pipeline early on a photo of a goat or a landscape. |
| 2 | CNN screening | Keras MobileNetV2 | `P(LSD-like)` from the trained classifier, plus the Grad-CAM overlay and nearest labelled cases that explain it. Deterministic, no LLM. |
| 3 | Lesion analysis | local VLM (`llava:7b`) | Structured description of visible findings — morphology, distribution, severity. **Never names a disease.** |
| 4 | Knowledge retrieval | TF-IDF over a curated corpus | Pulls the veterinary passages relevant to the evidence gathered so far. |
| 5 | Diagnostic synthesis | local LLM (`llama3.1:8b`) | Reconciles all three streams into a verdict with differentials, actions and citations. |
| 6 | Grounding & safety review | local LLM (`llama3.1:8b`) | Adversarial check: unsupported claims, CNN/VLM disagreement, missing safety framing. |

Agents 2 and 3 run **in parallel** and are deliberately blind to each other, so
agent 5 reconciles two genuinely independent signals and agent 6 can flag when
they conflict. Full architecture diagram and rationale: `backend/README.md`.

The frontend renders the pipeline live — each agent's status, summary and
duration stream in over Server-Sent Events while the diagnosis runs.

---

## What the system produces

Every submission returns a `DiagnosisReport` containing:

- a **verdict** (`lsd_likely` / `lsd_unlikely` / `inconclusive` / `not_cattle`)
  and a confidence the verification agent may adjust downwards;
- the **CNN probability** shown against its decision threshold, with per-image
  scores and a "near threshold" flag;
- **visual evidence** for that probability — a saliency overlay and the nearest
  labelled cases (see below);
- **differential diagnoses** drawn from the retrieved literature;
- **recommended next steps** with an urgency level;
- **citations** — every clinical claim traces to a `chunk_id` in the corpus;
- the full **agent trace**, including any degraded or failed steps.

---

## Showing the score, not just asserting it

`P(LSD-like) = 0.87` is not something a farmer or an examiner can check. So
every report that includes a CNN score also shows **why** that score came out
the way it did. Both explanations come from the classifier that already runs —
no second model, no download, about **150 ms** on top of a ~135 s pipeline.

**Where the model looked.** A Grad-CAM heat map over the photo, marking the
region that moved the decision. This is diagnostic in both directions: when the
heat sits on the animal's flank the score is meaningful, and when it sits on
mud, fencing or a bystander — which happens — the score is being driven by
scenery and should be discounted. That is visible at a glance and invisible in
the number.

**Most similar labelled cases.** The six nearest training images in the
classifier's own feature space, with their true labels and a vote. Leave-one-out
kNN in that space scores **0.955**, so the vote is a real signal rather than
decoration: in testing it correctly contradicted the CNN on a healthy cow the
classifier had scored at 0.56. The panel says plainly when the neighbours
disagree with the score, and the synthesis agent is told to lower its confidence
when they do.

Two honesty guarantees are built in, because both failures are silent:

- these are **training images**, labelled as such — they show what the score is
  pattern-matching against, never independent confirmation;
- if the uploaded photo is **itself in the training set**, the report says so
  and drops it from its own precedents. Demoing with a dataset image is the
  easiest way to make this system look better than it is.

**Examples in text answers.** The two panels above need an uploaded photo. But
"tell me about LSD and give me an example" is a question about *appearance*, and
answering it in prose alone reads as broken — so a text question that asks to be
shown something gets labelled reference photographs from the dataset too. Ask
how it spreads and you get none; ask what it looks like, how to recognise it, or
for the difference between healthy and affected animals, and you get pictures
(both classes, for a comparison).

Those images are chosen, not sampled: the most *typical* members of the class by
distance to its centroid, restricted to images the classifier scores
confidently, with a diversity filter so six consecutive frames of the same
animal never fill the strip. The gallery is attached to the answer and is never
shown to the language model, so pictures cannot change what it claims.

The precedent index is committed to this repository, so nothing needs building
to use these features. `make index` regenerates it (~20 s for 1043 images) and
requires the training dataset — only necessary if the CNN is retrained. Without it, saliency still works and
the precedent strip is simply absent — as with every other component here,
missing pieces degrade rather than fail.

---

## Windows

The final owner of this project runs Windows, so the whole toolchain has a
native entrypoint that needs nothing installed beforehand — no WSL, no Git Bash,
no `make`.

```powershell
.\make.ps1              # list every target
.\make.ps1 setup        # Python + Node + Ollama + venv + npm + weights + index
.\make.ps1 dev          # backend and frontend together; Ctrl-C stops both
.\make.ps1 doctor       # diagnose the setup
```

| | |
|---|---|
| **Same targets** | Every Makefile target exists with the same name and behaviour, so the rest of this README applies verbatim — swap `make x` for `.\make.ps1 x`. |
| **`make` still works** | From `cmd.exe`, `make setup` resolves to the `make.cmd` shim and hands off to the script. |
| **Execution policy** | The shim passes `-ExecutionPolicy Bypass` for that one invocation, which is the usual reason a PowerShell script silently does nothing on a fresh machine. It changes no system setting. |
| **Package managers** | winget, then Chocolatey, then Scoop. It asks before installing anything, and prints the download links if none is available. |
| **PATH after install** | Installers only extend PATH for new processes, so setup re-reads it mid-run rather than making you reopen the terminal. |
| **Interpreter** | Finds Python 3.9–3.12 via the `py` launcher, skipping the Microsoft Store stub (which exits 9009 and opens the Store instead of running). Override with `.\make.ps1 setup -Python C:\path\to\python.exe`. |

Preview what setup would do without touching the machine:

```powershell
.\make.ps1 system-deps -DryRun
```

Note that `make.ps1` is the *only* Windows-specific file — the backend, the
frontend and the tests are identical on every platform.

---

## Findings worth carrying into the write-up

### 1. The notebook's reported accuracy is not valid

The training notebook (`Processing.ipynb`, kept with the dataset outside this
repository) builds its validation set with `shuffle=False`, which takes
the last 20% of the *directory-ordered* file list. Since the classes sort as
`healthycows` then `lumpycows`, that split contains **zero healthy cows** — as
the notebook's own confusion matrix shows (`[[0, 0], [11, 197]]`). The reported
95% accuracy is recall on a single class, and because the training set used
`shuffle=True`, that split also overlaps the training data.

`backend/scripts/evaluate_model.py` rebuilds the split correctly:

| | Notebook (invalid) | Corrected held-out |
|---|---|---|
| Validation set | 208 images, **all lumpy** | 208 images, 99 healthy / 109 lumpy |
| Overlaps training data | yes | no |
| Accuracy | 0.95 (meaningless) | **0.8317** |
| ROC-AUC | not computable | **0.9033** |

### 2. The classifier is not calibrated

In the `p ∈ [0.10, 0.75)` bands the actual `lumpycows` rate is 55–63%, far from
the nominal probability. The sigmoid output is a similarity score, not a
probability of disease. The API therefore always reports the score *with* its
threshold, and treats the ±0.15 band around it — 9.1% of held-out cases — as
**inconclusive** rather than forcing a label.

### 3. A binary classifier is a closed world

The model has exactly two classes. Shown ringworm, dermatophilosis, mange,
insect bites or photosensitisation, it must still return one of them. That is
the core argument for the surrounding architecture: the triage agent rejects
non-cattle input, the lesion analyst describes what is actually visible, the
retrieved literature supplies the differentials the CNN cannot represent, and
the verification agent catches unsupported claims before a farmer sees them.

### 4. Some of the classifier's attention is on the background, not the animal

The Grad-CAM overlays make this checkable, and on a sample of dataset images the
heat sometimes lands on ground litter, fencing or handlers rather than on skin.
That is consistent with the two class folders differing systematically in
setting as well as in disease — the kind of shortcut a binary classifier will
happily learn, and one that inflates accuracy on this dataset while predicting
nothing about a new farm.

This is worth stating directly in the write-up: it explains part of the gap
between the corrected held-out accuracy (0.8317) and what the model would do in
the field, and it is an argument for the saliency panel rather than against it —
the failure was always there, and was simply not visible before.

---

## Tests

```bash
make test        # 127 tests — no models, no network, no credentials
```

- **Retrieval quality** — 17 realistic farmer queries against expected documents
- **Local LLM client** — run against a stub inference server: request shape,
  image attachment, schema-constrained decoding, JSON recovery from prose, the
  repair retry, and server-down degradation
- **CNN** — model loads and preprocessing separates the training classes
- **Reference gallery** — which questions earn example photographs, and that
  the selection skips near-duplicates and images the model is unsure of
- **Visual evidence** — saliency rendering and the index round-trip run
  everywhere; the model-backed checks skip cleanly when the weights or the
  index are absent. These pin the two failures that are invisible by
  inspection: a saturated-sigmoid heat map that is uniform but well-formed, and
  a re-uploaded training image being counted as its own strongest precedent
- **API** — endpoint contracts, upload limits, and that the pipeline still
  returns a safe, honestly-labelled report when the LLM is unavailable

---

## Environment notes

`make setup` installs and verifies all of this — the list is here so you know
what it puts on the machine.

- **Python 3.9–3.12.** TensorFlow publishes wheels for that range only, and
  setup refuses anything outside it with an actionable message. 3.11/3.12 are
  best: 3.9 works but caps Keras at 3.10, older than the Keras that saved the
  model (the loader compensates — `backend/README.md` → *Keras version
  compatibility*).
- **Node 18+** for the frontend.
- **Ollama** for local inference, plus ~9 GB of model weights.
- No credentials of any kind. `backend/.env` contains only local settings.

Want to see what setup would do before running it?

```bash
bash scripts/system-deps.sh check              # report only, install nothing
DRY_RUN=1 bash scripts/system-deps.sh ensure   # print the install commands
```

---

## ⚠️ Scope and safety

This system provides **decision support, not a diagnosis**. Lumpy skin disease
is confirmed by laboratory testing and is notifiable in most jurisdictions. A
photograph cannot show fever, lesion depth on palpation, lymph node enlargement
or herd context — the features that separate LSD from its mimics. Every report
carries that framing and directs the user to a veterinarian and the competent
veterinary authority. It must not be used as the basis for treatment, culling or
trade decisions.
