# Anima — LSD Diagnostic API

Multi-agent decision support for **Lumpy Skin Disease (LSD)** in cattle. A
FastAPI service that combines the trained CNN in `vision/` with a
**locally-hosted** multimodal LLM and retrieval-augmented generation over a
curated veterinary corpus.

> **No hosted model APIs.** Every model call runs on the machine hosting the
> backend — the CNN via TensorFlow in-process, the language models via Ollama on
> `127.0.0.1:11434`, and retrieval via local TF-IDF. There is no API key, no
> per-token cost, and no network egress: the only HTTP the service makes is to
> localhost.

---

## The agent workflow

```
                     ┌──────────────────────────────────────────┐
  images + message ─►│ [1] Intake triage      (local VLM)        │
                     │     Is this cattle? Is the image usable?  │
                     └──────────────┬───────────────────────────┘
                                    │ proceed?
                    ┌───────────────┴────────────────┐
                    │          in parallel           │
        ┌───────────▼──────────┐      ┌──────────────▼───────────┐
        │ [2] CNN screening    │      │ [3] Lesion analysis      │
        │     Keras MobileNetV2│      │     local VLM (llava)    │
        │     P(LSD-like)      │      │     structured findings  │
        └───────────┬──────────┘      └──────────────┬───────────┘
                    └───────────────┬────────────────┘
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │ [4] Knowledge retrieval  (RAG, TF-IDF)    │
                     │     query built from the evidence above   │
                     └──────────────┬───────────────────────────┘
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │ [5] Diagnostic synthesis  (local LLM)      │
                     │     verdict + differentials + citations    │
                     └──────────────┬───────────────────────────┘
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │ [6] Grounding & safety review (local LLM) │
                     │     adversarial: unsupported claims,       │
                     │     CNN/VLM disagreement, safety framing   │
                     └──────────────┬───────────────────────────┘
                                    ▼
                              DiagnosisReport
```

**Why this shape.** Agents 2 and 3 are deliberately *independent* evidence
streams: the CNN never sees the language model's opinion, and the lesion analyst
is never told the CNN's score or asked to name a disease. That means agent 5 is
genuinely reconciling two sources rather than one opinion echoed twice, and
agent 6 can flag when they disagree. Every stage is fault-tolerant — if an agent
fails, its step is marked `degraded` and the pipeline continues with the
remaining evidence.

---

## Quick start

From the repository root, one command installs and configures everything —
system dependencies, virtualenv, npm packages and model weights:

```bash
make setup
make backend          # start the API on :8000
```

See the root `README.md` for what `make setup` installs per platform. Or by
hand:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # no API keys in it — only local settings
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>
Health check: <http://127.0.0.1:8000/api/health>

Check `/api/health` first — it reports which components are live:

```json
{
  "status": "ok",
  "llm_available": true,
  "cnn_available": true,
  "knowledge_documents": 7,
  "knowledge_chunks": 36,
  "notes": ["Retrieval backend: sklearn-tfidf"]
}
```

### Running without the local LLM

The service starts and answers even when Ollama is not running or the weights
are not pulled. It falls back to **CNN + retrieval-only mode**: `/api/diagnose`
still returns the classifier probability and the relevant knowledge-base
passages, the report is flagged `degraded` with its confidence capped at 0.6,
and the answer says plainly that the reasoning agents did not run. Set
`LSD_ENABLE_LLM=0` to force this mode deliberately.

Useful on a machine that cannot host a 7B model, and it keeps the test suite
free of any model dependency.

### Choosing models

Both roles default to **one** multimodal model, `gemma3:4b` (3.3 GB). That
choice matters more than raw model quality on consumer hardware: Ollama keeps
one model resident at a time, so pointing the vision and text roles at
different models makes it evict and reload gigabytes between agents — which
dominates runtime.

Sizes below are the actual layer totals from the Ollama registry. The vision
role needs a **multimodal** model; text-only models cannot fill it.

| Model | Size | Notes |
|---|---|---|
| `llava-phi3` | 2.9 GB | Smaller than the default; shorter context window |
| **`gemma3:4b`** | **3.3 GB** | **Default — multimodal, large context** |
| `llava:7b` | 4.7 GB | Older; noticeably weaker at following the JSON schema |
| `qwen2.5vl:7b` | 6.0 GB | Best vision quality here; needs ~8 GB RAM |

If 3.3 GB genuinely will not fit, split to `moondream` (1.7 GB) for vision and
`llama3.2:3b` (2.0 GB) for text — and accept the reload cost between agents.

Set `LSD_VISION_MODEL` / `LSD_TEXT_MODEL` in `backend/.env`; `make models`
reads those values, so it always pulls what the service is configured to use.
The request also sends `keep_alive` (default `30m`) so the model stays resident
across the pipeline's four sequential calls.

### Measured runtime

One full six-agent diagnosis, CPU-only Apple Silicon laptop, `llava:7b` in both
roles (the default `gemma3:4b` is smaller and should be quicker):

| Agent | Time |
|---|---|
| Intake triage | 13.6 s |
| CNN screening | 0.5 s |
| Lesion analysis | 23.2 s |
| Knowledge retrieval | 0.02 s |
| Diagnostic synthesis | 66.6 s |
| Grounding & safety review | 32.1 s |
| **Total** | **135 s** |

A text-only knowledge question takes about 23 s. The CNN and retrieval stages
are effectively free — essentially all the time is local LLM inference, which is
why the UI streams per-agent progress instead of blocking.

### Structured output on small models

The structured agents rely on Ollama's **JSON-schema-constrained decoding**
(`format=<schema>`): the Pydantic schema is compiled to a grammar, so malformed
output is impossible rather than merely unlikely. That is what makes a 7B local
model dependable enough to drive the pipeline. The client additionally restates
the schema in the system prompt (small models produce better *values* when they
can see the field descriptions), recovers JSON from markdown fences or
surrounding prose, and makes one repair attempt with the validation error before
degrading the step.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Component availability and degradation notes |
| `GET` | `/api/config` | Non-secret runtime configuration |
| `POST` | `/api/chat` | Main frontend surface — routes to diagnosis or Q&A |
| `GET` | `/api/chat/sessions` | List conversations |
| `GET` | `/api/chat/sessions/{id}` | Conversation transcript |
| `DELETE` | `/api/chat/sessions/{id}` | Delete a conversation |
| `POST` | `/api/diagnose` | Diagnose a `multipart/form-data` upload |
| `POST` | `/api/diagnose/json` | Diagnose base64 data-URL images |
| `POST` | `/api/diagnose/stream` | Diagnose with live per-agent SSE progress |
| `GET` | `/api/knowledge/documents` | List corpus documents |
| `GET` | `/api/knowledge/search` | Inspect raw retrieval results |
| `POST` | `/api/knowledge/ask` | Grounded text-only question answering |

### Examples

```bash
# Diagnose an image
curl -X POST http://127.0.0.1:8000/api/diagnose \
  -F "images=@../vision/cow dataset/lumpycows/img1000.jpg" \
  -F "message=My cow has lumps all over"

# Text-only question, answered from the corpus
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How does lumpy skin disease spread?"}'

# Inspect what retrieval returns for a query (no LLM involved)
curl "http://127.0.0.1:8000/api/knowledge/search?q=vaccination%20coverage&top_k=3"
```

The SSE stream emits one `agent_step` event per state change, then a `report`:

```
event: agent_step
data: {"agent":"cnn_vision","status":"running",...}

event: agent_step
data: {"agent":"cnn_vision","status":"completed","summary":"P(LSD-like)=0.9992 ...","duration_ms":812.4,...}

event: report
data: {"request_id":"...","verdict":"lsd_likely","confidence":0.78,...}

event: done
data: {}
```

---

## The CNN agent

Wraps `vision/cattle_lsd_detector.keras` — a frozen ImageNet
MobileNetV2 backbone with a 128-unit dense head ending in a single sigmoid unit.

Preprocessing mirrors training exactly, which matters:

* training used `image_dataset_from_directory(..., image_size=(224, 224))`,
  a plain bilinear resize that does **not** preserve aspect ratio;
* the `Rescaling(1/127.5, offset=-1)` layer is **inside** the saved model, so
  the agent feeds raw 0–255 float pixels and must not normalise beforehand;
* `RandomRotation` / `RandomFlip` are also inside the model but inert at
  inference time.

Class order comes from the alphabetically sorted dataset directories:
`healthycows = 0`, `lumpycows = 1`, so the sigmoid output is `P(lumpycows)`.

**Multi-image aggregation** takes the *maximum* per-image probability
(screening-conservative), so one clearly affected photo is not diluted by other
frames of the same animal showing unaffected body regions.

### Visual evidence — showing the score instead of asserting it

`P(LSD-like) = 0.87` is not reviewable. A farmer cannot tell whether the model
looked at a lesion or at the mud behind the animal, and neither can an examiner.
So the CNN agent returns two explanations alongside the number, both derived
from the same forward pass (`app/agents/visual_explain.py`, ~150 ms total):

**1. Grad-CAM saliency.** Gradients of the score with respect to the backbone's
final `(7, 7, 1280)` feature map, ReLU'd and composited over the photo. Two
implementation notes, both non-obvious:

* The saved model is **Sequential**, so its layers were never called as a graph
  and `layer.output` raises. The usual
  `keras.Model(model.input, [conv.output, model.output])` recipe fails outright.
  `split_stack()` instead splits the layer list around the nested MobileNetV2
  sub-model and re-applies the head by hand.
* Gradients are taken on the **logit**, not the sigmoid. This model saturates
  hard — a confident image scores 0.9999, where the sigmoid's derivative
  underflows and the map comes out blank. Swapping the head to a linear
  activation for the duration of the tape raises the raw gradient span from
  ~0.001 to 0.13–0.86 on the same images. `test_saliency_is_not_a_uniform_wash`
  pins this, because the failure produces a perfectly well-formed, meaningless
  image.

**2. Precedent lookup** (`app/reference_index.py`). The nearest labelled
training images by cosine similarity in the trained 128-d penultimate layer.
Leave-one-out kNN in that space scores **0.955** (k=5) against 0.846 for the raw
1280-d pooled backbone features, which is why the dense layer is used — measured
on training images, so it describes retrieval behaviour, not held-out accuracy.

The neighbour vote is a *second read of the same classifier*, not a fourth
evidence stream, and the synthesis prompt says so explicitly: it may move
confidence, but it may never be cited as clinical evidence. When it disagrees
with the CNN's own label the report says so — in testing it correctly flagged a
healthy cow the classifier had scored at 0.56.

If the upload is **itself in the training set** the report says that too, and
excludes it from its own precedents. That is the single most common way a demo
overstates the system, and it is silent unless you look for it.

Everything here degrades to absent rather than failing: no index, no
TensorFlow gradient, or any other error still returns the screening probability
(`test_screening_survives_a_broken_explanation`).

### Reference gallery on the text path

Both features above require an uploaded image. `app/agents/reference_gallery.py`
covers the case where there is none: "what does LSD look like?" and "give me an
example" are questions about appearance, and answering them in prose alone reads
as a broken feature rather than a missing one.

`wants_examples()` matches the phrasings farmers and examiners actually use
(*example, show me, looks like, how do I recognise, signs, difference between*)
and is deliberately generous — a spurious thumbnail strip costs a moment,
whereas asking for an example and getting text is the bug that prompted this.
Purely administrative questions (*notifiable, quarantine period, cost*) are
excluded.

`ReferenceIndex.representatives()` picks the images. Not the highest-scoring
ones and not a random sample — both illustrate badly. It takes images the
classifier scores confidently, ranks them by closeness to the class centroid
(most typical first), then applies a greedy diversity filter so consecutive
frames of the same animal cannot fill the strip. When the filter starves the
gallery it tops up with the *least* similar remaining candidates, so rejected
near-duplicates are added back last.

The gallery attaches to the answer in `orchestrator.answer_text_question()`,
*after* generation, so it reaches both the LLM and retrieval-only paths and the
pictures can never influence what the model claims.

> **Index format v2.** The index now stores per-image probabilities, which the
> selection above needs. A v1 file is rejected with a "rebuild it" message
> rather than silently misread — run `make index --force`, or
> `.\make.ps1 index` on Windows.

### Citations are ids in the data, numbers on the screen

The agents cite passages by corpus id — `[lsd-clinical-signs#01]` — and that
stays: the verification agent resolves those ids to check every clinical claim,
so stripping them at the source would remove the grounding trail the design
rests on.

A farmer has never seen the corpus, though, so the id is noise on screen. The
frontend (`src/utils/citations.js`) resolves each id against the retrieved
passages and renders it as a footnote number keyed to the Sources list, in
first-appearance order. Ids that resolve to no passage are dropped from the
display rather than shown as broken references — the verification agent reports
those separately as unsupported claims.

One consequence worth knowing: the Sources list now shows only the passages the
answer actually cited, not everything retrieved, so it is usually shorter than
`retrieval.passages`. The full retrieval set is still in the API response.

### Keras version compatibility

The model was saved by a newer Keras than pip will install on Python 3.9
(Keras 3.10 is the ceiling there), and its `BatchNormalization` config carries
`renorm*` keys that older Keras rejects with *"Unrecognized keyword arguments"*.
`_relaxed_layer_configs()` in `app/agents/cnn_vision.py` patches `from_config`
on the affected layer classes for the duration of the load, retrying with the
rejected keys dropped. Note that `custom_objects` does **not** work for this:
Keras resolves built-in layers by module path and only consults `custom_objects`
via the config's `registered_name`, which is null for built-ins.

---

## ⚠️ Model evaluation — the notebook's metrics are not valid

`Processing.ipynb` builds its validation set with `shuffle=False`:

```python
val_generator = utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset="validation",
    seed=42, ..., shuffle=False)          # <-- the bug
```

With `shuffle=False` the split is taken from the end of the *directory-ordered*
file list. The classes sort as `healthycows` then `lumpycows`, so the last 20%
is **entirely `lumpycows`** — exactly what the notebook's own confusion matrix
shows:

```
[[  0   0]
 [ 11 197]]     ← zero healthy samples
```

So the reported **95% accuracy is recall on a single class**, precision/recall
for `Healthy Cows` are undefined (hence the `UndefinedMetricWarning`s), and
because the *training* set was built with `shuffle=True`, that validation set
also overlaps the training data.

`scripts/evaluate_model.py` rebuilds the split the way training did
(`shuffle=True`, same seed), giving the genuinely held-out 20%:

```bash
.venv/bin/python scripts/evaluate_model.py
```

**Corrected held-out results** (208 images: 99 healthy, 109 lumpy):

| Metric | Value |
|---|---|
| Accuracy | **0.8317** |
| ROC-AUC | **0.9033** |
| Precision (lumpycows) | 0.8627 |
| Recall (lumpycows) | 0.8073 |
| Precision (healthycows) | 0.8019 |
| Recall (healthycows) | 0.8586 |

Confusion matrix at threshold 0.5:

```
                  pred healthycows   pred lumpycows
true healthycows              85               14
true lumpycows                21               88
```

The script also prints a threshold sweep and a calibration table. Two findings
worth carrying into the write-up:

1. **The model is not calibrated.** In the `p ∈ [0.10, 0.75)` bands the actual
   `lumpycows` rate is only 55–63%, nowhere near the nominal probability. The
   sigmoid output should not be read as a probability of disease. This is why
   the API reports the raw score *with* its threshold and treats the ±0.15 band
   around it (9.1% of held-out cases) as **inconclusive** rather than forcing a
   label.
2. **83% accuracy is a screening signal, not a diagnosis** — which is precisely
   the argument for the surrounding multi-agent architecture: the visual
   analysis, the retrieved literature and the verification agent exist to catch
   what a binary classifier at this accuracy will get wrong.

---

## Retrieval (RAG)

Seven markdown documents in `app/rag/knowledge/`, chunked on `##` headings into
36 passages. Retrieval is TF-IDF + cosine similarity — deterministic, no
embedding service, trivially reproducible for a thesis appendix.

Three implementation choices that measurably improved ranking (see
`tests/test_retrieval.py`, which is the evaluation harness):

* **Heading weighting** — document title and section heading are repeated 3× in
  the indexed text; without it the body swamps the heading and queries land on
  the wrong section.
* **Query expansion** — farmers do not write in textbook vocabulary, so
  colloquial terms are expanded with clinical synonyms (`spread` →
  *transmission, transmitted, infectious*; `lump` → *nodule, nodular*;
  `shot` → *vaccination, vaccine*).
* **Light stemming** — collapses `nodule`/`nodules`/`nodular`,
  `morphology`/`morphological`, `lesion`/`lesions`.

Both backends (scikit-learn and the pure-Python fallback) share one analyzer, so
retrieval quality does not change with which dependencies happen to be
installed. Both score 17/17 on the top-3 retrieval test set.

Add a document by dropping a markdown file with YAML frontmatter
(`id`, `title`, `source`, `topics`) into `app/rag/knowledge/` and restarting.

---

## Configuration

All settings are environment variables — see `.env.example`. The ones that
change behaviour most:

| Variable | Default | Effect |
|---|---|---|
| `LSD_LLM_BASE_URL` | `http://127.0.0.1:11434` | Local inference server |
| `LSD_VISION_MODEL` | `llava:7b` | Triage + lesion analysis |
| `LSD_TEXT_MODEL` | `llama3.1:8b` | Synthesis, verification, Q&A |
| `LSD_LLM_TIMEOUT` | `300` | Read timeout — raise on CPU-only machines |
| `LSD_LLM_TEMPERATURE` | `0.1` | Low: these agents extract, they don't write creatively |
| `LSD_DECISION_THRESHOLD` | `0.5` | CNN decision boundary |
| `LSD_UNCERTAINTY_BAND` | `0.15` | Band around the threshold reported as inconclusive |
| `LSD_MODEL_PATH` | `../vision/cattle_lsd_detector.keras` | Trained model |
| `LSD_ENABLE_LLM` | `1` | Set `0` to run CNN + retrieval only |
| `LSD_ENABLE_VISUAL_EVIDENCE` | `1` | Set `0` to drop saliency + precedents |
| `LSD_VISUAL_NEIGHBOURS` | `6` | Precedent images shown per diagnosis |
| `LSD_VISUAL_INDEX_PATH` | `../vision/visual_index.npz` | Built by `make index` |

---

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

* `test_retrieval.py` — retrieval quality against 17 realistic farmer queries
* `test_cnn.py` — model loads; preprocessing separates the training classes
  (skipped automatically if TensorFlow or the model is absent)
* `test_local_llm.py` — the local LLM client, run against a **stub inference
  server** that mimics Ollama's `/api/tags` and `/api/chat`. Covers request
  shape, image attachment, schema-constrained decoding, JSON recovery from
  prose, the repair retry, and server-down degradation — without needing
  multi-gigabyte weights.
* `test_api.py` — endpoint contracts, validation limits, and that the pipeline
  still returns a safe, honestly-labelled report when the LLM is unavailable

The suite needs no model weights, no running Ollama and no credentials.

---

## Notes and limitations

* **Sessions are in-memory** with a TTL and are lost on restart. Swap
  `session_store.py` for Redis if durability is needed; the interface is small.
* **Local inference is slow without a GPU.** A full six-agent run on CPU can
  take minutes, most of it in the two vision calls. The SSE endpoint exists
  partly for this reason — the UI shows progress rather than hanging. Use a
  smaller vision model (`moondream`) on modest hardware.
* **Small models are weaker reasoners than a frontier model.** Expect shorter
  differentials and occasional schema repairs. The grammar constraint guarantees
  the *shape* of every response, not the quality of its judgement — which is
  what the verification agent is there to check.
* **The knowledge base is a compiled summary**, not primary source text.
  Passages are attributed to WOAH and FAO guidance and should be checked against
  the current editions before citing them in the thesis.
* **This is decision support, not a diagnosis.** LSD is confirmed by laboratory
  testing and is notifiable in most jurisdictions. Every report carries that
  framing, and the verification agent flags it when it goes missing.
