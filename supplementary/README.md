# MUGC evaluation - reproducible implementation details

This folder contains the implementation details requested by the reviewers for

- **Comment 1**: the exact prompts used to generate IG/VG Test Captions, and how videos are passed to the captioning model (frame selection and preprocessing);
- **Comment 2**: the exact scoring prompt, the JSON schema of the scoring output, a short runnable example, and how dimensions that are not applicable (in particular Text) are handled.

No API keys, provider URLs, or other credentials are stored in this folder. All
endpoints and credentials are supplied through environment variables.

---

## 1. Test Caption generation

### 1.1 Image generation (IG)

Each generated image is sent to the captioning model **directly**, as a single
image, in a single request. No resizing, cropping, or normalisation is applied.
The prompt is chosen according to the cultural context:

| Culture | Prompt file |
|---|---|
| Chinese (ZH) | `prompts/caption_ig_zh.txt` |
| American (US) | `prompts/caption_ig_en.txt` |
| Japanese (JP) | `prompts/caption_ig_jp.txt` |

### 1.2 Video generation (VG)

**The video file itself is never sent to the model.** It is decomposed into
still frames, each frame is captioned in a separate request, and the per-frame
descriptions are concatenated into one plain-text Test Caption.

1. Read the clip duration `D` (seconds) with `ffprobe`.
2. Sample frames at

   ```
   t = 0, 5, 10, 15, ...   while  t < floor(D)
   ```

   That is: one frame every five seconds, starting at 0 s, using only
   timestamps strictly below the duration rounded down to the nearest whole
   second. Example: a 5.04 s clip contains a single sampled frame, at 0 s.

3. Extract each frame with

   ```
   ffmpeg -v error -ss <t> -i <video> -frames:v 1 -q:v 2 -y <frame>.jpg
   ```

   One JPEG per timestamp, encoded at quality `-q:v 2`. **Original resolution is
   preserved**; no resizing, cropping, padding, letterboxing, or colour
   normalisation is applied.

4. Each frame is base64-encoded and sent as a `data:image/jpeg;base64,...` URI
   in a separate request.

5. The returned per-frame descriptions are joined in temporal order with a
   **single space**:

   ```
   Test Caption = desc(t0) + " " + desc(t5) + " " + ...
   ```

   The result is stored as plain text.

Prompts:

| Culture | Prompt file |
|---|---|
| Chinese (ZH) | `prompts/caption_vg_zh.txt` |
| American (US) | `prompts/caption_vg_en.txt` |
| Japanese (JP) | `prompts/caption_vg_jp.txt` |

The helper `utils/video_frames.py` implements steps 1-3. See
`examples/run_caption_example.py` for a runnable demonstration.

---

## 2. Scoring

### 2.1 Prompt

The full scoring prompt is `prompts/score_judge.txt`. The two placeholders
`{gold_caption}` and `{test_caption}` are replaced with the human-validated
reference description and the Test Caption respectively.

The scoring prompt requires the model to

- extract the structured representation internally (not emitted);
- assign four dimension scores in [0, 100]: `background_score`,
  `object_score`, `text_score`, `culture_score`;
- produce a holistic `total_score` in [0, 100].

### 2.2 Output format and JSON schema

The model returns a fixed key-value block:

```
score output format:
background_score: <0-100>
object_score: <0-100>
text_score: <0-100>
culture_score: <0-100>
total_score: <0-100>
```

`utils/parse_score.py` converts this block into the JSON record defined by
`schema/score_record.schema.json`:

```json
{
  "sample_id": "example-0001",
  "background_score": 100,
  "object_score": 92,
  "text_score": 0,
  "culture_score": 95,
  "total_score": 95,
  "text_applicable": false
}
```

`sample_id` and `text_applicable` are added by the parser; they are not
produced by the model.

### 2.3 How dimensions that are not applicable are handled

The scoring prompt states that **a dimension that is absent from the reference
does not contribute to `total_score` and does not lower it**. This applies in
particular to the Text dimension: if neither the reference nor the Test Caption
contains text, `text_score` is 0 and it is excluded from the holistic total.

For the per-dimension averages reported in Tables 4-7:

- Background, Object and Culture are averaged over all evaluated samples of a
  model in that setting;
- Text is reported only over samples whose reference captions contain text to
  be evaluated; the corresponding numbers of samples (ZH / US / JP) are
  19 / 42 / 28 for IU and IG, and 20 / 55 / 67 for VU and VG, as also stated
  in the captions of Tables 4-7.

---

## 3. Short runnable examples

### 3.1 Offline (no API key needed)

```bash
# Show the frame plan for a video (no model call)
python examples/run_caption_example.py --video path/to/clip.mp4 --offline

# Parse a cached scoring response into JSON and validate it
python examples/run_score_example.py --offline
```

### 3.2 Online (OpenAI-compatible endpoint)

```bash
export OPENAI_API_KEY="..."                 # required
export OPENAI_BASE_URL="https://..."        # optional, OpenAI-compatible endpoint
export OPENAI_MODEL="gpt-5"                 # optional, defaults to gpt-5

python examples/run_caption_example.py --video path/to/clip.mp4 --call
python examples/run_score_example.py
```

No endpoint or key is hard-coded anywhere in this folder.

---

## 4. Files

```
github_release/
├── README.md
├── prompts/
│   ├── caption_ig_zh.txt / caption_ig_en.txt / caption_ig_jp.txt
│   ├── caption_vg_zh.txt / caption_vg_en.txt / caption_vg_jp.txt
│   └── score_judge.txt
├── schema/
│   └── score_record.schema.json
├── utils/
│   ├── video_frames.py
│   └── parse_score.py
└── examples/
    ├── run_caption_example.py
    ├── run_score_example.py
    ├── sample_ref.json
    ├── sample_test.json
    └── cached_score_response.txt
```
