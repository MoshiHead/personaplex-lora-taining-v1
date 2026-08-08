"""
make_demo_rag_dataset.py
=========================
Generates a small (10-example) demo dataset in the exact format expected by
this repo's RAG training pipeline (moshi_local/moshi/dataset.py,
interleaver.py, train.py).

Format reverse-engineered directly from the code, not guessed:

  data/demo_rag_dataset/
    train.jsonl                        <- manifest consumed by dataset.py /
                                           sphn.dataset_jsonl. One JSON object
                                           per line: {"path": ..., "duration": ...}
                                           (field names match dataset.py's
                                           maybe_load_local_dataset(), which
                                           reads data["path"] / data["duration"]).
    audio/<name>.wav                   <- mono, 24 kHz (loaders.SAMPLE_RATE),
                                           PLACEHOLDER audio (quiet tone/silence
                                           at exactly the declared duration).
                                           This repo has no TTS step, so real
                                           speech audio must come from you --
                                           this script only makes the pipeline
                                           runnable end-to-end for format
                                           verification, not a usable checkpoint.
    audio/<name>.new.combined.json     <- sidecar consumed by
                                           InterleavedTokenizer.__call__ via
                                           `os.path.splitext(path)[0] + ".new.combined.json"`
                                           (interleaver.py). Keys:
                                             alignments      : [[word, [start_sec, end_sec], speaker], ...]
                                             segment_meta    : {"turns": [{"kind", "start_sec", "end_sec"}, ...]}
                                             text_conditions : {"reference": "<passage text>"}   (optional)
                                             metadata        : {"system_prompt": "<persona text>"} (optional)

segment_meta turn "kind" vocabulary (interleaver.py TurnSpan / _parse_segment_meta):
    regular | lookup | ref | body
("lead" / "filler" are deprecated -- the model must never speak between the
user's turn and the injected <ref>, so current data prep never produces
them. Old sidecars that still have them still parse, just with a logged
deprecation warning.)

`ref` is NOT written into the sidecar by us -- it is synthesized at
tokenization time by splicing text_conditions["reference"] immediately after
the injected <lookup> block (see interleaver.py: `_inject_rag_blocks`). A
sample only gets RAG injection if it has `lookup` + `body` spans AND a
non-empty text_conditions.reference.

Turn shape: USER audio -> <lookup> -> <ref> reference text <ref> -> MODEL body.
The model's first and only spoken audio is `body`, which only ever follows
the injected reference -- there is no lead-in or filler speech at all.

Note on "server decides the trigger" vs "model decides the trigger": that is
an INFERENCE-time choice (server.py vs server_lora.py) and does not change
this training data format at all. Both inference designs are trained from
the same literal <ref>/<lookup> splice convention built by train.py +
interleaver.py, which is what this dataset targets.

speaker labels: main_speaker_label="MODEL" is hard-coded in train.py's
Interleaver(...) construction, so assistant turns MUST use speaker "MODEL".
The user speaker label just needs to be consistently different; we use
"USER".

Of the 10 examples:
  - 8 are full RAG episodes (USER -> <lookup> -> <ref> -> MODEL body, no
    model speech before the answer), each with a different reference
    passage, so the model sees varied domains.
  - 2 are deliberately RAG-free ("regular" only, or no segment_meta at all)
    -- negative examples so the model also sees ordinary dialogue that must
    NOT trigger a lookup. (See prior discussion: the repo's own
    ReferenceBuffer wrong-reference mechanism is unused/dead code, so this
    demo cannot supply "wrong reference" contrastive examples either -- that
    gap has to be closed separately, this script only fixes the missing
    negative-turn coverage.)
"""

from __future__ import annotations

import json
import math
import os
import wave

import numpy as np

SAMPLE_RATE = 24000          # loaders.SAMPLE_RATE
FRAME_RATE = 12.5            # loaders.FRAME_RATE / mimi.frame_rate

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demo_rag_dataset")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")


def write_placeholder_wav_stereo(path: str, duration_sec: float, tone_hz: float,
                                  speaker_windows: list[tuple[float, float, str]]) -> None:
    """STEREO 16-bit PCM @ SAMPLE_RATE -- channel 0 = agent/MODEL audio,
    channel 1 = USER audio (a very quiet tone during each window in
    `speaker_windows` where that role is "speaking", silence elsewhere).
    NOT real speech -- exists only so the pipeline has a real audio file of
    the right shape/length/sample rate to run mimi.encode() on end-to-end.

    Stereo is required, not cosmetic: interleaver.py calls
    `mimi.encode(audio_tensor[:, None])` on this file's [2, T] samples,
    which treats the 2 channels as 2 batch items each independently encoded
    to 8 codebooks, then flattens to [1, 16, T] via `.view(1, -1, T)| --
    channel 0 -> codebooks 0-7 (agent), channel 1 -> codebooks 8-15 (user),
    matching the SILENCE_TOKENS(agent)/SINE_TOKENS(user) convention already
    hardcoded in `_build_forced_text_frames`. A mono file only produces 8
    codebooks total and raises:
        RuntimeError: Sizes of tensors must match except in dimension 2.
        Expected size 8 but got size 16 for tensor number 1 in the list.
    """
    n_samples = int(round(duration_sec * SAMPLE_RATE))
    agent = np.zeros(n_samples, dtype=np.float32)
    user = np.zeros(n_samples, dtype=np.float32)
    amplitude = 800 / 32767.0  # quiet

    t = np.arange(n_samples) / SAMPLE_RATE
    tone = amplitude * np.sin(2 * math.pi * tone_hz * t)

    for start_sec, end_sec, role in speaker_windows:
        s = max(0, int(round(start_sec * SAMPLE_RATE)))
        e = min(n_samples, int(round(end_sec * SAMPLE_RATE)))
        if s >= e:
            continue
        target = agent if role == "MODEL" else user
        target[s:e] = tone[s:e]

    stereo = np.stack([agent, user], axis=-1)  # [T, 2] for interleaved PCM
    pcm16 = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())


def words_alignment(text: str, start_sec: float, end_sec: float, speaker: str) -> list:
    """Evenly places each word of `text` inside [start_sec, end_sec).
    Placeholder timing (no real forced aligner available here) -- for a real
    dataset these come from an ASR/forced-alignment pass over real audio."""
    words = text.split()
    n = len(words)
    span = end_sec - start_sec
    step = span / n
    out = []
    for i, w in enumerate(words):
        w_start = start_sec + i * step
        w_end = w_start + step * 0.9  # leave a small gap between words
        out.append([w, [round(w_start, 3), round(w_end, 3)], speaker])
    return out


def turn(kind: str, start_sec: float, end_sec: float) -> dict:
    return {"kind": kind, "start_sec": round(start_sec, 3), "end_sec": round(end_sec, 3)}


# ---------------------------------------------------------------------------
# 8 RAG examples: USER question -> <lookup> marker -> <ref> (injected) ->
# body (MODEL, grounded in the reference, first model speech in the sample)
# -> trailing regular (MODEL). No lead/filler audio at all.
# ---------------------------------------------------------------------------

RAG_EXAMPLES = [
    dict(
        name="ex01_weather",
        system_prompt=None,
        user_question="Hey, what's the weather going to be like in Austin tomorrow?",
        body_model="Tomorrow in Austin it will be sunny with a high of thirty one degrees Celsius and light southerly wind.",
        tail_model="Anything else you would like to know?",
        reference=(
            "Austin, TX forecast for tomorrow: sunny skies, high 31C / low 19C, "
            "wind from the south at 12 km/h, 0% chance of precipitation, UV index high."
        ),
        tone_hz=220,
    ),
    dict(
        name="ex02_return_policy",
        system_prompt="You work for SwiftPlex Appliances. Your name is Farhod. Be concise and friendly.",
        user_question="If the blender I bought last week doesn't work, can I return it?",
        body_model="Yes, SwiftPlex accepts returns within thirty days of purchase with a receipt, and defective items get a full refund.",
        tail_model="Would you like me to start that return for you?",
        reference=(
            "SwiftPlex Appliances return policy: items may be returned within 30 days of "
            "purchase with proof of purchase. Defective items receive a full refund; "
            "non-defective returns receive store credit minus a 10% restocking fee."
        ),
        tone_hz=240,
    ),
    dict(
        name="ex03_capital_city",
        system_prompt=None,
        user_question="Quick question, what's the capital of Australia?",
        body_model="The capital of Australia is Canberra, not Sydney as people often assume.",
        tail_model="Happy to answer more geography questions.",
        reference=(
            "Australia's capital city is Canberra, located in the Australian Capital "
            "Territory. It was purpose-built as the capital in 1913 as a compromise "
            "between Sydney and Melbourne. Sydney is the largest city but not the capital."
        ),
        tone_hz=260,
    ),
    dict(
        name="ex04_product_spec",
        system_prompt="You are a support agent for NimbusTech laptops.",
        user_question="Does the Nimbus Air 14 support Wi-Fi 6E?",
        body_model="Yes, the Nimbus Air 14 ships with a Wi-Fi 6E card and also supports Bluetooth 5.3.",
        tail_model="Let me know if you need the full spec list.",
        reference=(
            "Nimbus Air 14 technical specifications: 14-inch 2.8K OLED display, "
            "Wi-Fi 6E and Bluetooth 5.3 wireless module, 16GB LPDDR5 RAM, "
            "512GB NVMe SSD, 65W USB-C charging."
        ),
        tone_hz=280,
    ),
    dict(
        name="ex05_recipe",
        system_prompt=None,
        user_question="How long do I need to bake a banana bread at 350 degrees?",
        body_model="At three hundred fifty degrees Fahrenheit, banana bread typically needs fifty five to sixty five minutes until a toothpick comes out clean.",
        tail_model="Want a tip for keeping it moist?",
        reference=(
            "Classic banana bread: bake at 350F (175C) for 55-65 minutes. Check "
            "doneness with a toothpick inserted in the center; it should come out "
            "clean or with a few moist crumbs. Tent with foil if browning too fast."
        ),
        tone_hz=300,
    ),
    dict(
        name="ex06_visa_requirement",
        system_prompt=None,
        user_question="Do US citizens need a visa to visit Japan for two weeks?",
        body_model="No, US citizens can visit Japan visa free for up to ninety days for tourism.",
        tail_model="Just make sure your passport is valid for the full stay.",
        reference=(
            "Japan visa policy: citizens of the United States may enter Japan "
            "visa-free for short-term stays of up to 90 days for tourism or business, "
            "provided their passport is valid for the duration of the stay."
        ),
        tone_hz=320,
    ),
    dict(
        name="ex07_dosage_info",
        system_prompt="You are a pharmacy assistant. Always recommend confirming with a pharmacist.",
        user_question="What's the usual adult dose for ibuprofen tablets?",
        body_model="The typical adult dose is two hundred to four hundred milligrams every four to six hours, not exceeding twelve hundred milligrams a day without medical advice.",
        tail_model="Please confirm with a pharmacist if you're taking other medication.",
        reference=(
            "Ibuprofen adult dosing (OTC): 200-400 mg every 4-6 hours as needed, "
            "maximum 1200 mg per day without medical supervision. Take with food. "
            "Consult a pharmacist if combined with other medications."
        ),
        tone_hz=340,
    ),
    dict(
        name="ex08_exchange_rate",
        system_prompt=None,
        user_question="Roughly how many euros is one hundred US dollars right now?",
        body_model="Today one hundred US dollars is worth about ninety two euros, based on this morning's rate.",
        tail_model="Rates change daily, so double check before a big transfer.",
        reference=(
            "Reference exchange rate snapshot (this morning): 1 USD = 0.92 EUR. "
            "100 USD is therefore approximately 92 EUR. Rates fluctuate throughout "
            "the trading day."
        ),
        tone_hz=360,
    ),
]

# ---------------------------------------------------------------------------
# 2 negative (non-RAG) examples
# ---------------------------------------------------------------------------

REGULAR_EXAMPLES = [
    dict(
        name="ex09_smalltalk_labeled_regular",
        # segment_meta present, but only "regular" spans -- no lookup/filler/body,
        # so no <ref> injection happens (interleaver.py: has_rag_spans is False).
        turns_text=[
            ("USER", "Good morning, how are you doing today?"),
            ("MODEL", "Good morning! I'm doing well, thanks for asking. How about you?"),
            ("USER", "Pretty good, just getting started on some work."),
            ("MODEL", "Nice, hope it goes smoothly today."),
        ],
        tone_hz=380,
    ),
    dict(
        name="ex10_smalltalk_no_segment_meta",
        # No "segment_meta" key at all -- ordinary Moshi-style dialogue data
        # with no RAG annotation, exactly like pre-existing non-RAG training
        # data. interleaver.py handles this fine (segment_meta stays None).
        turns_text=[
            ("USER", "Do you have any plans for the weekend?"),
            ("MODEL", "Not much planned yet, maybe just some reading. You?"),
            ("USER", "Thinking about going hiking if the weather holds up."),
            ("MODEL", "That sounds great, I hope it doesn't rain on you."),
        ],
        tone_hz=400,
    ),
]


def build_rag_example(spec: dict) -> tuple[dict, float, list]:
    # timeline (seconds): USER question -> silence/wait -> <lookup> marker ->
    # <ref> (injected, no real audio) -> MODEL body (first model speech in
    # the sample) -> trailing regular follow-up. No lead/filler audio at all
    # -- the model must not speak before <ref> is injected. `lookup_start`
    # sits right where `body` would have started in this raw (pre-injection)
    # timeline; interleaver.py's _inject_rag_blocks splices <lookup> then
    # <ref> there and pushes this real "body" audio out past both blocks.
    user_start, user_end = 0.0, 4.0
    lookup_start = lookup_end = 4.25          # silence/wait gap, then marker
    body_start, body_end = 4.25, 8.25
    tail_start, tail_end = 8.25, 9.65
    duration_sec = 12.0

    alignments = []
    alignments += words_alignment(spec["user_question"], user_start, user_end, "USER")
    alignments += words_alignment(spec["body_model"], body_start, body_end, "MODEL")
    alignments += words_alignment(spec["tail_model"], tail_start, tail_end, "MODEL")
    alignments.sort(key=lambda a: a[1][0])

    turns = [
        turn("regular", user_start, user_end),
        turn("lookup", lookup_start, lookup_end),
        turn("body", body_start, body_end),
        turn("regular", tail_start, tail_end),
    ]

    sidecar = {
        "alignments": alignments,
        "segment_meta": {"turns": turns},
        "text_conditions": {"reference": spec["reference"]},
    }
    if spec.get("system_prompt"):
        sidecar["metadata"] = {"system_prompt": spec["system_prompt"]}

    speaker_windows = [
        (user_start, user_end, "USER"),
        (body_start, body_end, "MODEL"),
        (tail_start, tail_end, "MODEL"),
    ]
    return sidecar, duration_sec, speaker_windows


def build_regular_example(spec: dict, include_segment_meta: bool) -> tuple[dict, float, list]:
    seg_len = 2.0
    duration_sec = seg_len * len(spec["turns_text"])
    alignments = []
    turns = []
    speaker_windows = []
    t = 0.0
    for speaker, text in spec["turns_text"]:
        alignments += words_alignment(text, t, t + seg_len, speaker)
        turns.append(turn("regular", t, t + seg_len))
        speaker_windows.append((t, t + seg_len, speaker))
        t += seg_len
    alignments.sort(key=lambda a: a[1][0])

    sidecar = {"alignments": alignments}
    if include_segment_meta:
        sidecar["segment_meta"] = {"turns": turns}
    return sidecar, duration_sec, speaker_windows


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    manifest_lines = []

    for spec in RAG_EXAMPLES:
        sidecar, duration_sec, speaker_windows = build_rag_example(spec)
        wav_path = os.path.join(AUDIO_DIR, spec["name"] + ".wav")
        json_path = os.path.join(AUDIO_DIR, spec["name"] + ".new.combined.json")
        write_placeholder_wav_stereo(wav_path, duration_sec, spec["tone_hz"], speaker_windows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        manifest_lines.append({"path": os.path.abspath(wav_path).replace("\\", "/"),
                                "duration": duration_sec})

    for idx, spec in enumerate(REGULAR_EXAMPLES):
        include_meta = idx == 0  # ex09 keeps segment_meta(regular-only), ex10 omits it
        sidecar, duration_sec, speaker_windows = build_regular_example(spec, include_segment_meta=include_meta)
        wav_path = os.path.join(AUDIO_DIR, spec["name"] + ".wav")
        json_path = os.path.join(AUDIO_DIR, spec["name"] + ".new.combined.json")
        write_placeholder_wav_stereo(wav_path, duration_sec, spec["tone_hz"], speaker_windows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        manifest_lines.append({"path": os.path.abspath(wav_path).replace("\\", "/"),
                                "duration": duration_sec})

    manifest_path = os.path.join(OUT_DIR, "train.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for line in manifest_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"wrote {len(manifest_lines)} examples to {OUT_DIR}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
