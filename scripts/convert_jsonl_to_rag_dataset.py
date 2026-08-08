"""
convert_jsonl_to_rag_dataset.py
================================
Converts a text-only RAG conversation dataset (fields: id, user, reference,
body[, lead, filler, raw_response]) into the audio-grounded format the
training pipeline actually requires:

    <out_dir>/train.jsonl                    manifest: {"path", "duration"}
    <out_dir>/audio/<id>.wav                 synthesized speech, STEREO, 24kHz --
                                              channel 0 = agent/MODEL audio, channel 1 =
                                              USER audio, both spanning the full duration
                                              simultaneously (required -- see write_wav_stereo)
    <out_dir>/audio/<id>.new.combined.json   alignments + segment_meta + text_conditions

The model never speaks before the reference is injected:

    USER audio -> silence/wait -> <lookup> -> <ref> reference text <ref> -> MODEL body

Field mapping:

    user      -> spoken by "USER", turn kind "regular"
    (none)    -> synthetic "lookup" marker, placed right after "user" ends
                 (no audio is spoken for it -- the <lookup> tag is injected
                 as text with forced-silent audio by InterleavedTokenizer at
                 train time, same as <ref>). Zero-width in the sidecar is
                 fine: interleaver.py's _inject_rag_blocks re-points this
                 span to the actual injected extent itself.
    reference -> NOT spoken. Goes into text_conditions.reference; spliced in
                 as a silent <ref>...<ref> text block immediately after
                 <lookup> -- there is no filler segment to skip over.
    body      -> spoken by "MODEL", turn kind "body". This is the FIRST and
                 ONLY model audio in the sample.
    lead, filler -> IGNORED. If your source rows still carry these fields
                 (e.g. from an older data format), they are read but never
                 synthesized -- no lead/filler audio, no lead/filler spans.
                 Requires interleaver.py's updated _inject_rag_blocks, which
                 only needs `lookup` + `body` spans (the old version required
                 `filler` too and would silently skip injection without it).
    raw_response -> not needed for training; used here only as an optional
                 sanity check that body matches its [BODY] portion.

Audio is synthesized per-segment with pyttsx3 (offline SAPI5 TTS, reused
from moshi_local/moshi/build_index.py's text_to_audio()), each segment
resampled to 24kHz, concatenated in order with a short silence gap in
between (the "silence/wait" between the user's turn and the <lookup>
marker). Segment start/end seconds come directly from the real synthesized
clip lengths -- no timing is guessed.

IMPORTANT CAVEAT: pyttsx3/SAPI5 voices are robotic, single-speaker-per-voice,
not naturalistic full-duplex conversational speech. This makes the pipeline
run correctly end-to-end and is fine for small-scale validation, but for
20k examples intended to actually teach the model good RAG behavior, swap
`synthesize()` for a proper neural multi-speaker TTS (XTTS-v2, Bark, Piper
with multiple voices, or a commercial API) -- the function boundary here is
built so that's a one-function change, not a rewrite of the alignment/turn
logic.

Usage:
    python convert_jsonl_to_rag_dataset.py \
        --input data/sample_20k_input.jsonl \
        --out-dir data/rag_from_20k \
        --limit 2          # drop --limit to process the whole file
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 24000     # loaders.SAMPLE_RATE (must match mimi)
SEGMENT_GAP_SEC = 0.25  # silence between spoken segments
USER_VOICE_HINT = "zira"   # substring match against installed SAPI5 voice names
MODEL_VOICE_HINT = "david"


# ---------------------------------------------------------------------------
# TTS backend (swap this out for a better engine if you want real quality)
# ---------------------------------------------------------------------------

_ENGINE = None
_VOICES_BY_ROLE: dict[str, str] = {}


def _init_engine():
    global _ENGINE, _VOICES_BY_ROLE
    if _ENGINE is not None:
        return
    import pyttsx3
    _ENGINE = pyttsx3.init()
    voices = _ENGINE.getProperty("voices")
    user_id = model_id = voices[0].id
    for v in voices:
        name = v.name.lower()
        if USER_VOICE_HINT in name:
            user_id = v.id
        if MODEL_VOICE_HINT in name:
            model_id = v.id
    _VOICES_BY_ROLE = {"USER": user_id, "MODEL": model_id}


def synthesize(text: str, role: str) -> np.ndarray:
    """Returns float32 mono samples at SAMPLE_RATE for `text` spoken by `role`
    ("USER" or "MODEL"). Swap this function's body for a different TTS
    engine; everything downstream only depends on this contract."""
    import soundfile as sf
    import sphn

    _init_engine()
    _ENGINE.setProperty("voice", _VOICES_BY_ROLE.get(role, _VOICES_BY_ROLE["MODEL"]))
    _ENGINE.setProperty("rate", 165)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        _ENGINE.save_to_file(text, tmp_path)
        _ENGINE.runAndWait()
        audio, sr = sf.read(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = sphn.resample(audio[None, :], src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)[0]
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Segment / turn assembly
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    text: str
    role: str          # "USER" | "MODEL"
    kind: str | None   # turn kind to emit, or None (e.g. plain "regular")
    samples: np.ndarray = None
    start_sec: float = 0.0
    end_sec: float = 0.0


def words_alignment(text: str, start_sec: float, end_sec: float, speaker: str) -> list:
    words = text.split()
    if not words:
        return []
    n = len(words)
    step = (end_sec - start_sec) / n
    out = []
    for i, w in enumerate(words):
        w_start = start_sec + i * step
        w_end = w_start + step * 0.9
        out.append([w, [round(w_start, 3), round(w_end, 3)], speaker])
    return out


def write_wav_stereo(path: str, agent_channel: np.ndarray, user_channel: np.ndarray) -> None:
    """Writes a 2-channel WAV: channel 0 = agent/MODEL audio, channel 1 =
    USER audio, both spanning the FULL sample duration simultaneously (not
    concatenated end-to-end). This is required, not cosmetic: interleaver.py
    calls `mimi.encode(audio_tensor[:, None])` on this file's [2, T] samples,
    which treats the 2 channels as 2 batch items, each independently encoded
    to 8 codebooks (MimiModel.encode: [B,C,T] -> [B,K,T], K=8 per item), then
    `.view(1, -1, T)` flattens the [2, 8, T] result into [1, 16, T] --
    channel 0 becomes codebooks 0-7, channel 1 becomes codebooks 8-15. That
    16-wide layout must match the [agent(8), user(8)] convention already
    hardcoded in `_build_forced_text_frames`'s SILENCE_TOKENS (agent)
    /SINE_TOKENS (user) blocks, or a mono file 8 codebooks short of what the
    rest of the pipeline expects, raising:
        RuntimeError: Sizes of tensors must match except in dimension 2.
        Expected size 8 but got size 16 for tensor number 1 in the list.
    """
    n = max(len(agent_channel), len(user_channel))
    agent_channel = np.pad(agent_channel, (0, n - len(agent_channel)))
    user_channel = np.pad(user_channel, (0, n - len(user_channel)))

    stereo = np.stack([agent_channel, user_channel], axis=-1)  # [T, 2] for interleaved PCM
    clipped = np.clip(stereo, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())


def convert_one(row: dict, audio_dir: str) -> tuple[dict, float, str]:
    ex_id = str(row["id"])
    gap_samples = int(SEGMENT_GAP_SEC * SAMPLE_RATE)

    # Only two spoken segments: the user's question, then the model's
    # grounded answer. No lead/filler audio is synthesized at all -- the
    # model must not speak between the user's turn and the injected <ref>.
    segs = [
        Segment(row["user"], "USER", "regular"),
        Segment(row["body"], "MODEL", "body"),
    ]
    for s in segs:
        s.samples = synthesize(s.text, s.role)

    # Timeline (seconds/samples) is unaffected by the stereo change below --
    # turn/alignment timing describes WHEN something happens, not which
    # channel it's on. What changes is that each speaker's clip is placed at
    # its time offset on ITS OWN channel (silence elsewhere), not
    # concatenated end-to-end into one channel.
    t = 0.0
    turns = []
    alignments = []
    for i, s in enumerate(segs):
        if i > 0:
            t += gap_samples / SAMPLE_RATE
        s.start_sec = t
        t += len(s.samples) / SAMPLE_RATE
        s.end_sec = t

        if s.kind == "body":
            # <lookup> marker sits exactly at body's start in the raw
            # (pre-injection) timeline -- there is no filler speech to place
            # it before, and no real audio time is consumed by the marker
            # itself; interleaver.py splices <lookup> then <ref> there and
            # pushes this real "body" audio out past both injected blocks.
            turns.append({"kind": "lookup", "start_sec": round(s.start_sec, 3),
                          "end_sec": round(s.start_sec, 3)})
        turns.append({"kind": s.kind, "start_sec": round(s.start_sec, 3),
                      "end_sec": round(s.end_sec, 3)})
        alignments += words_alignment(s.text, s.start_sec, s.end_sec, s.role)

    total_samples = int(round(t * SAMPLE_RATE))
    agent_channel = np.zeros(total_samples, dtype=np.float32)
    user_channel = np.zeros(total_samples, dtype=np.float32)
    for s in segs:
        start_sample = int(round(s.start_sec * SAMPLE_RATE))
        end_sample = start_sample + len(s.samples)
        target = agent_channel if s.role == "MODEL" else user_channel
        target[start_sample:end_sample] = s.samples

    duration_sec = total_samples / SAMPLE_RATE

    wav_path = os.path.join(audio_dir, f"{ex_id}.wav")
    json_path = os.path.join(audio_dir, f"{ex_id}.new.combined.json")
    write_wav_stereo(wav_path, agent_channel, user_channel)

    sidecar = {
        "alignments": sorted(alignments, key=lambda a: a[1][0]),
        "segment_meta": {"turns": turns},
        "text_conditions": {"reference": row["reference"]},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)

    # optional sanity check against raw_response, warn only -- only the
    # [BODY] portion matters now, lead/filler are never synthesized
    raw = row.get("raw_response", "")
    if raw and "[BODY]" in raw:
        body_part = raw.split("[BODY]", 1)[1]
        if "".join(row["body"].split()) != "".join(body_part.split()):
            print(f"  [warn] {ex_id}: body does not exactly match raw_response's [BODY] section")

    return {"path": os.path.abspath(wav_path).replace("\\", "/"), "duration": round(duration_sec, 3)}, duration_sec, ex_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_lines = []
    durations = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entry, dur, ex_id = convert_one(row, audio_dir)
            manifest_lines.append(entry)
            durations.append(dur)
            print(f"[{i+1}] {ex_id}: duration={dur:.2f}s")

    manifest_path = os.path.join(args.out_dir, "train.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for line in manifest_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    if durations:
        durations.sort()
        p50 = durations[len(durations) // 2]
        p99 = durations[min(len(durations) - 1, int(len(durations) * 0.99))]
        print(f"\nwrote {len(manifest_lines)} examples -> {manifest_path}")
        print(f"duration stats: min={durations[0]:.2f}s p50={p50:.2f}s p99={p99:.2f}s max={durations[-1]:.2f}s")
        print(f"recommended --duration-sec for train.py: >= {durations[-1]:.1f} (max observed), "
              f"or {p99:.1f} (p99) if you're OK dropping/truncating rare long outliers")


if __name__ == "__main__":
    main()
