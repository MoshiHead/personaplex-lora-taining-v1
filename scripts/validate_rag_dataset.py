"""
validate_rag_dataset.py
========================
Validates a converted RAG dataset (train.jsonl + audio/*.new.combined.json)
against the ACTUAL repo parsing/injection code (moshi_local/moshi/interleaver.py)
-- not a reimplementation of the rules, the real thing -- and reports the five
checks for the current (no lead/filler) format:

    1. user span exists                     (a "regular" turn is present)
    2. lookup exists immediately after user  (no lead/filler speech in between)
    3. ref precedes the answer               (confirmed by actually running injection)
    4. no lead/filler spans exist            (deprecated kinds absent)
    5. answer (body) follows ref             (confirmed by actually running injection)

Usage:
    python validate_rag_dataset.py --dataset-dir data/demo_rag_dataset
    python validate_rag_dataset.py --dataset-dir data/rag_from_20k --tokenizer /path/to/tokenizer_spm_32k_3.model

Without --tokenizer, a lightweight stand-in tokenizer is used (whitespace
split -> deterministic fake ids) so this can run without network access or
downloaded model weights. This is sufficient to validate frame ORDERING and
ADJACENCY (what these checks are actually about) but not exact <ref> token
counts -- pass --tokenizer for a byte-for-byte match with real training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load_interleaver(repo_root: str):
    sys.path.insert(0, os.path.join(repo_root, "moshi_local"))
    from moshi.interleaver import InterleavedTokenizer, Interleaver
    return InterleavedTokenizer, Interleaver


class _FakeTokenizer:
    """Whitespace-split, deterministic fake ids -- good enough to exercise
    frame math without needing the real SentencePiece model on disk."""
    def encode(self, text):
        return [1000 + (hash(w) % 30000) for w in text.split()] or [0]
    def decode(self, ids):
        return " ".join(str(i) for i in ids)
    def bos_id(self):
        return 1
    def eos_id(self):
        return 2


def _real_tokenizer(path: str):
    import sentencepiece
    return sentencepiece.SentencePieceProcessor(path)


def validate(dataset_dir: str, repo_root: str, tokenizer_path: str | None, frame_rate: float = 12.5):
    InterleavedTokenizer, Interleaver = _load_interleaver(repo_root)
    import torch

    tok = _real_tokenizer(tokenizer_path) if tokenizer_path else _FakeTokenizer()
    interleaver = Interleaver(
        tokenizer=tok, audio_frame_rate=frame_rate, text_padding=32000,
        end_of_text_padding=32001, zero_padding=32002, device="cpu",
    )

    class _IT(InterleavedTokenizer):
        def __init__(self, duration_sec, frame_rate, interleaver):
            self.duration_sec = duration_sec
            class _M:
                pass
            self.mimi = _M()
            self.mimi.frame_rate = frame_rate
            self.interleaver = interleaver
            self.max_ref_tokens = 500

    manifest_path = os.path.join(dataset_dir, "train.jsonl")
    entries = [json.loads(l) for l in open(manifest_path, encoding="utf-8") if l.strip()]
    if not entries:
        print(f"no entries in {manifest_path}")
        return False

    max_dur = max(e["duration"] for e in entries)
    it = _IT(duration_sec=max_dur, frame_rate=frame_rate, interleaver=interleaver)

    n_total = n_pass = 0
    n_regular_only = 0  # non-RAG negative examples are valid, just skip RAG-specific checks
    failures = []

    for e in entries:
        n_total += 1
        wav_path = e["path"]
        sidecar_path = os.path.splitext(wav_path)[0] + ".new.combined.json"
        name = os.path.basename(wav_path)

        if not os.path.exists(sidecar_path):
            failures.append((name, "sidecar file missing"))
            continue

        data = json.load(open(sidecar_path, encoding="utf-8"))
        raw_meta = data.get("segment_meta")
        ref = data.get("text_conditions", {}).get("reference", "") or ""

        if raw_meta is None:
            n_regular_only += 1
            n_pass += 1  # plain non-RAG sample, nothing to check
            continue

        meta = it._parse_segment_meta(raw_meta, chunk_start_sec=0.0)
        if meta is None:
            failures.append((name, "segment_meta present but parsed to no spans"))
            continue

        regular = meta.spans_of("regular")
        lookup = meta.spans_of("lookup")
        body = meta.spans_of("body")
        lead = meta.spans_of("lead")
        filler = meta.spans_of("filler")

        if not (lookup or body or ref):
            n_regular_only += 1
            n_pass += 1  # negative example (no RAG turn at all) -- valid
            continue

        checks = {
            "1. user span exists": bool(regular),
            "2. lookup immediately after user (no lead/filler between)": (
                bool(lookup) and (not regular or lookup[0].start_frame >= regular[0].end_frame)
            ),
            "4. no lead/filler spans exist": not lead and not filler,
        }

        if not all(checks.values()):
            for k, v in checks.items():
                if not v:
                    failures.append((name, k))
            continue

        # checks 3 & 5 require actually running injection
        T = int(round(max(t.end_frame for t in meta.turns) + 10))
        text_tokens = torch.full((1, 1, T), interleaver.zero_padding, dtype=torch.long)
        audio_tokens = torch.zeros((1, 16, T), dtype=torch.long)
        try:
            _, _, meta2, inj_lookup, inj_ref = it._inject_rag_blocks(
                text_tokens=text_tokens, audio_tokens=audio_tokens, segment_meta=meta,
                reference_text=ref, this_num_audio_frames=T, path=wav_path, start_sec=0.0,
            )
        except Exception as exc:
            failures.append((name, f"injection raised: {exc}"))
            continue

        if not (inj_lookup and inj_ref):
            failures.append((name, "injection did not fire (missing lookup/body spans or empty reference)"))
            continue

        lk = meta2.spans_of("lookup")[0]
        rf = meta2.spans_of("ref")[0]
        bd = meta2.spans_of("body")[0]

        checks_2 = {
            "3. ref precedes answer (starts exactly where lookup block ends)": rf.start_frame == lk.end_frame,
            "5. answer (body) follows ref (starts exactly where ref block ends)": bd.start_frame == rf.end_frame,
        }
        if not all(checks_2.values()):
            for k, v in checks_2.items():
                if not v:
                    failures.append((name, k))
            continue

        n_pass += 1

    print(f"checked {n_total} examples: {n_pass} passed, {len(failures)} failed "
          f"({n_regular_only} were non-RAG negative examples, auto-pass)")
    if failures:
        print("\nfailures:")
        for name, reason in failures:
            print(f"  {name}: {reason}")
    return len(failures) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="dir containing train.jsonl + audio/")
    ap.add_argument("--repo-root", default=os.path.join(os.path.dirname(__file__), ".."),
                     help="repo root containing moshi_local/ (for importing the real interleaver.py)")
    ap.add_argument("--tokenizer", default=None, help="path to tokenizer_spm_32k_3.model for an exact check")
    ap.add_argument("--frame-rate", type=float, default=12.5)
    args = ap.parse_args()

    ok = validate(args.dataset_dir, args.repo_root, args.tokenizer, args.frame_rate)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
