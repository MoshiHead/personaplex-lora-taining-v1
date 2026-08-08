"""Generates notebooks/convert_20k_dataset_gpu.ipynb. Run once locally;
the .ipynb it produces is what you actually upload/run on RunPod."""
import json
import os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}

cells = []

cells.append(md(
"# Convert 20k text-only RAG dataset -> PersonaPlex training format (multi-GPU)",
"",
"Reads a JSONL file with rows `{id, user, reference, body[, lead, filler, raw_response]}` and produces "
"the exact format `moshi_local/moshi/train.py` / `dataset.py` / `interleaver.py` require. The model "
"never speaks before the reference is injected:",
"",
"```",
"USER audio -> silence/wait -> <lookup> -> <ref> reference text <ref> -> MODEL answer",
"```",
"",
"```",
"<out_dir>/train.jsonl                    manifest: {\"path\", \"duration\"}",
"<out_dir>/audio/<id>.wav                 synthesized speech, mono, 24kHz",
"<out_dir>/audio/<id>.new.combined.json   alignments + segment_meta.turns + text_conditions.reference",
"```",
"",
"**Field mapping:**",
"",
"| input field | becomes |",
"|---|---|",
"| `user` | spoken by `USER`, turn kind `regular` |",
"| *(none)* | synthetic `lookup` marker placed immediately after `user` ends |",
"| `reference` | **not spoken** -> `text_conditions.reference`, spliced in immediately after `<lookup>` -- no filler segment to skip over |",
"| `body` | spoken by `MODEL`, turn kind `body`. This is the **first and only** model audio in the sample |",
"| `lead`, `filler` | **ignored.** Read if present (older-format input rows) but never synthesized -- no lead/filler audio, no lead/filler spans |",
"| `raw_response` | unused for training; kept only for an optional consistency check against `body` |",
"",
"This requires `moshi_local/moshi/interleaver.py`'s `_inject_rag_blocks` (the updated injection path): "
"the old version required a `filler` span to exist at all before it would inject anything. The updated "
"version only requires `lookup` + `body`.",
"",
"**Why GPU TTS instead of the earlier pyttsx3 script:** pyttsx3 is Windows-only SAPI5, CPU-bound, and "
"hung when driven non-interactively. This notebook targets RunPod (Linux, CUDA) and uses **Coqui TTS "
"(VITS/VCTK)** -- a real neural, multi-speaker, GPU-capable TTS model -- instead.",
"",
"**Package note:** the original `coqui-ai/TTS` PyPI package is unmaintained and its last published "
"wheels only support Python <3.12, so `pip install TTS` fails outright on any pod running a newer "
"Python. Section 1 installs the actively-maintained community fork instead, published as `coqui-tts` "
"on PyPI -- it keeps the identical import path (`from TTS.api import TTS`), so no other cell changes.",
"",
"**Second package note:** `coqui-tts` unconditionally imports its XTTS code path on `import TTS` (even "
"though this notebook only uses the VITS/VCTK model), and that path imports `isin_mps_friendly` from "
"`transformers.pytorch_utils` -- a helper that only exists in a narrow `transformers` release window. "
"With no pin, `pip` resolves to whatever the latest `transformers` is, which no longer has it, and the "
"import breaks. Verified directly by inspecting the published wheels: `isin_mps_friendly` exists in "
"`transformers` 4.45.0-5.0.0 inclusive, is absent below 4.45.0, and was removed again starting 5.1.0. "
"Section 1 pins into that verified-safe window explicitly.",
"",
"**On \"100% GPU utilization\":** this workload is many short, independent utterances (4 per row x 20k "
"rows = 80k), not one large matmul, so there's no single kernel that pins a GPU at 100% for the whole "
"job. What this notebook does to maximize utilization:",
"- one worker **process per GPU** (or more, via `WORKERS_PER_GPU`), each with its own CUDA context, so "
"every physical GPU actually runs in parallel instead of sitting idle round-robin,",
"- optional oversubscription (`WORKERS_PER_GPU > 1`) so one worker's CPU-side text/IO work doesn't leave "
"its GPU idle between calls,",
"- every worker is fed its own independent shard, no cross-process waiting.",
"",
"Set `NUM_GPUS` / `WORKERS_PER_GPU` in Section 2 to match your pod (4x4090 -> `NUM_GPUS=4`; 6xA4000 -> "
"`NUM_GPUS=6`) -- auto-detected by default, override if you want.",
"",
"**Run order:** Section 1 -> 2 -> 3 -> 4 -> 5 (with `LIMIT` small first!) -> 6 (validate) -> only then "
"bump `LIMIT` to `None` and re-run 4-6 for the full 20k.",
))

cells.append(md("## 1. Environment setup (run once per pod)"))

cells.append(code(
"# espeak-ng is a native OS binary (not a pip package) that VITS/VCTK's default phonemizer",
"# shells out to for text -> phoneme conversion. Without it TTS.api.TTS(...) raises:",
"#   FileNotFoundError: [!] No espeak backend found. Install espeak-ng or espeak to your system.",
"# RunPod containers run as root, so no sudo needed.",
"!apt-get -qq update && apt-get -qq install -y espeak-ng > /dev/null",
"!espeak-ng --version",
"",
"!pip install -q -U pip",
"# coqui-tts = maintained fork of the abandoned 'TTS' package, same 'TTS.api' import path,",
"# supports modern Python (3.12/3.13) unlike the original which is capped <3.12.",
"#",
"# transformers is pinned to a verified-safe window: coqui-tts's XTTS import path (pulled",
"# in unconditionally by `import TTS`, even though we only use VITS/VCTK) needs",
"# transformers.pytorch_utils.isin_mps_friendly, which only exists in transformers",
"# 4.45.0-5.0.0 (confirmed by inspecting the published wheels directly -- absent below",
"# 4.45.0, removed again from 5.1.0 onward). Without this pin, pip resolves to the latest",
"# transformers and the import breaks with: ImportError: cannot import name",
"# 'isin_mps_friendly' from 'transformers.pytorch_utils'",
"!pip install -q coqui-tts \"transformers>=4.45.0,<5.1.0\" soundfile tqdm sphn",
"!python -c \"import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())\"",
"!python -c \"import transformers; print('transformers', transformers.__version__)\"",
"!python -c \"from TTS.api import TTS; print('TTS import OK')\"",
))

cells.append(code(
"import torch",
"",
"detected_gpus = torch.cuda.device_count()",
"print(f'Detected {detected_gpus} CUDA device(s):')",
"for i in range(detected_gpus):",
"    print(f'  cuda:{i} -> {torch.cuda.get_device_name(i)}')",
"assert detected_gpus > 0, 'No CUDA GPUs visible to torch -- check the RunPod GPU pod / drivers.'",
))

cells.append(md("## 2. Config -- edit these for your run"))

cells.append(code(
"import os",
"",
"# --- paths -----------------------------------------------------------------",
"INPUT_JSONL = '/workspace/data/rag_20k.jsonl'             # <-- point this at your real 20k file",
"OUT_DIR     = '/workspace/data/rag_from_20k'               # output dataset dir (train.jsonl + audio/)",
"REPO_ROOT   = '/workspace/personaplex-lora-taining-try'    # this repo, for Section 6 validation only",
"",
"# --- GPU / parallelism ------------------------------------------------------",
"NUM_GPUS        = detected_gpus     # override e.g. NUM_GPUS = 4   or   NUM_GPUS = 6",
"GPU_IDS         = list(range(NUM_GPUS))",
"WORKERS_PER_GPU = 1                  # raise to 2-3 to keep each GPU busier (uses more VRAM per GPU)",
"",
"# --- TTS ----------------------------------------------------------------------",
"TTS_MODEL_NAME  = 'tts_models/en/vctk/vits'   # multi-speaker, no reference audio needed, GPU-capable",
"SAMPLE_RATE     = 24000                        # must match loaders.SAMPLE_RATE (what mimi expects)",
"SEGMENT_GAP_SEC = 0.25                         # silence inserted between spoken segments",
"",
"# --- run size --------------------------------------------------------------------",
"LIMIT = 4   # <-- SMOKE TEST FIRST: only the first 4 rows. Set to None for the full 20k once verified.",
"",
"os.makedirs(os.path.join(OUT_DIR, 'audio'), exist_ok=True)",
"print('OUT_DIR:', OUT_DIR)",
))

cells.append(md(
"## 3. Worker module",
"",
"Written with `%%writefile` to a real, importable `.py` file -- Jupyter's `__main__` functions are not "
"reliably picklable across processes, which is the standard multiprocessing-in-a-notebook gotcha. "
"`SAMPLE_RATE`/`SEGMENT_GAP_SEC` are duplicated here as literal constants (keep in sync with Section 2 "
"if you change them there -- they can't be interpolated into a `%%writefile` cell).",
))

cells.append(code(
"%%writefile _rag_tts_worker.py",
"\"\"\"Worker module for convert_20k_dataset_gpu.ipynb -- one instance of process_shard()",
"runs per spawned subprocess, each pinned to a single GPU.\"\"\"",
"import json",
"import os",
"import wave",
"",
"import numpy as np",
"",
"SAMPLE_RATE = 24000        # keep in sync with the notebook's Config cell",
"SEGMENT_GAP_SEC = 0.25     # keep in sync with the notebook's Config cell",
"",
"",
"def words_alignment(text, start_sec, end_sec, speaker):",
"    words = text.split()",
"    if not words:",
"        return []",
"    n = len(words)",
"    step = (end_sec - start_sec) / n",
"    out = []",
"    for i, w in enumerate(words):",
"        w_start = start_sec + i * step",
"        w_end = w_start + step * 0.9",
"        out.append([w, [round(w_start, 3), round(w_end, 3)], speaker])",
"    return out",
"",
"",
"def write_wav_stereo(path, agent_channel, user_channel):",
"    # STEREO is required, not cosmetic: interleaver.py calls",
"    # mimi.encode(audio_tensor[:, None]) on this file's [2, T] samples, which",
"    # treats the 2 channels as 2 batch items each independently encoded to 8",
"    # codebooks (MimiModel.encode: [B,C,T] -> [B,K,T], K=8), then flattens to",
"    # [1, 16, T] via .view(1, -1, T) -- channel 0 -> codebooks 0-7 (agent),",
"    # channel 1 -> codebooks 8-15 (user), matching the SILENCE_TOKENS(agent)/",
"    # SINE_TOKENS(user) convention already hardcoded in _build_forced_text_frames.",
"    # A mono file only produces 8 codebooks total and raises:",
"    #   RuntimeError: Sizes of tensors must match except in dimension 2.",
"    #   Expected size 8 but got size 16 for tensor number 1 in the list.",
"    n = max(len(agent_channel), len(user_channel))",
"    agent_channel = np.pad(agent_channel, (0, n - len(agent_channel)))",
"    user_channel = np.pad(user_channel, (0, n - len(user_channel)))",
"    stereo = np.stack([agent_channel, user_channel], axis=-1)  # [T, 2] for interleaved PCM",
"    pcm16 = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype('<i2')",
"    with wave.open(path, 'wb') as w:",
"        w.setnchannels(2)",
"        w.setsampwidth(2)",
"        w.setframerate(SAMPLE_RATE)",
"        w.writeframes(pcm16.tobytes())",
"",
"",
"def _load_tts(gpu_id, model_name):",
"    # Local imports: torch/TTS must not be imported at module import time in the",
"    # parent process -- only after CUDA_VISIBLE_DEVICES is pinned in this child.",
"    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)",
"    from TTS.api import TTS",
"    return TTS(model_name).to('cuda:0')  # cuda:0 here == the physical GPU pinned above",
"",
"",
"def _synth(tts, text, speaker):",
"    import sphn",
"    wav = tts.tts(text=text, speaker=speaker)",
"    audio = np.asarray(wav, dtype=np.float32)",
"    try:",
"        sr = tts.synthesizer.output_sample_rate",
"    except Exception:",
"        sr = 22050",
"    if sr != SAMPLE_RATE:",
"        audio = sphn.resample(audio[None, :], src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)[0]",
"    return audio.astype(np.float32)",
"",
"",
"def _wav_duration(path):",
"    with wave.open(path, 'rb') as f:",
"        return f.getnframes() / f.getframerate()",
"",
"",
"def process_shard(args):",
"    shard_rows, gpu_id, out_dir, worker_tag, model_name, user_speaker, model_speaker = args",
"    audio_dir = os.path.join(out_dir, 'audio')",
"    tts = _load_tts(gpu_id, model_name)",
"    gap = np.zeros(int(SEGMENT_GAP_SEC * SAMPLE_RATE), dtype=np.float32)",
"",
"    manifest = []",
"    n_done, n_skipped = 0, 0",
"    for row in shard_rows:",
"        ex_id = str(row['id'])",
"        wav_path = os.path.join(audio_dir, ex_id + '.wav')",
"        json_path = os.path.join(audio_dir, ex_id + '.new.combined.json')",
"",
"        if os.path.exists(wav_path) and os.path.exists(json_path):",
"            # resume support: skip work already done in a previous run",
"            n_skipped += 1",
"            manifest.append({'path': os.path.abspath(wav_path), 'duration': round(_wav_duration(wav_path), 3)})",
"            continue",
"",
"        # Only two spoken segments: user question, then the model's grounded",
"        # answer. No lead/filler audio -- the model must not speak between",
"        # the user's turn and the injected <ref>. Any 'lead'/'filler' fields",
"        # in the source row (older-format input) are simply not read here.",
"        segs = [",
"            (row['user'], user_speaker, 'USER', 'regular'),",
"            (row['body'], model_speaker, 'MODEL', 'body'),",
"        ]",
"        synthesized = [(role, kind, _synth(tts, text, speaker_id)) for text, speaker_id, role, kind in segs]",
"",
"        # Timeline math is unaffected by the stereo change below -- turn/",
"        # alignment timing describes WHEN something happens, not which",
"        # channel it's on. What changes is that each speaker's clip is",
"        # placed at its time offset on ITS OWN channel (silence elsewhere),",
"        # not concatenated end-to-end into one channel. Placement (start_sec,",
"        # samples) is computed once here and reused below to build the",
"        # channel buffers, rather than recomputed in a second pass.",
"        turns, alignments, placements = [], [], []",
"        t = 0.0",
"        for i, (role, kind, samples) in enumerate(synthesized):",
"            if i > 0:",
"                t += len(gap) / SAMPLE_RATE",
"            start_sec = t",
"            placements.append((role, start_sec, samples))",
"            t += len(samples) / SAMPLE_RATE",
"            end_sec = t",
"            if kind == 'body':",
"                # <lookup> marker sits exactly at body's start in the raw",
"                # (pre-injection) timeline -- no filler speech to place it",
"                # before, no real audio time consumed by the marker itself;",
"                # interleaver.py splices <lookup> then <ref> there and",
"                # pushes this real 'body' audio out past both blocks.",
"                turns.append({'kind': 'lookup', 'start_sec': round(start_sec, 3), 'end_sec': round(start_sec, 3)})",
"            turns.append({'kind': kind, 'start_sec': round(start_sec, 3), 'end_sec': round(end_sec, 3)})",
"            text_for_row = row['user'] if role == 'USER' else row['body']",
"            alignments += words_alignment(text_for_row, start_sec, end_sec, role)",
"",
"        total_samples = int(round(t * SAMPLE_RATE))",
"        agent_channel = np.zeros(total_samples, dtype=np.float32)",
"        user_channel = np.zeros(total_samples, dtype=np.float32)",
"        for role, start_sec, samples in placements:",
"            start_sample = int(round(start_sec * SAMPLE_RATE))",
"            target = agent_channel if role == 'MODEL' else user_channel",
"            target[start_sample:start_sample + len(samples)] = samples",
"",
"        duration_sec = total_samples / SAMPLE_RATE",
"        write_wav_stereo(wav_path, agent_channel, user_channel)",
"",
"        sidecar = {",
"            'alignments': sorted(alignments, key=lambda a: a[1][0]),",
"            'segment_meta': {'turns': turns},",
"            'text_conditions': {'reference': row['reference']},",
"        }",
"        with open(json_path, 'w', encoding='utf-8') as f:",
"            json.dump(sidecar, f, ensure_ascii=False, indent=2)",
"",
"        manifest.append({'path': os.path.abspath(wav_path), 'duration': round(duration_sec, 3)})",
"        n_done += 1",
"",
"    part_path = os.path.join(out_dir, 'train.part_' + str(worker_tag) + '.jsonl')",
"    with open(part_path, 'w', encoding='utf-8') as f:",
"        for line in manifest:",
"            f.write(json.dumps(line, ensure_ascii=False) + '\\n')",
"    return worker_tag, len(manifest), n_done, n_skipped, part_path",
))

cells.append(md(
"## 4. Pick speakers and build shards",
"",
"VITS/VCTK ships ~109 built-in speakers, no reference audio needed. We load the model once here just to "
"read `tts.speakers` and pick two distinct IDs for USER vs MODEL.",
))

cells.append(code(
"from TTS.api import TTS as _TTS_probe",
"",
"_probe = _TTS_probe(TTS_MODEL_NAME)",
"speakers = list(_probe.speakers) if _probe.speakers else []",
"assert len(speakers) >= 2, f'expected a multi-speaker model, got: {speakers}'",
"USER_SPEAKER, MODEL_SPEAKER = speakers[0], speakers[1]",
"print('USER_SPEAKER  =', USER_SPEAKER)",
"print('MODEL_SPEAKER =', MODEL_SPEAKER)",
"del _probe",
))

cells.append(code(
"import json",
"",
"rows = []",
"with open(INPUT_JSONL, encoding='utf-8') as f:",
"    for i, line in enumerate(f):",
"        if LIMIT is not None and i >= LIMIT:",
"            break",
"        line = line.strip()",
"        if line:",
"            rows.append(json.loads(line))",
"print(f'loaded {len(rows)} rows (LIMIT={LIMIT})')",
"",
"n_workers = len(GPU_IDS) * WORKERS_PER_GPU",
"shards = [rows[i::n_workers] for i in range(n_workers)]",
"shard_gpu_ids = [GPU_IDS[i % len(GPU_IDS)] for i in range(n_workers)]",
"for i, s in enumerate(shards):",
"    print(f'  worker {i} -> gpu {shard_gpu_ids[i]} -> {len(s)} rows')",
))

cells.append(md("## 5. Run conversion across all GPUs in parallel"))

cells.append(code(
"import multiprocessing as mp",
"import sys",
"import time",
"from concurrent.futures import ProcessPoolExecutor, as_completed",
"",
"sys.path.insert(0, os.getcwd())",
"import _rag_tts_worker as W  # the file written by Section 3's %%writefile",
"",
"ctx = mp.get_context('spawn')  # required: CUDA + multiprocessing needs spawn, not fork",
"",
"jobs = []",
"for i, (shard, gpu_id) in enumerate(zip(shards, shard_gpu_ids)):",
"    if not shard:",
"        continue",
"    jobs.append((shard, gpu_id, OUT_DIR, i, TTS_MODEL_NAME, USER_SPEAKER, MODEL_SPEAKER))",
"",
"t0 = time.time()",
"results = []",
"with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as ex:",
"    futures = {ex.submit(W.process_shard, job): job[3] for job in jobs}",
"    for fut in as_completed(futures):",
"        worker_tag, n_total, n_done, n_skipped, part_path = fut.result()",
"        print(f'[worker {worker_tag}] {n_total} rows ({n_done} synthesized, {n_skipped} reused) -> {part_path}')",
"        results.append(part_path)",
"",
"print(f'all shards finished in {time.time()-t0:.1f}s')",
))

cells.append(md(
"## 5b. Merge shard manifests into the final `train.jsonl`",
"",
"Also reports duration stats and the `--duration-sec` to pass to `train.py`. Picking too small a value "
"silently **truncates** longer examples (`interleaver.py` hard-cuts audio at "
"`duration_sec * frame_rate` frames), so size it to your longest real example, not the average.",
))

cells.append(code(
"import glob",
"",
"manifest_lines = []",
"for part_path in sorted(glob.glob(os.path.join(OUT_DIR, 'train.part_*.jsonl'))):",
"    with open(part_path, encoding='utf-8') as f:",
"        for line in f:",
"            line = line.strip()",
"            if line:",
"                manifest_lines.append(json.loads(line))",
"",
"manifest_path = os.path.join(OUT_DIR, 'train.jsonl')",
"with open(manifest_path, 'w', encoding='utf-8') as f:",
"    for line in manifest_lines:",
"        f.write(json.dumps(line, ensure_ascii=False) + '\\n')",
"",
"durations = sorted(m['duration'] for m in manifest_lines)",
"print(f'wrote {len(manifest_lines)} examples -> {manifest_path}')",
"if durations:",
"    p50 = durations[len(durations) // 2]",
"    p99 = durations[min(len(durations) - 1, int(len(durations) * 0.99))]",
"    print(f'duration stats: min={durations[0]:.2f}s p50={p50:.2f}s p99={p99:.2f}s max={durations[-1]:.2f}s')",
"    print(f'recommended --duration-sec for train.py: >= {durations[-1]:.1f} (max observed)')",
))

cells.append(md(
"## 6. Validate against the *actual* repo parsing code",
"",
"Runs `scripts/validate_rag_dataset.py` (shipped alongside this notebook in the repo) against the "
"generated dataset. It loads every sidecar through the real `InterleavedTokenizer._parse_segment_meta` "
"**and actually runs `_inject_rag_blocks`** (not just a heuristic span check), then confirms all five "
"conditions for the current format:",
"",
"1. user span exists",
"2. `<lookup>` exists immediately after the user turn (no lead/filler speech in between)",
"3. `<ref>` precedes the answer (confirmed by running real injection, not inferred)",
"4. no lead/filler spans exist",
"5. the answer (`body`) follows `<ref>`",
"",
"A clean run here means the RAG examples will actually inject `<lookup>` -> `<ref>` -> `body` at training "
"time in exactly that order, not just that the JSON happens to parse.",
))

cells.append(code(
"sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))",
"from validate_rag_dataset import validate",
"",
"ok = validate(dataset_dir=OUT_DIR, repo_root=REPO_ROOT, tokenizer_path=None)",
"print()",
"print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED -- see failures above before training on this data')",
))

cells.append(md(
"## Next steps",
"",
"1. Re-run Sections 4-6 with `LIMIT = None` once the smoke test above looks right.",
"2. Point training at the output:",
"   ```bash",
"   python -m moshi.train --train-data /workspace/data/rag_from_20k --duration-sec <max observed above> ...",
"   ```",
"3. What this notebook does **not** fix, carried over from the earlier discussion:",
"   - VITS/VCTK audio is still synthetic TTS, not real recorded conversational speech -- much more "
"natural than the earlier robotic SAPI5 test, but still not human speech.",
"   - No wrong/irrelevant-reference negatives are generated here -- every example's `<ref>` block is the "
"correct one for its `body`. Teaching the model to distrust a bad retrieval requires deliberately "
"constructing mismatched (reference, body) pairs, which this notebook does not do.",
"   - Pick `--duration-sec` from the **max** observed duration (printed above), not the median, or "
"longer examples get silently truncated by `interleaver.py`.",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "convert_20k_dataset_gpu.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_path)
