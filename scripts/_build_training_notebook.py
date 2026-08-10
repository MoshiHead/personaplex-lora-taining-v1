"""Generates notebooks/PersonaPlex_LoRA_Training_RunPod.ipynb."""
import json
import os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}

cells = []

# ---------------------------------------------------------------------------
cells.append(md(
"# PersonaPlex — LoRA RAG Training on RunPod (multi-GPU)",
"",
"Dedicated **training-only** notebook, built from `PersonaPlex_RunPod_RTX5090.ipynb` as a reference for "
"environment setup (workspace layout, system packages, Blackwell/cu130 torch wheels, HF auth). That "
"notebook is for *serving* the model; this one is for **LoRA fine-tuning it** on your RAG dataset via "
"`moshi.train`, launched with `accelerate` across however many GPUs the pod has.",
"",
"### Two things fixed relative to the reference notebook, on purpose",
"",
"1. **Security:** the reference notebook has a Hugging Face access token hardcoded in plain text in its "
"HF-auth cell (with a comment noting it wasn't even the author's own token). **If that notebook has been "
"shared or uploaded anywhere, revoke that token now** at "
"https://huggingface.co/settings/tokens — independently of anything in this notebook. This notebook "
"never hardcodes a token; it only ever prompts via hidden input or reads `HF_TOKEN` from the environment.",
"2. **Correctness:** the training code in this session (`moshi_local/moshi/interleaver.py`'s "
"`_inject_rag_blocks`, `moshi_local/moshi/train.py`'s Accelerate rewrite, `scripts/validate_rag_dataset.py`) "
"only exists in **this local working copy** -- it has not been pushed to the public repo the reference "
"notebook clones. If this notebook just cloned that public URL, it would silently train against the "
"*old* interleaver, which requires a `filler` span that your current dataset format no longer has -- "
"every RAG example would silently fall through to being treated as plain non-RAG dialogue, with **no "
"error**, just a model that never learns to use `<ref>`. Section 5 below is a hard verification gate for "
"exactly this failure mode: it does not let training start unless the fixes are actually present in "
"whatever checkout ends up on the pod.",
"",
"### What you need before running this top-to-bottom",
"",
"- A RunPod pod with one or more GPUs (multi-GPU is auto-detected; works fine with 1 GPU too).",
"- Your converted, validated RAG dataset (`train.jsonl` + `audio/*.wav` + `audio/*.new.combined.json`) "
"uploaded to the pod. Section 6 auto-detects it from a few common locations, or set `DATASET_DIR` "
"manually in that cell.",
"- Either: (a) push this session's local changes to your fork/`REPO_URL` before running this notebook, or "
"(b) upload this local working copy directly to `REPO_DIR` on the pod instead of letting Section 4 clone "
"fresh. Section 5's verification gate tells you clearly which case you're in.",
"- A Hugging Face token with access to `nvidia/personaplex-7b-v1` accepted (Section 8).",
))

cells.append(md("## 1. Environment sanity checks"))
cells.append(code(
"import platform",
"import sys",
"",
"print('Platform:', platform.platform())",
"print('Python:', sys.version)",
"",
"assert sys.version_info >= (3, 10), (",
"    f'PersonaPlex (moshi/pyproject.toml) requires Python >= 3.10, found {sys.version_info}.'",
")",
"print('Python version OK.')",
))

cells.append(md(
"## 1b. GPU detection",
"",
"Counts GPUs at the OS level (before torch is even installed) so every later cell -- the `accelerate "
"launch` command in particular -- can size itself to however many GPUs this pod actually has, with no "
"manual editing.",
))
cells.append(code(
"import subprocess",
"",
"!nvidia-smi",
"",
"try:",
"    gpu_names = subprocess.run(",
"        ['nvidia-smi', '-L'], capture_output=True, text=True, check=True",
"    ).stdout.strip().splitlines()",
"except Exception as exc:",
"    raise RuntimeError(",
"        'nvidia-smi failed -- confirm this pod actually has a GPU attached and the driver is healthy.'",
"    ) from exc",
"",
"NUM_GPUS = len(gpu_names)",
"print(f'Detected {NUM_GPUS} GPU(s):')",
"for line in gpu_names:",
"    print(' ', line)",
"assert NUM_GPUS >= 1, 'No GPUs detected by nvidia-smi.'",
))

# ---------------------------------------------------------------------------
cells.append(md(
"## 2. Persistent storage & path setup",
"",
"Same `/workspace` convention as the reference notebook (RunPod's persistent Network Volume). "
"`DATASET_DIR = None` triggers auto-detection in Section 6; set it explicitly here if your upload lives "
"somewhere not on the candidate list.",
))
cells.append(code(
"import os",
"",
"WORKSPACE = '/workspace' if os.path.isdir('/workspace') else os.path.expanduser('~')",
"REPO_URL = 'https://github.com/MoshiHead/personaplex-original-code-streaming-s-system-try.git'",
"REPO_DIR = os.path.join(WORKSPACE, 'personaplex')",
"HF_CACHE_DIR = os.path.join(WORKSPACE, '.cache', 'huggingface')",
"HF_REPO_ID = 'nvidia/personaplex-7b-v1'",
"",
"# Set this explicitly if your dataset isn't auto-detected in Section 6, e.g.:",
"#   DATASET_DIR = '/workspace/my_rag_dataset'",
"DATASET_DIR = None",
"",
"OUT_DIR = os.path.join(WORKSPACE, 'lora_checkpoints')       # training checkpoints (step_N/lora/...)",
"EXPORT_DIR = os.path.join(WORKSPACE, 'lora_export')          # final adapter + run config, copied at the end",
"TRAIN_LOG_PATH = os.path.join(WORKSPACE, 'lora_training.log')",
"",
"os.makedirs(WORKSPACE, exist_ok=True)",
"os.makedirs(HF_CACHE_DIR, exist_ok=True)",
"os.makedirs(OUT_DIR, exist_ok=True)",
"os.makedirs(EXPORT_DIR, exist_ok=True)",
"os.environ['HF_HOME'] = HF_CACHE_DIR",
"",
"print('WORKSPACE :', WORKSPACE)",
"print('REPO_DIR  :', REPO_DIR)",
"print('OUT_DIR   :', OUT_DIR)",
"print('HF_HOME   :', os.environ['HF_HOME'])",
))

cells.append(md("## 3. System package installation"))
cells.append(code(
"SUDO = '' if os.geteuid() == 0 else 'sudo '",
"",
"!{SUDO}apt-get update -qq",
"!{SUDO}apt-get install -y -qq --no-install-recommends git ca-certificates libopus-dev",
"print('System packages installed.')",
))

cells.append(md(
"## 4. Repository setup",
"",
"Skips cloning if `REPO_DIR` already has a package present (either layout -- see Section 5's note on "
"`moshi/` vs `moshi_local/` naming) -- e.g. because you uploaded your own working copy to the volume "
"beforehand. **That's the recommended path if you want today's training fixes**, since they are not yet "
"in the public `REPO_URL`.",
))
cells.append(code(
"import pathlib",
"",
"def _find_package_dir(repo_dir):",
"    \"\"\"The Python package directory (containing pyproject.toml) is named 'moshi' in the public repo",
"    layout but 'moshi_local' in this session's local working copy -- check both.\"\"\"",
"    for name in ('moshi', 'moshi_local'):",
"        candidate = pathlib.Path(repo_dir) / name",
"        if (candidate / 'pyproject.toml').exists():",
"            return candidate",
"    return None",
"",
"package_dir = _find_package_dir(REPO_DIR)",
"if package_dir is not None:",
"    print(f'Repository already present at {REPO_DIR} (package dir: {package_dir.name}), skipping clone.')",
"else:",
"    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)",
"    package_dir = _find_package_dir(REPO_DIR)",
"    print(f'Cloned into {REPO_DIR}.')",
"",
"assert package_dir is not None, (",
"    f'No pyproject.toml found under {REPO_DIR}/moshi or {REPO_DIR}/moshi_local after clone/upload. '",
"     'If you uploaded your own copy, confirm it landed at REPO_DIR.'",
")",
"REPO_SCRIPTS_DIR = pathlib.Path(REPO_DIR) / 'scripts'",
"print('package_dir      :', package_dir)",
"print('REPO_SCRIPTS_DIR :', REPO_SCRIPTS_DIR, '(exists:', REPO_SCRIPTS_DIR.exists(), ')')",
))

cells.append(md("## 5. Python dependency installation"))
cells.append(code(
"%pip install -q --upgrade pip setuptools wheel",
'%pip install -q "{package_dir}/."',
))

cells.append(code(
"# Blackwell (RTX 5090) requires CUDA-13.0-built PyTorch wheels -- same as the reference notebook's",
"# Section 5, 'Extra step for Blackwell based GPUs'. Harmless on non-Blackwell GPUs too.",
"%pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",
))

cells.append(code(
"# accelerate: multi-GPU launcher/DDP wrapper used by moshi.train.",
"# peft: LoRA. NOT in moshi/pyproject.toml's base dependencies -- moshi.train needs it directly.",
"%pip install -q accelerate peft",
))

cells.append(md(
"## 6. Verification gate — confirm this checkout has today's training fixes",
"",
"**Does not let training start** if the checkout on this pod is missing the session's fixes to "
"`interleaver.py`/`train.py`. Silently training against the old code would not error -- it would just "
"produce a model that never learns to use `<ref>`, since the old injection code requires a `filler` span "
"your current dataset format doesn't have. This is the single most important cell in this notebook.",
))
cells.append(code(
"import sys",
"",
"sys.path.insert(0, str(package_dir))",
"",
"problems = []",
"",
"try:",
"    from moshi.interleaver import InterleavedTokenizer",
"    if not hasattr(InterleavedTokenizer, '_inject_rag_blocks'):",
"        problems.append(",
"            \"moshi.interleaver.InterleavedTokenizer has no '_inject_rag_blocks' method -- this checkout \"",
"            \"has the OLD interleaver.py, which requires a 'filler' span your dataset doesn't have. \"",
"            \"RAG injection would silently never fire.\"",
"        )",
"except ImportError as exc:",
"    problems.append(f'could not import moshi.interleaver at all: {exc}')",
"",
"try:",
"    import inspect",
"    from moshi import train as moshi_train",
"    train_src = inspect.getsource(moshi_train)",
"    if 'Accelerator' not in train_src:",
"        problems.append(",
"            \"moshi.train does not reference Accelerator -- this checkout has the OLD single-GPU-only \"",
"            \"train.py. Multi-GPU launch below would either fail or silently run single-process.\"",
"        )",
"except ImportError as exc:",
"    problems.append(f'could not import moshi.train at all: {exc}')",
"",
"if not (REPO_SCRIPTS_DIR / 'validate_rag_dataset.py').exists():",
"    problems.append(",
"        f\"{REPO_SCRIPTS_DIR / 'validate_rag_dataset.py'} does not exist -- Section 9's dataset \"",
"        \"validation step will not be able to run.\"",
"    )",
"",
"if problems:",
"    msg = 'Checkout on this pod is missing required fixes:\\n  - ' + '\\n  - '.join(problems)",
"    msg += (",
"        '\\n\\nFix: push this session\\'s local edits to your fork/REPO_URL and re-clone, OR upload this '",
"        'local working copy directly to REPO_DIR instead of letting Section 4 clone the public URL.'",
"    )",
"    raise RuntimeError(msg)",
"",
"print('Verification gate passed: interleaver.py, train.py, and validate_rag_dataset.py all present and current.')",
))

cells.append(md("## 7. CUDA / GPU verification"))
cells.append(code(
"import torch",
"",
"print('Torch version      :', torch.__version__)",
"print('Torch CUDA version :', torch.version.cuda)",
"print('CUDA available     :', torch.cuda.is_available())",
"",
"if not torch.cuda.is_available():",
"    raise RuntimeError('No CUDA GPU detected by PyTorch -- see the nvidia-smi output in Section 1b.')",
"",
"visible = torch.cuda.device_count()",
"print(f'torch sees {visible} CUDA device(s) (nvidia-smi saw {NUM_GPUS}):')",
"for i in range(visible):",
"    name = torch.cuda.get_device_name(i)",
"    x = torch.randn(256, 256, device=f'cuda:{i}', dtype=torch.bfloat16)",
"    y = x @ x",
"    torch.cuda.synchronize(i)",
"    print(f'  cuda:{i} -> {name}  (bf16 matmul smoke test OK)')",
"",
"assert visible == NUM_GPUS, (",
"    f'torch sees {visible} GPU(s) but nvidia-smi saw {NUM_GPUS} -- mismatched CUDA_VISIBLE_DEVICES?'",
")",
))

cells.append(md(
"## 8. Hugging Face authentication",
"",
"**Manual step required before running this cell:** accept the license at "
"[`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1), then create a read-access "
"token at <https://huggingface.co/settings/tokens>.",
"",
"No token is ever hardcoded here -- set `HF_TOKEN` in the pod's environment beforehand, or this cell "
"prompts for one with hidden input.",
))
cells.append(code(
"from getpass import getpass",
"",
"from huggingface_hub import login",
"",
"hf_token = os.environ.get('HF_TOKEN')",
"if not hf_token:",
"    hf_token = getpass('Enter your Hugging Face access token (input hidden): ')",
"",
"os.environ['HF_TOKEN'] = hf_token",
"login(token=hf_token, add_to_git_credential=False)",
"print('Logged in to Hugging Face Hub.')",
))

cells.append(md(
"## 9. Dataset auto-load & validation",
"",
"Scans a few common upload locations for `train.jsonl`; set `DATASET_DIR` in Section 2 if yours lives "
"somewhere else. Then runs the real `scripts/validate_rag_dataset.py` (the five requirement-9 checks: "
"user span exists, `<lookup>` immediately after user, `<ref>` precedes the answer, no lead/filler spans, "
"answer follows `<ref>`) against the *actual* parser/injector -- **training does not start if this "
"fails.**",
))
cells.append(code(
"import json",
"",
"_CANDIDATE_DATASET_DIRS = [",
"    DATASET_DIR,",
"    os.path.join(WORKSPACE, 'dataset'),",
"    os.path.join(WORKSPACE, 'data'),",
"    os.path.join(WORKSPACE, 'rag_dataset'),",
"    os.path.join(WORKSPACE, 'train_dataset'),",
"    os.path.join(WORKSPACE, 'personaplex_dataset'),",
"    os.path.join(REPO_DIR, 'data', 'rag_from_20k'),",
"]",
"",
"resolved_dataset_dir = None",
"for candidate in _CANDIDATE_DATASET_DIRS:",
"    if candidate and os.path.exists(os.path.join(candidate, 'train.jsonl')):",
"        resolved_dataset_dir = candidate",
"        break",
"",
"if resolved_dataset_dir is None:",
"    raise FileNotFoundError(",
"        'Could not auto-detect a dataset. Checked: '",
"        + ', '.join(c for c in _CANDIDATE_DATASET_DIRS if c)",
"        + '. Upload your dataset to one of these paths, or set DATASET_DIR explicitly in Section 2 '",
"          'and re-run from there.'",
"    )",
"",
"DATASET_DIR = resolved_dataset_dir",
"print('Using dataset:', DATASET_DIR)",
"",
"manifest_entries = [json.loads(l) for l in open(os.path.join(DATASET_DIR, 'train.jsonl'), encoding='utf-8') if l.strip()]",
"durations = sorted(e['duration'] for e in manifest_entries)",
"print(f'{len(manifest_entries)} training examples')",
"print(f'duration stats: min={durations[0]:.2f}s p50={durations[len(durations)//2]:.2f}s max={durations[-1]:.2f}s')",
"",
"eval_jsonl = os.path.join(DATASET_DIR, 'eval.jsonl')",
"EVAL_DATA_ARG = eval_jsonl if os.path.exists(eval_jsonl) else ''",
"print('eval data:', EVAL_DATA_ARG or '(none found -- training will run without periodic eval)')",
))
cells.append(code(
"sys.path.insert(0, str(REPO_SCRIPTS_DIR))",
"from validate_rag_dataset import validate",
"",
"dataset_ok = validate(dataset_dir=DATASET_DIR, repo_root=REPO_DIR, tokenizer_path=None)",
"if not dataset_ok:",
"    raise RuntimeError(",
"        'Dataset validation FAILED (see failures printed above). Fix the dataset before training -- '",
"        'training on unvalidated data risks silently never injecting <lookup>/<ref> for the affected examples.'",
"    )",
"print('\\nDataset validation PASSED -- safe to train on.')",
))

cells.append(md(
"## 10. Training configuration",
"",
"The one section you might actually want to edit. Defaults are conservative (small LoRA-fine-tune-on-"
"pretrained-7B learning rate, gradient checkpointing on). `DURATION_SEC` is computed from your dataset's "
"real max duration plus a safety margin, not guessed -- picking too small a value silently truncates "
"longer examples (`interleaver.py` hard-cuts audio at `duration_sec * frame_rate` frames).",
"",
"`--max-steps` is deliberately **not** set here -- `moshi.train` computes it itself from `--epochs`, the "
"actual `--train-data` sample count, `--batch-size`, `--grad-accum`, and however many processes "
"`accelerate launch` actually starts (Section 11), so `EPOCHS` below means the same number of full data "
"passes no matter how many GPUs this pod has. This cell only *predicts* that same number ahead of time "
"(reusing the exact same counting function) so `--save-every`/`--eval-every` can be sized sensibly "
"relative to it, instead of being hardcoded to values that made sense for a fixed 2000-step run.",
))
cells.append(code(
"from moshi.dataset import count_training_samples",
"",
"DURATION_SEC = round(durations[-1] * 1.1, 1)   # max observed + 10% margin",
"EPOCHS = 10                                     # full passes over the dataset; same for any GPU count",
"BATCH_SIZE = 4                                  # PER-GPU; global = this * NUM_GPUS * GRAD_ACCUM",
"GRAD_ACCUM = 4",
"",
"total_samples = count_training_samples(DATASET_DIR, DURATION_SEC)",
"global_batch = BATCH_SIZE * NUM_GPUS * GRAD_ACCUM",
"steps_per_epoch = -(-total_samples // global_batch)  # ceil, matches train.py's own math exactly",
"predicted_max_steps = steps_per_epoch * EPOCHS",
"print(f'{total_samples} samples / (batch_size={BATCH_SIZE} * {NUM_GPUS} GPU(s) * grad_accum={GRAD_ACCUM} '",
"      f'= {global_batch} global batch) = {steps_per_epoch} steps/epoch * {EPOCHS} epochs '",
"      f'= {predicted_max_steps} steps (train.py will compute this same number itself)')",
"",
"TRAIN_CONFIG = {",
"    '--train-data': DATASET_DIR,",
"    '--eval-data': EVAL_DATA_ARG,",
"    '--duration-sec': str(DURATION_SEC),",
"    '--batch-size': str(BATCH_SIZE),",
"    '--max-ref-tokens': '500',",
"    '--hf-repo': HF_REPO_ID,",
"    '--lora-rank': '128',",
"    '--lora-scaling': '2.0',",
"    '--lr': '2e-6',",
"    '--weight-decay': '0.1',",
"    '--epochs': str(EPOCHS),        # train.py auto-computes --max-steps from this -- see Section 10 markdown",
"    '--grad-accum': str(GRAD_ACCUM),",
"    '--first-codebook-weight': '100.0',",
"    '--text-padding-weight': '0.5',",
"    '--ref-context-weight': '0.0',   # was 1.0 -- excludes the injected <ref> block from the text loss",
"    '--lookup-weight': '0.0',        # was 5.0 -- excludes the injected <lookup> block from the text loss",
"    '--body-weight': '2.0',",
"    '--no-rag-weight': '1.0',",
"    '--out-dir': OUT_DIR,",
"    '--save-every': str(max(50, predicted_max_steps // 20)),   # ~20 checkpoints over the whole run",
"    '--log-every': '10',",
"    '--eval-every': str(max(50, predicted_max_steps // 10)),   # ~10 evals over the whole run",
"}",
"",
"print(f'Global batch size = {TRAIN_CONFIG[\"--batch-size\"]} (per-GPU) '",
"      f'x {NUM_GPUS} (GPUs) x {TRAIN_CONFIG[\"--grad-accum\"]} (grad_accum) = '",
"      f'{int(TRAIN_CONFIG[\"--batch-size\"]) * NUM_GPUS * int(TRAIN_CONFIG[\"--grad-accum\"])}')",
"for k, v in TRAIN_CONFIG.items():",
"    print(f'  {k} = {v}')",
))

cells.append(md(
"## 11. Launch training",
"",
"`accelerate launch` with `--num_processes` set to the GPU count detected in Section 1b -- one process "
"per GPU, each training a full DDP replica of the LoRA-adapted model on its own data shard (see "
"`train.py`'s module docstring for the mechanism). With 1 GPU this degrades to a normal single-process "
"run; no separate code path needed. Logs stream to `TRAIN_LOG_PATH` and are tailed here so you can watch "
"progress without losing the ability to `Run All` unattended.",
))
cells.append(code(
"import time",
"",
"launch_cmd = [sys.executable, '-m', 'accelerate.commands.launch']",
"if NUM_GPUS > 1:",
"    launch_cmd += ['--multi_gpu', '--num_processes', str(NUM_GPUS)]",
"launch_cmd += ['--mixed_precision', 'bf16', '-m', 'moshi.train']",
"for k, v in TRAIN_CONFIG.items():",
"    if v == '':",
"        continue   # e.g. --eval-data with no eval set -- omit rather than pass an empty string flag",
"    launch_cmd += [k, v]",
"",
"print('Launch command:')",
"print(' ', ' '.join(launch_cmd))",
"",
"env = os.environ.copy()",
"env['PYTHONPATH'] = str(package_dir) + os.pathsep + env.get('PYTHONPATH', '')",
"",
"log_file = open(TRAIN_LOG_PATH, 'w', encoding='utf-8')",
"train_proc = subprocess.Popen(",
"    launch_cmd, cwd=str(package_dir), env=env,",
"    stdout=log_file, stderr=subprocess.STDOUT,",
")",
"print(f'Training started, PID={train_proc.pid}. Logs -> {TRAIN_LOG_PATH}')",
))
cells.append(code(
"def tail(path, n=40):",
"    try:",
"        with open(path, encoding='utf-8', errors='replace') as f:",
"            return ''.join(f.readlines()[-n:])",
"    except FileNotFoundError:",
"        return '(log file not created yet)'",
"",
"POLL_S = 30",
"last_size = -1",
"",
"while True:",
"    ret = train_proc.poll()",
"    size = os.path.getsize(TRAIN_LOG_PATH) if os.path.exists(TRAIN_LOG_PATH) else 0",
"    if size != last_size:",
"        print(f'--- log tail @ {time.strftime(\"%H:%M:%S\")} ---')",
"        print(tail(TRAIN_LOG_PATH))",
"        last_size = size",
"    if ret is not None:",
"        break",
"    time.sleep(POLL_S)",
"",
"log_file.close()",
"if train_proc.returncode != 0:",
"    print('\\n=== TRAINING FAILED (non-zero exit) -- full log tail below ===')",
"    print(tail(TRAIN_LOG_PATH, n=150))",
"    raise RuntimeError(",
"        f'accelerate launch exited with code {train_proc.returncode}. See log above and {TRAIN_LOG_PATH} in full.'",
"    )",
"print(f'\\nTraining process exited cleanly (code 0). Full log at {TRAIN_LOG_PATH}.')",
))

cells.append(md(
"## 12. Verify checkpoint output",
"",
"Confirms `train.py` actually wrote a final adapter (not just that the subprocess exited 0 -- e.g. a crash "
"during the very first save would still need catching here).",
))
cells.append(code(
"import glob",
"",
"step_dirs = sorted(",
"    glob.glob(os.path.join(OUT_DIR, 'step_*')),",
"    key=lambda p: int(os.path.basename(p).split('_')[1]),",
")",
"if not step_dirs:",
"    raise RuntimeError(f'No step_* checkpoint directories found under {OUT_DIR} -- training did not save anything.')",
"",
"final_ckpt = step_dirs[-1]",
"lora_dir = os.path.join(final_ckpt, 'lora')",
"adapter_config = os.path.join(lora_dir, 'adapter_config.json')",
"",
"print(f'{len(step_dirs)} checkpoint(s) found, most recent: {final_ckpt}')",
"assert os.path.exists(adapter_config), f'Expected {adapter_config} to exist (peft save_pretrained output).'",
"print('adapter_config.json present -- LoRA adapter saved correctly.')",
"",
"with open(adapter_config) as f:",
"    cfg = json.load(f)",
"print('LoRA r        :', cfg.get('r'))",
"print('LoRA alpha    :', cfg.get('lora_alpha'))",
"print('target_modules:', cfg.get('target_modules'))",
"",
"# Sanity check: catch a target_modules/architecture mismatch here, at",
"# export time, instead of only discovering it later as garbled/echoed",
"# speech in production. These are the real nn.Linear submodule names in",
"# this repo's transformer/gating code -- anything outside this set (e.g.",
"# the old 'in_proj'/'fc1'/'fc2'/'linear' defaults) matches NO module and",
"# silently receives no LoRA weights at all.",
"_VALID_TARGET_MODULES = {'out_proj', 'linear1', 'linear2', 'linear_in', 'linear_out', 'input_proj'}",
"_bad = [m for m in (cfg.get('target_modules') or []) if m not in _VALID_TARGET_MODULES]",
"if _bad:",
"    raise RuntimeError(",
"        f'adapter_config.json target_modules contains {_bad}, which do not match any real submodule '",
"        f'name in moshi_local/moshi/modules/transformer.py or gating.py -- these entries silently '",
"        f'match nothing and receive no LoRA weights. Fix apply_lora() in train.py and retrain.'",
"    )",
"if 'out_proj' not in (cfg.get('target_modules') or []):",
"    print(\"[warn] 'out_proj' not in target_modules -- attention output projection was not adapted.\")",
"print('target_modules verified against real model architecture -- OK.')",
))

cells.append(md(
"## 13. Export final LoRA adapter + run config",
"",
"Copies the final adapter to a clearly-named export directory (separate from the step-numbered "
"checkpoints in `OUT_DIR`, which stay around for `--resume`) and writes a `training_run_config.json` "
"recording exactly what produced this adapter -- dataset, hyperparameters, base model, GPU count, "
"timestamp -- so the run is reproducible later.",
))
cells.append(code(
"import shutil",
"import datetime",
"",
"export_lora_dir = os.path.join(EXPORT_DIR, 'lora_adapter')",
"if os.path.exists(export_lora_dir):",
"    shutil.rmtree(export_lora_dir)",
"shutil.copytree(lora_dir, export_lora_dir)",
"",
"run_config = {",
"    'exported_at_utc': datetime.datetime.utcnow().isoformat() + 'Z',",
"    'base_model_hf_repo': HF_REPO_ID,",
"    'dataset_dir': DATASET_DIR,",
"    'num_training_examples': len(manifest_entries),",
"    'num_gpus': NUM_GPUS,",
"    'final_checkpoint_dir': final_ckpt,",
"    'train_config': TRAIN_CONFIG,",
"}",
"with open(os.path.join(EXPORT_DIR, 'training_run_config.json'), 'w', encoding='utf-8') as f:",
"    json.dump(run_config, f, indent=2)",
"",
"print('Exported LoRA adapter ->', export_lora_dir)",
"print('Run config            ->', os.path.join(EXPORT_DIR, 'training_run_config.json'))",
"print('\\nContents:')",
"for name in sorted(os.listdir(export_lora_dir)):",
"    print(' ', name)",
))

cells.append(md(
"## 14. Troubleshooting",
"",
"| Symptom | Likely cause | Fix |",
"|---|---|---|",
"| Section 6 verification gate raises `RuntimeError` | Checkout on the pod is the old public repo, missing this session's fixes | Push local changes to your fork and re-clone, or upload this local working copy to `REPO_DIR` directly |",
"| Section 9 raises `FileNotFoundError` for the dataset | Upload didn't land at a checked path | Upload to one of the listed candidates, or set `DATASET_DIR` in Section 2 |",
"| Section 9's `validate()` reports failures | Dataset has malformed/old-format sidecars | Fix the flagged examples (often stale `lead`/`filler` spans from an old conversion run) and re-run Section 9 |",
"| CUDA OOM during training | `--batch-size`/`--duration-sec` too large for available VRAM | Lower `--batch-size` in Section 10 (global batch size already accounts for `NUM_GPUS`), or shorten `DURATION_SEC` if your dataset allows it |",
"| Training hangs right after launch with multiple GPUs | NCCL P2P/IB issue on some pod networking configs | Add `env['NCCL_P2P_DISABLE'] = '1'` before launching in Section 11 |",
"| `401`/`403` downloading model assets | License not accepted, or `HF_TOKEN` doesn't belong to the account that accepted it | Re-check Section 8 |",
"| Section 12 can't find `adapter_config.json` | Training crashed before the first `--save-every` step | Check the full log at `TRAIN_LOG_PATH`; reduce `--save-every` to checkpoint more frequently while debugging |",
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

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "PersonaPlex_LoRA_Training_RunPod.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_path)
