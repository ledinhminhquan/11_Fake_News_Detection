"""Generate the H100 AUTOPILOT Colab notebook as valid .ipynb JSON.

Run:  python notebooks/_build_notebook.py
Produces: notebooks/Fake_News_Detection_Colab_Training_H100_AUTOPILOT.ipynb

Building from a Python generator guarantees valid JSON. Mirrors the resume-safe,
GPU-auto-profiling, Colab-safe-install pattern proven in P02–P10.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = "Fake_News_Detection_Colab_Training_H100_AUTOPILOT.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(lines)}


cells = []

cells.append(md(
    "# 🕵️ Fake News & Misinformation Detection — Colab Training (H100 AUTOPILOT, resume-safe)\n",
    "\n",
    "Fine-tunes the **fake-news classifier** (HF Trainer, macro-F1) + trains the TF-IDF baseline. The\n",
    "stance/NLI model and the evidence embedder are pretrained (zero-shot).\n",
    "\n",
    "**How to use:** set the controls in cell 0, then **Runtime → Run all**. Resume-safe (re-run cell 9).\n",
    "Auto-adapts H100/A100/L4/T4.\n",
    "\n",
    "> ⚖️ This tool **flags content for human review and shows its evidence — it never auto-censors.**\n",
))

cells.append(code(
    "#@title 0) Controls — set these, then `Runtime → Run all`  { display-mode: \"form\" }\n",
    "GIT_REPO_URL = \"https://github.com/ledinhminhquan/fakenews\"  #@param {type:\"string\"}\n",
    "GIT_BRANCH   = \"main\"  #@param {type:\"string\"}\n",
    "USE_DRIVE    = True     #@param {type:\"boolean\"}\n",
    "DRIVE_SUBDIR = \"fakenews\"  #@param {type:\"string\"}\n",
    "\n",
    "# --- model / data ---\n",
    "CLF_BASE     = \"distilbert-base-uncased\"  #@param [\"distilbert-base-uncased\", \"microsoft/deberta-v3-base\", \"answerdotai/ModernBERT-base\", \"roberta-base\"]\n",
    "CLF_DATASET  = \"GonzaloA/fake_news\"  #@param [\"GonzaloA/fake_news\", \"ErfanMoosaviMonazzah/fake-news-detection-dataset-English\", \"chengxuphd/liar2\"]\n",
    "FAKE_LABEL_VALUE = 0   #@param {type:\"integer\"}\n",
    "MAX_TRAIN_SAMPLES = 24000  #@param {type:\"integer\"}\n",
    "EPOCHS       = 3        #@param {type:\"integer\"}\n",
    "RUN_AUTOPILOT = True   #@param {type:\"boolean\"}\n",
    "HF_TOKEN     = \"\"       #@param {type:\"string\"}\n",
    "print('Controls set. Classifier =', CLF_BASE, '| dataset =', CLF_DATASET, '(fake=', FAKE_LABEL_VALUE, ')')\n",
))

cells.append(code(
    "#@title 1) Check the GPU\n",
    "import subprocess\n",
    "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout or 'No GPU — Runtime→Change runtime type→GPU')\n",
))

cells.append(code(
    "#@title 2) Mount Drive + artifact paths & HF caches  (BEFORE importing torch)\n",
    "import os\n",
    "ART = '/content/artifacts'\n",
    "if USE_DRIVE:\n",
    "    try:\n",
    "        from google.colab import drive\n",
    "        drive.mount('/content/drive')\n",
    "        ART = f'/content/drive/MyDrive/{DRIVE_SUBDIR}/artifacts'\n",
    "    except Exception as e:\n",
    "        print('Drive mount skipped:', e)\n",
    "os.makedirs(ART, exist_ok=True)\n",
    "os.environ['FAKENEWS_ARTIFACTS_DIR'] = ART\n",
    "os.environ['HF_HOME'] = f'{ART}/hf_cache'\n",
    "os.makedirs(os.environ['HF_HOME'], exist_ok=True)\n",
    "if HF_TOKEN:\n",
    "    os.environ['HF_TOKEN'] = HF_TOKEN; os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN\n",
    "print('Artifacts ->', ART)\n",
))

cells.append(code(
    "#@title 3) Get the project source (git clone, or copy from Drive)\n",
    "import os\n",
    "os.chdir('/content')\n",
    "if os.path.isdir('/content/fakenews'):\n",
    "    os.chdir('/content/fakenews'); os.system('git pull')\n",
    "elif GIT_REPO_URL and 'ledinhminhquan' not in GIT_REPO_URL:\n",
    "    os.system(f'git clone -b {GIT_BRANCH} {GIT_REPO_URL} /content/fakenews'); os.chdir('/content/fakenews')\n",
    "else:\n",
    "    drive_src = f'/content/drive/MyDrive/{DRIVE_SUBDIR}/fakenews'\n",
    "    if os.path.isdir(drive_src):\n",
    "        os.system(f'cp -r {drive_src} /content/fakenews'); os.chdir('/content/fakenews')\n",
    "    else:\n",
    "        raise SystemExit('Set GIT_REPO_URL to your repo, or upload the project to Drive at ' + drive_src)\n",
    "print('cwd =', os.getcwd()); print(sorted(os.listdir('.'))[:20])\n",
))

cells.append(code(
    "#@title 4) Install dependencies (Colab-safe: NEVER reinstall torch)\n",
    "!pip -q install -r requirements_colab.txt\n",
    "!pip -q install -e . --no-deps\n",
    "print('\\u2713 deps installed')\n",
))

cells.append(code(
    "#@title 5) Verify environment + performance knobs (TF32)\n",
    "import torch\n",
    "print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    torch.backends.cuda.matmul.allow_tf32 = True\n",
    "    torch.backends.cudnn.allow_tf32 = True\n",
    "    print('GPU:', torch.cuda.get_device_name(0))\n",
    "import fakenews, transformers, datasets\n",
    "print('fakenews', fakenews.__version__, '| transformers', transformers.__version__)\n",
))

cells.append(code(
    "#@title 6) Auto GPU profile (classifier batch + precision)\n",
    "import torch\n",
    "name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'\n",
    "n = name.upper()\n",
    "if 'H100' in n:     BATCH, PREC = 32, 'bf16'\n",
    "elif 'A100' in n:   BATCH, PREC = 24, 'bf16'\n",
    "elif 'L4' in n:     BATCH, PREC = 16, 'bf16'\n",
    "elif 'T4' in n:     BATCH, PREC = 8,  'fp16'\n",
    "else:               BATCH, PREC = 8,  'fp16'\n",
    "if any(k in CLF_BASE.lower() for k in ('deberta-v3-large', 'large')): BATCH = max(4, BATCH // 2)\n",
    "BF16, FP16, TF32 = (PREC=='bf16'), (PREC=='fp16'), ('H100' in n or 'A100' in n)\n",
    "print(f'GPU={name} -> batch={BATCH} precision={PREC}')\n",
))

cells.append(code(
    "#@title 7) Write the Colab training config  (configs/train_colab.yaml)\n",
    "import yaml, os\n",
    "cfg = {\n",
    "  'project_title': 'Fake News & Misinformation Detection System', 'author': 'Le Dinh Minh Quan', 'student_id': '23127460',\n",
    "  'data': {'clf_dataset': CLF_DATASET, 'fake_label_value': int(FAKE_LABEL_VALUE), 'use_hf': True,\n",
    "           'max_train_samples': int(MAX_TRAIN_SAMPLES), 'max_eval_samples': 4000,\n",
    "           'liar_dataset': 'chengxuphd/liar2', 'seed': 42},\n",
    "  'classifier': {'base_model': CLF_BASE, 'max_length': 384, 'num_labels': 2,\n",
    "                 'num_train_epochs': int(EPOCHS), 'learning_rate': 2.0e-5,\n",
    "                 'per_device_train_batch_size': int(BATCH), 'use_class_weights': True,\n",
    "                 'bf16': bool(BF16), 'fp16': bool(FP16), 'tf32': bool(TF32), 'eval_steps': 200, 'save_steps': 200},\n",
    "  'stance': {'nli_model': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', 'enabled': True},\n",
    "  'retrieval': {'embedder': 'sentence-transformers/all-MiniLM-L6-v2', 'top_k': 5},\n",
    "}\n",
    "os.makedirs('configs', exist_ok=True)\n",
    "yaml.safe_dump(cfg, open('configs/train_colab.yaml','w'), sort_keys=False)\n",
    "print(open('configs/train_colab.yaml').read())\n",
))

cells.append(code(
    "#@title 8) Sanity-check the dataset (streaming probe)\n",
    "!PYTHONPATH=src python -m fakenews.cli --config configs/train_colab.yaml data\n",
))

cells.append(md(
    "## ⭐ ONE BUTTON — autopilot (resume-safe)\n",
    "Trains the baseline + the transformer classifier, evaluates vs baselines + fact-check, runs error\n",
    "analysis, and writes **report.pdf + slides.pptx + grading + a submission bundle**. Re-run to resume.\n",
))

cells.append(code(
    "#@title 9) ⭐ ONE BUTTON autopilot  (re-run to resume)\n",
    "import os\n",
    "if RUN_AUTOPILOT:\n",
    "    os.system('PYTHONPATH=src python -m fakenews.cli --config configs/train_colab.yaml autopilot '\n",
    "              f'--limit {int(MAX_TRAIN_SAMPLES)}')\n",
    "else:\n",
    "    print('RUN_AUTOPILOT is off — use the individual steps below.')\n",
))

cells.append(md("## Individual steps (optional) — idempotent + resume-safe\n"))

cells.append(code(
    "#@title 10a) Train the transformer classifier (resumes from the last checkpoint)\n",
    "!PYTHONPATH=src python -m fakenews.cli --config configs/train_colab.yaml train-classifier --limit $MAX_TRAIN_SAMPLES --base-model \"$CLF_BASE\"\n",
))

cells.append(code(
    "#@title 10b) Train the TF-IDF + LogReg baseline (sklearn, fast)\n",
    "!PYTHONPATH=src python -m fakenews.cli --config configs/train_colab.yaml train-baseline\n",
))

cells.append(code(
    "#@title 10c) Evaluate — classifier vs baselines + fact-check (macro-F1, ROC-AUC, ECE)\n",
    "!PYTHONPATH=src python -m fakenews.cli --config configs/train_colab.yaml evaluate\n",
))

cells.append(code(
    "#@title 11) Diagnostics: eval metrics + model metadata\n",
    "import json, glob, os\n",
    "rd = os.path.join(os.environ['FAKENEWS_ARTIFACTS_DIR'], 'runs', 'eval', 'latest.json')\n",
    "if os.path.exists(rd):\n",
    "    print(json.dumps(json.load(open(rd)).get('summary', {}), indent=2))\n",
    "for m in glob.glob(os.path.join(os.environ['FAKENEWS_ARTIFACTS_DIR'], 'models', 'classifier', '*', 'model_meta.json')):\n",
    "    print(m); print(json.dumps(json.load(open(m)), indent=2)[:500])\n",
))

cells.append(md("## ✅ Test the trained model\n"))

cells.append(code(
    "#@title 12) Classify + fact-check with the trained model\n",
    "!PYTHONPATH=src python -m fakenews.cli --config configs/train_colab.yaml classify --text \"Miracle pill melts thirty pounds in a single day, doctors furious.\"\n",
    "print('\\n--- FACT-CHECK ---')\n",
    "!PYTHONPATH=src python -m fakenews.cli --config configs/train_colab.yaml factcheck --claim \"Drinking bleach cures every virus overnight.\"\n",
))

cells.append(code(
    "#@title 13) Locate deliverables (report.pdf + slides.pptx + bundle)\n",
    "import glob, os\n",
    "base = os.environ['FAKENEWS_ARTIFACTS_DIR']\n",
    "for pat in ['submission/*/report.pdf', 'submission/*/slides.pptx', 'submission/*/submission_bundle.zip']:\n",
    "    for f in glob.glob(os.path.join(base, pat)):\n",
    "        print(round(os.path.getsize(f)/1024, 1), 'KB', f)\n",
))

cells.append(code(
    "#@title 14) (Optional) Serve the API + Gradio UI\n",
    "# !PYTHONPATH=src python -m fakenews.cli --config configs/infer.yaml serve --ui --port 7860\n",
    "print('Uncomment to serve. On Colab add a tunnel (e.g. cloudflared) to expose :7860.')\n",
))

cells.append(md(
    "## ✅ Final checklist\n",
    "- [ ] GPU profile picked a sensible batch/precision\n",
    "- [ ] `train-classifier` wrote `models/classifier/<version>/`; `train-baseline` wrote `tfidf_logreg.joblib`\n",
    "- [ ] `evaluate` shows the classifier beating majority + TF-IDF on macro-F1 (and report ECE for calibration)\n",
    "- [ ] **Cross-domain** number reported (train PolitiFact → test GossipCop) — the source-leakage signal\n",
    "- [ ] `report.pdf` + `slides.pptx` + `submission_bundle.zip` exist under `artifacts/submission/`\n",
    "- [ ] Remember: the tool **flags for review, never auto-censors**; abstain is a valid outcome\n",
))


def main():
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(__file__).resolve().parent / NB
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    json.loads(out.read_text(encoding="utf-8"))   # validate
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
