# NeuroScope AI

**AI-assisted clinical decision support for brain tumor MRI.**
Upload a brain MRI and NeuroScope segments the tumor, measures its regions,
finds matching scientific evidence, and summarizes everything in a structured
report for the treating physician.

> ⚠️ Research prototype built during a hackathon. Not a medical device. Results
> must not be used for diagnosis or treatment decisions.

## How it works

```
MRI scan ──► MONAI segmentation ──► Tumor metrics ──► Amass evidence ──► LLM report
(FLAIR, T1,   (SegResNet, GPU)      (volumes in mL)   (papers, trials)   (for the physician)
 T1ce, T2)
```

1. **See the tumor:** a pretrained MONAI SegResNet segments three tumor regions:
   peritumoral edema, non-enhancing tumor/necrosis and contrast-enhancing tumor.
2. **Measure it:** volumes (whole tumor, tumor core, enhancing tumor, edema),
   their ratios and the number of separate lesions.
3. **Find the evidence:** the findings are translated into medical search terms
   and matched against peer-reviewed literature and recruiting clinical trials
   via the [Amass](https://amass.tech) API.
4. **Explain it:** an LLM reads the findings and the evidence and writes a
   structured report: key findings, differential diagnosis, suggested next
   steps and limitations, each statement citing the evidence it is based on.

## Features

- Patient records with health history, shown as context next to the MRI analysis
- Upload of the four MRI sequences as separate files or as one 4D NIfTI scan
- Segmentation overlay with a color legend and per-region volumes
- Results in three tabs: **Overview**, full **Report**, and **Evidence** with links
- Built-in safeguards: the LLM may only cite retrieved sources (invented
  citations are removed automatically), gets no patient identifiers, and
  NeuroScope falls back to a rule-based report if the LLM is unavailable

## Tech stack

| Part | Technology |
|---|---|
| Segmentation | [MONAI](https://monai.io) `brats_mri_segmentation` bundle (SegResNet), PyTorch |
| GPU inference | [Modal](https://modal.com) |
| Backend | Python, FastAPI |
| Evidence | Amass API (BiomedCore literature, TrialCore clinical trials) |
| Report | `openai/gpt-oss-120b` via [Groq](https://groq.com) |
| Frontend | HTML, CSS, JavaScript (no framework) |

## Quick start

Requires Python 3.14, a [Modal](https://modal.com) account for GPU inference,
and optionally API keys for Amass and Groq.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Pretrained model weights
.venv/bin/python -m monai.bundle download --name brats_mri_segmentation --bundle_dir bundles

# Dataset (~7 GB) and one extracted example patient
.venv/bin/python preprocessing.py
.venv/bin/python ml/prepare_real_patient.py

# Deploy the segmentation to Modal
.venv/bin/modal setup
.venv/bin/modal deploy ml/modal_inference.py

# Start the app on http://localhost:8000
.venv/bin/uvicorn backend.app:app --port 8000
```

Optional keys go in a `.env` file in the project root. Without them the app still
works, but without literature search (Amass) and with a rule-based instead of
an LLM-written report (Groq):

```
AMASS_API_KEY=...
GROQ_API_KEY=...
```

## Data and model

- **Dataset:** Medical Segmentation Decathlon, Task01 Brain Tumour, derived from
  the BraTS challenge. Antonelli et al., *The Medical Segmentation Decathlon*,
  Nature Communications 13, 4128 (2022).
- **BraTS:** Menze et al., IEEE TMI 34(10), 2015; Bakas et al., Scientific Data 4, 2017.
- **Model:** MONAI Model Zoo `brats_mri_segmentation`, based on Myronenko,
  *3D MRI brain tumor segmentation using autoencoder regularization*, 2018.
- The patients shown in the app are the anonymized scans in
  `ml/data/real_patients/<ID>/` (created by `ml/prepare_real_patient.py`).
  An optional `patient.json` in a patient folder adds a clinical history
  (fields documented in `frontend/js/patients.js`).
