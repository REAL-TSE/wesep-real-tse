# WeSep (REAL-TSE Challenge Baseline Toolkit)

## Overview

> **WeSep** is a lightweight baseline toolkit for the **Real-TSE Challenge**, designed for **target speaker extraction (TSE)**.

Target speaker extraction aims to isolate a specific speaker’s voice from overlapped multi-speaker audio — a classic *cocktail party problem*.

⚠️ **Important**  
This repository is **NOT required** to participate in the challenge.
- You are free to use **any model, framework, or training pipeline**
- WeSep is provided as:
  - a **baseline system**
  - a **reference implementation**
  - a **quick starting point**

## What is this repo for?
WeSep is designed to help you:
- ✅ Run the **official baseline model**
- ✅ Quickly test TSE on your own data
- ✅ Train your own model (optional)
- ✅ Explore different **target speaker representations**

## Relation to the Challenge
👉 This repository **does NOT include**:
- Official dataset  
- Evaluation toolkit  
- Submission pipeline  

Please refer to the **challenge website** and **main challenge repository** for:
- Data access  
- Evaluation protocol  
- Submission instructions  
---

## Quick Start (3 steps)

### 1. Clone & install

```bash
git clone https://github.com/REAL-TSE/wesep-real-tse.git
cd wesep-real-tse

conda create -n wesep python=3.10
conda activate wesep

# Recommended (aligned with evaluation toolkit)
pip install torch==2.7.1 torchaudio==2.7.1
# Note: This command may install the **CPU version** of PyTorch by default.
# If you are using a GPU, please install the CUDA-enabled version manually.
# Alternative: (if your GPU doesn't support PyTorch 2.x)
# conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c 

pip install -r requirements.txt

# speaker embedding support
pip install git+https://github.com/wenet-e2e/wespeaker.git@8f53b6485d9f88a207bd17e7f8dba899495ec794
```

### 2. Download pretrained model

The checkpoints are available at [Google Drive](https://drive.google.com/uc?export=download&id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC) . The directory structure for the pretrained models in the REAL-T project is suggested to be:

```
REAL-T/
├── pretrained/
│ ├── spk_emb_100/
│ │ ├── avg_model.pt
│ │ └── config.yaml
│ ├── spk_emb_causal_100/
│ ├── tfmap_context_100/
│ └── tfmap_context_causal_100/
```

### 3. Run inference
To use a checkpoint for extracting a target speech from "mixture.wav" with "enroll.wav":

``` sh
python evaluate.py \
  --pretrain path/to/model_folder \
  --mixture path/to/mix.wav \
  --enroll path/to/enroll.wav \
  --output path/to/output.wav \
```
---

## Supported Models
- **BSRNN-based separator**
  - Causal
  - Non-causal
- **Audio-based target speaker features**:
  - Speaker Embedding (via **WeSpeaker**)
  - USEF Feature
  - TF-Map Feature
  - Contextual Embedding

## Training (Optional)

WeSep also provides a **training template**, but:

- Training is **NOT required** for the challenge
- You can use any training pipeline you prefer

Training recipes (WIP) are provided [here](https://github.com/REAL-TSE/wesep-real-tse/tree/main/examples/audio) .

Currently includes:

- Libri2Mix-based training pipeline
- VoxCeleb-based online mixing pipeline

If you are new, we recommend starting with **Libri2Mix**.

Note:
These examples are under active development (WIP) and may be updated.

## Data Pipeline (Advanced)

WeSep adopts a modular data processing pipeline design (inspired by Wenet and WeSpeaker), enabling flexible data simulation and feature construction.

<img src="resources/datapipe.png" width="800px">


## Citations
If you find WeSep useful, please cite it as

```bibtex
@inproceedings{wang24fa_interspeech,
  title     = {WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction},
  author    = {Shuai Wang and Ke Zhang and Shaoxiong Lin and Junjie Li and Xuefei Wang and Meng Ge and Jianwei Yu and Yanmin Qian and Haizhou Li},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4273--4277},
  doi       = {10.21437/Interspeech.2024-1840},
}
```
