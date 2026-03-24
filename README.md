# Wesep

## Overview

> Target speaker extraction (TSE) focuses on isolating the speech of a specific target speaker from overlapped multi-talker speech, which is a typical setup in the cocktail party problem.
WeSep is featured with flexible target speaker modeling, scalable data management, effective on-the-fly data simulation, structured recipes and deployment support.

This version of **Wesep** is a **lightweight and competition-oriented release**. It is designed to:

- Provide a **reproducible training template**
- Support **official baseline system usage**
- Serve as a **reference implementation** for participants

<img src="resources/tse.png" width="600px">

### Install for development & deployment
* Clone this repo
``` sh
https://github.com/wenet-e2e/wesep.git
```

* Wesep is under active development and aims to support multi-cue inputs (speaker, visual, spatial, and semantic) as well as multiple modeling paradigms (discriminative, generative, and autoregressive).
``` sh
conda create -n wesep python=3.10
conda activate wesep

# Recommended (modern PyTorch)
pip install torch>=2.6 torchaudio>=2.6
# Alternative (legacy GPUs, e.g. V100)
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -r requirements.txt
pre-commit install  # for clean and tidy code
```

## Supported Features

### Model
- **BSRNN-based separator**
  - Causal
  - Non-causal

### Speaker Feature Representation Support

This version supports multiple types of **audio-based target speaker cues**:

- Speaker Embedding (via **WeSpeaker**)
- USEF Feature
- TF-Map Feature
- Contextual Embedding

## Pretrained Models

Will release soon.

## Data Pipe Design

Following Wenet and Wespeaker, WeSep organizes the data processing modules as a pipeline of a set of different processors. The following figure shows such a pipeline with essential processors.

<img src="resources/datapipe.png" width="800px">

## Discussion

For Chinese users, you can scan the QR code on the left to join our group directly. If it has expired, please scan the personal Wechat QR code on the right.

|<img src='resources/Wechat_group.jpg' style=" width: 200px; height: 300px;">|<img src='resources/Wechat.jpg' style=" width: 200px; height: 300px;">|
| ---- | ---- |



## Citations
If you find wespeaker useful, please cite it as

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
