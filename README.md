# CPD-GNN

**Cross-Modal Prompt Disentangled Graph Neural Networks for Incomplete Conversational Emotion Recognition**

This repository accompanies the paper **"Cross-Modal Prompt Disentangled Graph Neural Networks for Incomplete Conversational Emotion Recognition"** and is intended to host the code, configurations, and experimental materials for incomplete multimodal conversational emotion recognition.

## Overview

Multimodal emotion recognition in conversation relies on text, audio, and visual signals. In real-world settings, however, one or more modalities may be unavailable because of noise, occlusion, recognition errors, or device failure. This repository focuses on **incomplete multimodal conversational emotion recognition**, where the model must remain robust under missing-modality conditions.

To address this problem, the paper proposes **CPD-GNN**, a unified framework that combines:

- shared-private feature disentanglement
- prototype-driven cross-modal prompt interaction
- dynamic modality balancing
- temporal and speaker interaction modeling
- hypergraph-based higher-order relational reasoning

## Method Highlights

### 1. Shared-private feature disentanglement
Incomplete multimodal inputs are projected into a unified semantic space and decomposed into:
- **shared features** for cross-modal semantic coordination
- **private features** for modality-specific information retention

### 2. Prototype-driven cross-modal prompting
Each modality learns compact prototypes from dialogue-level shared semantics. These prototypes guide cross-modal prompt generation so that available modalities can compensate for missing ones more effectively.

### 3. Dynamic modality balancing
A multimodal router assigns adaptive weights to different modalities according to contextual reliability, reducing modality imbalance during fusion.

### 4. Dual-graph plus hypergraph modeling
The model builds:
- a **temporal interaction graph**
- a **speaker interaction graph**

It then performs graph propagation and hypergraph convolution to capture both local dependencies and higher-order affective relations among utterances.

### 5. Joint optimization
The framework is trained with a joint objective that combines:
- task loss
- reconstruction loss
- disentanglement loss

## Datasets

Experiments are conducted on three benchmark datasets:

- **IEMOCAP**
  - four-class setting
  - six-class setting
- **CMU-MOSI**
- **CMU-MOSEI**

## Main Findings

The paper reports that CPD-GNN achieves competitive or superior performance under different missing rates on all three benchmark datasets.

Representative results include:

- On **IEMOCAPSix**, CPD-GNN shows clear gains under severe modality missingness.
- On **CMU-MOSEI**, the method outperforms strong baselines at multiple missing rates.
- Ablation results show that prompt generation, modality balancing, temporal interaction modeling, and speaker interaction modeling all contribute to final performance.

## Suggested Repository Structure

This is a suggested organization for the repository:

```text
CPD-GNN/
├── README.md
├── configs/
├── data/
├── datasets/
├── models/
├── modules/
├── scripts/
├── utils/
├── checkpoints/
└── results/
```

## Planned Contents

The repository can be organized to include:

- data preprocessing scripts
- model implementation
- training and evaluation scripts
- configuration files
- ablation settings
- visualization scripts for confusion matrices and parameter analysis

## Citation

If you use this work, please cite the corresponding paper.

```bibtex
@article{qiao2026cpdgnn,
  title={Cross-Modal Prompt Disentangled Graph Neural Networks for Incomplete Conversational Emotion Recognition},
  author={Qiao, Shi and Yang, Rui and Hu, Bin and Peng, Hong and Dang, Jisheng},
  journal={Preprint},
  year={2026}
}
```

## Status

This repository is currently being organized. Code, configs, and detailed usage instructions can be added as they are finalized.

## Contact

For questions about the manuscript or repository content, please contact the authors listed in the paper.
