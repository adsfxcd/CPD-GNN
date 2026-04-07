# Cross-Modal Prompt Disentangled Graph Neural Networks for Incomplete Conversational Emotion Recognition

An official PyTorch implementation of **CPD-GNN** (Cross-Modal Prompt Disentangled Graph Neural Network) for incomplete multimodal conversational emotion recognition.

## Overview

Multimodal emotion recognition relies on the joint modeling of text, speech, and visual signals. In real-world scenarios, however, some modalities are often unavailable due to noise, occlusion, recognition errors, or device malfunction. This not only weakens the use of complementary cross-modal information, but also intensifies imbalance in modality learning.

To address this issue, we propose **CPD-GNN**. The overall architecture is illustrated below:

![Model Architecture](figures/model.png)

CPD-GNN implements the following core capabilities:

+ **Unified Semantic Projection & Feature Disentanglement** — Projects incomplete multimodal features into a unified semantic space and disentangles them into shared representations and modality-specific representations.
+ **Cross-modal Information Compensation** — Jointly performs cross-modal information compensation to recover missing modality information.
+ **Dynamic Modality Balancing** — Adaptively balances modality contributions to mitigate learning imbalance under missing-modality conditions.
+ **Contextual & Higher-order Dependency Modeling** — Provides a robust foundation for modeling both local contextual interactions and higher-order conversational dependencies.

## Environment

+ Python 3.8.18
+ CUDA 11.6
+ torch 1.12.0
+ torch-geometric 2.4.0

(For details, see requirements.txt)

## Datasets

The following datasets are used in this research:

[IEMOCAP](https://sail.usc.edu/iemocap/index.html), [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/), [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

We also provide the [dataset features]() used in the code.

## Code

The complete code will be released after the paper is accepted.



