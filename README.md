# Overcoming Catastrophic Forgetting via Generative Replay
### A Generative Model-based Approach to Continual Learning preserving Privacy

**Konstantina Marina Bletsa** · **Evangelos Apostolou** 

---

## Overview

This repository contains all the code developed for our undergraduate thesis on **Continual Learning** and the problem of **Catastrophic Forgetting**. The core idea is to replace traditional memory buffers (which store real data and raise privacy concerns) with **Generative Replay** — training generative models to synthesize past data distributions instead of storing them.

We implement and compare multiple architectures across visual and tabular datasets, with a particular focus on privacy-preserving continual learning for sensitive medical data.

---

## Repository Structure

```
Notebooks/
│
├── 📂 Baselines & Generative Replay — Visual Data
│   ├── VAE-MNIST.ipynb                          # VAE-based DGR on MNIST
│   ├── DC_GAN_MNIST_plots.ipynb                 # DCGAN on MNIST with plots
│   ├── GAN-CNN-CIFAR.ipynb                      # CNN WGAN-GP on CIFAR-10
│   ├── AC-WGAN-GP fASHION MNIST.ipynb           # AC-WGAN-GP on Fashion-MNIST
│   └── Conditional Deep VAE-CIFAR.ipynb         # Conditional VAE on CIFAR-10
│
├── 📂 Progressive Neural Networks (PNN)
│   ├── PNN MNIST Fashion.ipynb                  # PNN on MNIST → Fashion-MNIST
│   ├── PNN SVHN.ipynb                           # PNN on SVHN
│   ├── PNN CIFAR.ipynb                          # PNN on CIFAR-10
│   ├── PNN ChestXRay MNIST.ipynb                # PNN on ChestX-ray14 + MNIST
│   └── PNN 2 Datasets.ipynb                     # PNN comparative evaluation
│
├── 📂 Hybrid Generative Replay
│   ├── Hybrid_Mark_1.ipynb                      # Hybrid DGR v1: Anchor Buffer + KD
│   └── Hybrid_Mark_2.ipynb                      # Hybrid DGR v2: + Confidence Thresholding
│
└── 📂 Tabular VAE — Privacy-Preserving Medical Data
    ├── TabularDataMark1(Breast_Cancer_Dataset).ipynb
    ├── TabularDataMark2(Diabetes_Dataset).ipynb
    ├── TabularDataMark3(Forest_Covertype_Dataset_Domain-Incremental).ipynb
    └── TabularDataMark4(Forest_Covertype_Dataset_Class-Incremental).ipynb
```


## Methods

| Method | Generator | Dataset(s) | Scenario |
|---|---|---|---|
| Naive Sequential | — | All | Lower bound |
| Memory Buffer | — | All | Upper bound |
| VAE DGR | VAE | MNIST | Class-Incremental |
| DCGAN DGR | DCGAN | MNIST | Class-Incremental |
| AC-WGAN-GP DGR | AC-WGAN-GP | Fashion-MNIST, CIFAR-10 | Class-Incremental |
| Progressive Neural Networks | — | MNIST, Fashion, SVHN, CIFAR, ChestX-ray | Class-Incremental |
| **Hybrid DGR** *(proposed)* | AC-WGAN-GP | Fashion-MNIST, CIFAR-10 | Class-Incremental |
| Tabular VAE | VAE | Breast Cancer, Diabetes, Covtype | Domain / Class-Incremental |

---


## Evaluation Metrics

- **Average Accuracy** — mean classification accuracy across all tasks after training
- **Forgetting Measure** — difference between peak accuracy and final accuracy per task
- **Memory Footprint** — MB required by each method (generator weights vs. stored images)

---

## Datasets

**Visual:** MNIST · Fashion-MNIST · SVHN · CIFAR-10 · ChestX-ray14  
**Tabular / Medical:** Breast Cancer Wisconsin · Pima Indians Diabetes · Forest Covertype 

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas
```

All notebooks were developed in **PyTorch** and are self-contained with inline training loops, evaluation, and plots.
