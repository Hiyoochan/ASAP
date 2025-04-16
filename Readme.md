# ASAP: Actively and Sequentially AdaPt foundation models for medical image segmentation

## 👩🏻‍💻 Overview

Welcome! This is the official repo of the ISBI 2025 paper "Adapting Foundation Models for Few-Shot Medical Image Segmentation: Actively and Sequentially". 

- Jingyun Yang, Guoqing Zhang, Jingge Wang, Yang Li

We propose an active and sequential domain adaptation (ASAP) framework to adapt foundation models for few-shot medical image segmentation.

### Goal
-Leverage a wealth of public medical resources: foundation models & available auxiliary datasets
-Objective: efficiently utilize these resources to tailor a model for the desired few-shot target task.
![image](https://github.com/user-attachments/assets/576b9c65-a56a-4876-a79f-6a23d3536235)


## 🪜 Framework

We formulate FSDA as a multi-armed bandit problem and derive an efficient reward function to prioritize training on auxiliary datasets that align closely with the target task, through a single-round fine-tuning.

<img width="818" alt="截屏2024-10-08 23 26 38" src="https://github.com/user-attachments/assets/d5538a15-9384-434a-ad33-a0938bb5415e">

## 🧀 Results
<img width="416" alt="image" src="https://github.com/user-attachments/assets/832659a0-36f2-41ed-afb1-fc665b60e08f" />


<img width="414" alt="image" src="https://github.com/user-attachments/assets/f8f480ef-2fb4-4257-b84e-209a80a64c0a" />


<img width="415" alt="image" src="https://github.com/user-attachments/assets/cf8681de-f86d-41b8-8d02-4a346066a636" />


## ⭐ Citation

Please cite our paper if you find our work useful. Thanks!

```bibtex
@article{yang2025adapting,
  title={Adapting Foundation Models for Few-Shot Medical Image Segmentation: Actively and Sequentially},
  author={Yang, Jingyun and Zhang, Guoqing and Wang, Jingge and Li, Yang},
  journal={arXiv preprint arXiv:2502.01000},
  year={2025}
}


