# ASAP: Actively and Sequentially AdaPt foundation models for medical image segmentation

## 👩🏻‍💻 Overview

Welcome! This is the official repo of the ISBI 2025 paper "Adapting Foundation Models for Few-Shot Medical Image Segmentation: Actively and Sequentially". 

- Jingyun Yang, Guoqing Zhang, Jingge Wang, Yang Li

We propose an active and sequential domain adaptation (ASAP) framework to adapt foundation models for few-shot medical image segmentation.

### Goal
- Leverage a wealth of public medical resources: foundation models & available auxiliary datasets
- Objective: efficiently utilize these resources to tailor a model for the desired few-shot target task.
<img width="800" alt="image" src="https://github.com/user-attachments/assets/7e87b57d-7c49-4445-b12a-06337d5e9567" />



## 🪜 Framework

- Sequential domain adaptation strategy: transfer the knowledge from the source domain to the target domain step by step

- Dynamic dataset selection algorithm: prioritize training on auxiliary datasets with similar solution spaces to the target task in a single-round computational complexity 

<img width="800" alt="image" src="https://github.com/user-attachments/assets/076a7588-c529-423f-8058-0dd5bfcc512a" />


## 🧀 Results
<img width="800" alt="image" src="https://github.com/user-attachments/assets/832659a0-36f2-41ed-afb1-fc665b60e08f" />


<img width="800" alt="image" src="https://github.com/user-attachments/assets/f8f480ef-2fb4-4257-b84e-209a80a64c0a" />


<img width="800" alt="image" src="https://github.com/user-attachments/assets/cf8681de-f86d-41b8-8d02-4a346066a636" />


## ⭐ Citation

Please cite our paper if you find our work useful. Thanks!

```bibtex
@article{yang2025adapting,
  title={Adapting Foundation Models for Few-Shot Medical Image Segmentation: Actively and Sequentially},
  author={Yang, Jingyun and Zhang, Guoqing and Wang, Jingge and Li, Yang},
  journal={arXiv preprint arXiv:2502.01000},
  year={2025}
}


