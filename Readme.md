An active and sequential domain adaptation (ASAP) framework for dynamic auxiliary dataset selection in few-shot domain adaptation.



Fine-tuning pre-trained models on specific low-resource medical segmentation tasks has become a standard practice.
To ensure reliable and robust model adaptation when the target task has a large domain gap and few annotated samples we utilize auxiliary datasets.

<img width="535" alt="截屏2024-10-08 23 29 32" src="https://github.com/user-attachments/assets/d596ad07-c5d7-4236-ad24-13ff5f95738b">

We formulate FSDA as a multi-armed bandit problem and derive an efficient reward function to prioritize training on auxiliary datasets that align closely with the target task, through a single-round fine-tuning.

<img width="818" alt="截屏2024-10-08 23 26 38" src="https://github.com/user-attachments/assets/d5538a15-9384-434a-ad33-a0938bb5415e">
