# Modern AI with Unsloth.ai
**Author:** Siddharth Rao Kartik  
**Institution:** San José State University  
**Course:** Modern AI – Assignment Submission

## Overview
This repository contains five Google Colab notebooks showcasing practical workflows for **fine-tuning and reinforcement learning of open-weight LLMs** using **Unsloth.ai**.  
Each notebook focuses on one stage of model adaptation: **Full Fine-Tuning**, **LoRA**, **RLHF**, **GRPO (reasoning)**, and **Continued Pretraining**.  
All experiments run on **Google Colab Pro (T4/A100)**.

## Notebooks
- [Colab 1 – Full Fine-Tuning](./Colab1-Full_Finetuning.ipynb)
- [Colab 2 – LoRA Parameter-Efficient Fine-Tuning](./colab2_siddu.ipynb)
- [Colab 3 – Reinforcement Learning (RLHF)](./colab3_Reinforcement_learning.ipynb)
- [Colab 4 – GRPO (Reasoning RL)](./colab4-reinformcement%20learning%20with%20grpo.ipynb)
- [Colab 5 – Continued Pretraining](./colab5-Continued%20pretraining.ipynb)

> Open directly in Colab:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siddharthraokartik/Modern-AI-with-Unsloth/blob/main/Colab1-Full_Finetuning.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siddharthraokartik/Modern-AI-with-Unsloth/blob/main/colab2_siddu.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siddharthraokartik/Modern-AI-with-Unsloth/blob/main/colab3_Reinforcement_learning.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siddharthraokartik/Modern-AI-with-Unsloth/blob/main/colab4-reinformcement%20learning%20with%20grpo.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siddharthraokartik/Modern-AI-with-Unsloth/blob/main/colab5-Continued%20pretraining.ipynb)

## Assignment Summary
| Notebook | Task | Objective | Model | Dataset |
|:--|:--|:--|:--|:--|
| Colab 1 | Full Fine-Tuning | Train most parameters on instruction data | Smollm2 (135M) | FineTome-100k |
| Colab 2 | LoRA (PEFT) | Efficient adapter-based fine-tuning | Smollm2 (135M) | FineTome-100k |
| Colab 3 | RLHF | Reward-aligned responses (chosen/rejected) | Smollm2 (135M) | FineTome subset |
| Colab 4 | GRPO | Improve reasoning and structure | Smollm2 (135M) | Reasoning dataset |
| Colab 5 | Continued Pretraining | Extend knowledge with new text | Smollm2 (135M) | WikiText / domain corpus |

## Setup
```bash
pip install -U unsloth transformers datasets trl peft bitsandbytes accelerate torch
