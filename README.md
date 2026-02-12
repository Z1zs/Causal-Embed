
# CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding

[![arXiv](https://img.shields.io/badge/arXiv-2601.21262-b31b1b.svg?style=for-the-badge)](https://www.arxiv.org/abs/2601.21262)
[![GitHub](https://img.shields.io/badge/CausalEmbed-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Z1zs/Causal-Embed)
[![Hugging Face](https://img.shields.io/badge/Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Z1zs/causalembed)


<div>
<div align="center">
    <a href='https://z1zs.github.io/' target='_blank'>Jiahao Huo<sup>1,2,3</sup></a>&emsp;
    <a href='https://hardenyu21.github.io/' target='_blank'>Yu Huang<sup>1,2</sup></a>&emsp;
    <a href='https://openreview.net/profile?id=~Yibo_Yan1' target='_blank'>Yibo Yan<sup>1,2,4</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=9RetdWMAAAAJ&hl=en' target='_blank'>Ye Pan<sup>1</sup></a>&emsp;<br>
    <a href='https://openreview.net/profile?id=~Yi_Cao10' target='_blank'>Yi Cao<sup>2</sup></a>&emsp;
    <a href='https://openreview.net/profile?id=~Mingdong_Ou1' target='_blank'>Mingdong Ou<sup>♣️,2</sup></a>&emsp;
    <a href='https://cs.uic.edu/profiles/philip-yu/' target='_blank'>Philip S. Yu<sup>3</sup></a>&emsp;
    <a href='https://xuminghu.github.io/' target='_blank'>Xuming Hu<sup>✉,1,4</sup></a>&emsp;
</div>
<div>
<div align="center">
    <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)&emsp;
    <sup>2</sup>Alibaba Cloud Computing&emsp;<br>
    <sup>3</sup>University of Illinois Chicago&emsp;
    <sup>4</sup>The Hong Kong University of Science and Technology&emsp;<br>
    <sup>♣️</sup> Project Leader&emsp; <sup>✉</sup> Corresponding Author
</div>
</div>

---

Official implementation of "[CAUSALEMBED: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding](https://arxiv.org/pdf/2601.21262)".  
This repository is built upon the official implementation of [ColPali](https://github.com/illuin-tech/colpali). Thanks a lot for their efforts!

## Updates

- **29 Jan, 2026** : Paper published in Arxiv.
- **10 Feb, 2025** : Code and models published.

---

This repository contains the **official implementation** of the following paper:

> **CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding** https://arxiv.org/abs/2601.21262
>
> **Abstract:** _Although Multimodal Large Language Models (MLLMs) have shown remarkable potential in Visual Document Retrieval (VDR) through generating high-quality multi-vector embeddings, the substantial storage overhead caused by representing a page with thousands of visual tokens limits their practicality in real-world applications. To address this challenge, we propose an auto-regressive generation approach, CausalEmbed, for constructing multi-vector embeddings. By incorporating iterative margin loss during contrastive training, CausalEmbed encourages the embedding models to learn compact and well-structured representations. Our method enables efficient VDR tasks using only dozens of visual tokens, achieving a 30-155x reduction in token count while maintaining highly competitive performance across various backbones and benchmarks. Theoretical analysis and empirical results demonstrate the unique advantages of auto-regressive embedding generation in terms of training efficiency and scalability at test time. As a result, CausalEmbed introduces a flexible test-time scaling strategy for multi-vector VDR representations and sheds light on the generative paradigm within multimodal document retrieval._

## List of CausalEmbed models

| Model                                                               | Score on [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) 🏆 | License    | Comments                                                                                                                                                       | Currently supported |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| [Z1zs/CausalQwen2.5](https://huggingface.co/Z1zs/CausalQwen2.5)             | 81.1                                                                          | Qwen2.5-VL      | • Based on `Qwen/Qwen2.5-VL-3B-Instruct`.<br />• Checkpoint used in the CausalEmbed paper.                                                                         | ✅                   |
| [Z1zs/CausalPali](https://huggingface.co/Z1zs/CausalPali)   | 75.0                                                                          | Gemma      | • Based on `google/paligemma-3b-mix-448`.<br />• Fix right padding for queries.                                                                                | ✅                   |

## Environment Installation

```bash
# Create and activate a Conda environment
conda create -n causal python=3.12
conda activate causal

# Install dependencies
pip install -r requirements.txt

# Install FlashAttention for faster attention computation
pip install flash-attn

# Install ColPali engine in editable mode
python -m pip install -e .
```

## Training
We use `accelerate` for multi-GPU training. Example commands::
```bash
accelerate launch --multi-gpu train_causalqwen25.py \
  --epoch 1 \
  --dtoken_num 32 \
  --qtoken_num 16 \
  --loss symmargin \
  --wm 1 \
  --wp 0.1 \
  --wn 0.1 \
  --wq 0.1 \
  --bs 8
```

```bash
accelerate launch --multi-gpu train_causalqwen25.py \
  --epoch 1 \
  --dtoken_num 32 \
  --qtoken_num 16 \
  --loss symmargin \
  --wm 1 \
  --wp 0.1 \
  --wn 0.1 \
  --wq 0.1 \
  --bs 3
```

### Key Arguments
`--loss`: Loss function to use. Options: `["ce", "pairwise", "symmargin"]`  

**Recommendation**: Use `pairwise` (without regularization) or `symmargin` (with regularization).

`--ckpt`: Path to the renamed model weights for resuming training (run `rename.py` first to ensure weight keys match the expected format).

`--wm/wp/wn/wq`: Weights for the main loss, positive regularization, negative regularization, and query regularization, respectively.

## Weight Renaming
To ensure checkpoints can be loaded properly when resuming training, rename the `.safetensors` files using the provided scripts:

In `rename_xxx.py`, set:

`checkpoint_path`: the source checkpoint directory

`output_path`: the target directory for renamed weights

Run the renaming scripts:

```bash
python rename_qwen.py
python rename_pali.py
```

## Evaluation
Evaluate the trained model using:
```bash
bash eval.sh
```
It will evaluate the preformance of CausalEmbed model on each subset of ViDoRe.


## Contributing

We welcome contributions to CausalEmbed! 🤗  
Please submit your PR on new models/results/functions freely, and we will review it as soon as possible.

## Citation

**CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding**  

Authors: Jiahao Huo, Yu Huang, Yibo Yan, Ye Pan, Yi Cao, Mingdong Ou, Philip S. Yu, Xuming Hu

```latex
@misc{huo2026causalembed,
      title={CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding}, 
      author={Jiahao Huo and Yu Huang and Yibo Yan and Ye Pan and Yi Cao and Mingdong Ou and Philip S. Yu and Xuming Hu},
      year={2026},
      eprint={2601.21262},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.21262}, 
}
```
