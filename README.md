# README
This repository is built upon the official implementation of [ColPali](https://github.com/illuin-tech/colpali).

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
  --bs 1
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
  --bs 1
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