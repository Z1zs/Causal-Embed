import argparse
import shutil
from pathlib import Path
import os
import json
import torch
from torch import nn
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss, ColbertSymMarginPairwiseCELoss
from colpali_engine.models import CausalPali
from colpali_engine.models import ColPaliProcessor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./models/causalpali", help="where to write model + script copy")
    p.add_argument("--model_dir", type=str, default="vidore/colpaligemma-3b-mix-448-base", help="model path")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="generative", choices=["ce", "pairwise", "symmargin"], help="loss function to use")
    p.add_argument("--peft", action="store_true", help="use PEFT for training")
    p.add_argument("--epoch", type=int, default=3, help="epoches")
    p.add_argument("--trainset", type=str, default="col",choices=["col"], help="train dataset")
    p.add_argument("--dim", type=int, default=128, help="dim for projection")
    p.add_argument("--dtoken_num", type=int, default=32, help="num of doc tokens")
    p.add_argument("--qtoken_num", type=int, default=16, help="num of query tokens")
    p.add_argument("--bs", type=int, default=8, help="batch size")
    p.add_argument("--max_vtoken", type=int, default=768, help="batch size")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
    p.add_argument("--wp", type=float, default=0, help="weight for pos regular")
    p.add_argument("--wn", type=float, default=0, help="weight for neg regular")
    p.add_argument("--wm", type=float, default=1, help="weight for main")
    p.add_argument("--wq", type=float, default=0, help="weight for query regular")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(
            normalize_scores=False,
        )
    elif args.loss == "symmargin":
        loss_func = ColbertSymMarginPairwiseCELoss(
            normalize_scores=False,
            weight_pos=args.wp,
            weight_neg=args.wn,
            weight_main=args.wm,
            weight_query_reg=args.wq
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")


    processor=ColPaliProcessor.from_pretrained(
                pretrained_model_name_or_path="vidore/colpali-v1.3",max_num_visual_tokens=10
            )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.add_tokens("<|latent|>")
    latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")

    model=CausalPali.from_pretrained(
            pretrained_model_name_or_path=args.model_dir,
            torch_dtype=torch.bfloat16,
            doc_token_num=args.dtoken_num, query_token_num=args.qtoken_num,
            attn_implementation="flash_attention_2",
        )
    model.latent_token_id = latent_id
    model.resize_token_embeddings(len(processor.tokenizer))

    train_set=load_train_set()

    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=processor,
        model=model,
        train_dataset=train_set,
        eval_dataset=ColPaliEngineDataset(
            load_dataset("vidore/colpali_train_set", split="test"), pos_target_column_name="image"
        ),
        run_eval=True,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=args.epoch,
            per_device_train_batch_size=args.bs,
            gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            dataloader_num_workers=8,
            save_steps=0.1,
            logging_steps=10,
            eval_steps=200,
            warmup_steps=100,
            learning_rate=args.lr,
            save_strategy="steps",
            resume_from_checkpoint=args.ckpt,
            report_to="wandb",
            project="CausalEmbed"
        ),
        peft_config=LoraConfig(
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        )
        if args.peft
        else None,
    )

    # save models
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    args_dict = vars(args) 
    args_save_path = Path(config.output_dir) / "training_args.json"
    with open(args_save_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False) 
    print(f"Weights saved in: {args_save_path}")

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
