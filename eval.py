import os
import sys
import argparse
import json
import torch
from typing import Optional
from datasets import load_dataset
from datetime import datetime
from importlib.metadata import version

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA, ViDoReEvaluatorBEIR
from vidore_benchmark.evaluation.interfaces import MetadataModel, ViDoReBenchmarkResults
from vidore_benchmark.retrievers import VisionRetriever, BaseVisionRetriever
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor, CausalQwen2_5
from colpali_engine.models import CausalPali
from colpali_engine.models import ColPaliProcessor

class ViDoReEvaluatorBEIRV3(ViDoReEvaluatorBEIR):
    def __init__(self, vision_retriever: BaseVisionRetriever):
        super().__init__(vision_retriever)
        self.corpus_id_column = "corpus_id"
        self.query_id_column = "query_id"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument('--task_type', type=str, choices=['beir', 'qa', 'v3'], required=True)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--savedir_datasets', type=str, default='./results/eval_results')
    parser.add_argument("--dtoken_num", type=int, default=32)
    parser.add_argument("--qtoken_num", type=int, default=16)
    parser.add_argument("--max_vtoken", type=int, default=768)
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.savedir_datasets, exist_ok=True)

    sanitized_name = args.dataset_name.replace("/", "_")
    savepath_results = os.path.join(args.savedir_datasets, f"{sanitized_name}_metrics.json")
    
    if os.path.exists(savepath_results):
        print(f"Skipping {args.dataset_name}, already exists.")
        return

    # 1.load model
    device = torch.device("cuda:0")
    
    if "pali" in args.model_path.lower():
        processor = ColPaliProcessor.from_pretrained(args.model_path)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.add_tokens("<|latent|>")
        latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")

        model = CausalPali.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            doc_token_num=args.dtoken_num, 
            query_token_num=args.qtoken_num,
            attn_implementation="flash_attention_2",
        )
        model.latent_token_id = latent_id
        model.resize_token_embeddings(len(processor.tokenizer))
    else:
        processor = ColQwen2_5_Processor.from_pretrained(
            args.model_path, 
            max_num_visual_tokens=args.max_vtoken, 
            fix_mistral_regex=True
        )
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.add_tokens("<|latent|>")
        latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")

        model = CausalQwen2_5.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            doc_token_num=args.dtoken_num, 
            query_token_num=args.qtoken_num,
            attn_implementation="flash_attention_2",
        )
        model.latent_token_id = latent_id

    model.to(device)
    model.eval()

    vision_retriever = VisionRetriever(model=model, processor=processor)
    
    # 2. evaluation
    print(f"Evaluating dataset: {args.dataset_name} on GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    if True:
        metrics = {}
        if args.task_type in ["beir", "v3"]:
            evaluator = ViDoReEvaluatorBEIR(vision_retriever) if args.task_type == "beir" else ViDoReEvaluatorBEIRV3(vision_retriever)
            ds = {
                "corpus": load_dataset(args.dataset_name, name="corpus", split="test", revision=args.revision),
                "queries": load_dataset(args.dataset_name, name="queries", split="test", revision=args.revision),
                "qrels": load_dataset(args.dataset_name, name="qrels", split="test", revision=args.revision)
            }
            metrics_dict = evaluator.evaluate_dataset(
                ds=ds, batch_query=args.batch_size, batch_passage=args.batch_size,
                batch_score=512, dataloader_prebatch_query=512, dataloader_prebatch_passage=512,
                save_dir=args.savedir_datasets
            )
            metrics = {args.dataset_name: metrics_dict}
            
        elif args.task_type == "qa":
            evaluator = ViDoReEvaluatorQA(vision_retriever)
            metrics_dict = evaluator.evaluate_dataset(
                ds=load_dataset(args.dataset_name, split="test"),
                batch_query=args.batch_size, batch_passage=args.batch_size,
                batch_score=64, dataloader_prebatch_query=128, dataloader_prebatch_passage=128,
                save_dir=args.savedir_datasets
            )
            metrics = {args.dataset_name: metrics_dict}

        # 3. save results
        results = ViDoReBenchmarkResults(
            metadata=MetadataModel(
                timestamp=datetime.now(),
                vidore_benchmark_version=version("vidore_benchmark"),
            ),
            metrics=metrics,
        )
        
        with open(savepath_results, "w", encoding="utf-8") as f:
            f.write(results.model_dump_json(indent=4))
            
        print(f"Successfully finished {args.dataset_name}")

if __name__ == "__main__":
    main()
