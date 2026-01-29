#!/bin/bash

MODEL_PATH="model_path"
SAVE_DIR="./results/model"
GPUS=(0 1 2 3)
MAX_JOBS_PER_GPU=4 
BATCH_SIZE=64

declare -A BEIR_TASKS
BEIR_TASKS["vidore/esg_reports_v2"]="c05a1da867bbedebef239d4aa96cab19160b3d88"
BEIR_TASKS["vidore/biomedical_lectures_v2"]="9daa25abc1026f812834ca9a6b48b26ecbc61317"
BEIR_TASKS["vidore/economics_reports_v2"]="909aa23589332c30d7c6c9a89102fe2711cbb7a9"
BEIR_TASKS["vidore/esg_reports_human_labeled_v2"]="d8830ba2d04b285cfb2532b95be3748214e305da"

QA_TASKS=(
    "vidore/arxivqa_test_subsampled"
    "vidore/docvqa_test_subsampled"
    "vidore/infovqa_test_subsampled"
    "vidore/tabfquad_test_subsampled"
    "vidore/tatdqa_test"
    "vidore/syntheticDocQA_government_reports_test"
    "vidore/syntheticDocQA_healthcare_industry_test"
    "vidore/shiftproject_test"
    "vidore/syntheticDocQA_artificial_intelligence_test"
    "vidore/syntheticDocQA_energy_test"
)

gpu_idx=0
job_count=0
total_gpus=${#GPUS[@]}

# BEIR
for ds in "${!BEIR_TASKS[@]}"; do
    rev=${BEIR_TASKS[$ds]}
    current_gpu=${GPUS[$gpu_idx]}
    
    echo "Launching BEIR: $ds on GPU $current_gpu"
    CUDA_VISIBLE_DEVICES=$current_gpu python eval.py \
        --model_path "$MODEL_PATH" \
        --dataset_name "$ds" \
        --task_type "beir" \
        --revision "$rev" \
        --batch_size 64 \
        --savedir_datasets "$SAVE_DIR" &
    
done
# QA
for ds in "${QA_TASKS[@]}"; do
    current_gpu=${GPUS[$gpu_idx]}
    
    echo "Launching QA: $ds on GPU $current_gpu"
    CUDA_VISIBLE_DEVICES=$current_gpu python eval.py \
        --model_path "$MODEL_PATH" \
        --dataset_name "$ds" \
        --task_type "qa" \
        --batch_size 64 \
        --savedir_datasets "$SAVE_DIR" &
    
done

wait
echo "All evaluation tasks completed."
