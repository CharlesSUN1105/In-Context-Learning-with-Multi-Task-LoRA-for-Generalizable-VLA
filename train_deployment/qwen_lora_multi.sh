#!/bin/bash
#SBATCH --job-name=roboprompt-vllm        
#SBATCH --output=job_logs/%j/vllm_%j.out  
#SBATCH --error=job_logs/%j/vllm_%j.err   
#SBATCH --ntasks=1                        
#SBATCH --gres=gpu:h200:1                      
#SBATCH --time=12:00:00                   
#SBATCH --partition=agent-long     


# --------------- environment preparation ---------------
source ~/.conda_init
conda activate qwen                      #  conda environment name

# print current public IP
curl ifconfig.me; echo

# --------------- start vLLM service ---------------
OMP_NUM_THREADS=4 vllm serve /home/s84414554/qwen3_finetune/checkpoints/Qwen3-8B \
  --served-model-name "Qwen3-8B" \
  --enable-lora \
  --lora-modules \
      close_jar_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/close_jar \
      put_money_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/put_money_in_safe \
      drawer_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/open_drawer \
      place_wine_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/place_wine_at_rack_location \
      put_groceries_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/put_grocery \
      stack_cups_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/stack_cup \
      light_bulb_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/light_bulb_in \
      stack_blocks_adapter=/home/s84414554/qwen3_finetune/roboprompt-data/output/stack_blocks \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --port 8001 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  # --quantization bitsandbytes



# &> vllm_launch.log &
