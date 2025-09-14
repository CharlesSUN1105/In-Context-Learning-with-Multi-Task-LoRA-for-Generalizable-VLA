# In-Context Learning with Multi-Task LoRA for Generalizable VLA: Setup and Evaluation Guide

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
  - [1.1 Clone the Repository and Install Dependencies](#11-clone-the-repository-and-install-dependencies)
  - [1.2 Configure OpenAI API / vLLM API](#12-configure-openai-api--vllm-api)
- [2. Dataset Generation](#2-dataset-generation)
  - [2.1 Navigate to the RLBench Data Generation Directory](#21-navigate-to-the-rlbench-data-generation-directory)
  - [2.2 Generate Training Data (100 Episodes)](#22-generate-training-data-100-episodes)
  - [2.3 Generate Test Data (25 Episodes)](#23-generate-test-data-25-episodes)
  - [2.4 Example Directory Structure](#24-example-directory-structure)
- [3. Building In-Context Learning Demonstrations (ICL Prompts)](#3-building-in-context-learning-demonstrations-icl-prompts)
- [4. Running Evaluation](#4-running-evaluation)
  - [4.1 Establish Remote Connection to vLLM](#41-establish-remote-connection-to-vllm)
  - [4.2 Expected Logs](#42-expected-logs)
  - [4.3 Results Directory](#43-results-directory)
- [5. Notes and Common Issues](#5-notes-and-common-issues)
- [6. End-to-End Workflow](#6-end-to-end-workflow)

---

## 1. Environment Setup

### 1.1 Clone the Repository and Install Dependencies
```bash
pip install -r requirements.txt

cd RLBench
pip install -e .
```

### 1.2 Configure OpenAI API / vLLM API
```bash
API: export OPENAI_API_KEY=sk-xxxxxx
vLLM: export OPENAI_API_BASE="http://localhost:8001/v1"
```

---

## 2. Dataset Generation

> Replace task names in the commands with your target RLBench task (e.g., `push_buttons`, `stack_cups`).

### 2.1 Navigate to the RLBench Data Generation Directory
```bash
cd RLBench/tools
```

### 2.2 Generate Training Data (100 Episodes)
```bash
DISPLAY=:0.0 python dataset_generator.py   --tasks=stack_cups   --save_path=data/train   --renderer=opengl   --episodes_per_task=100   --processes=1   --variations=1   --all_variations=False

mv data/train/stack_cups/variation0    data/train/stack_cups/all_variations
```

### 2.3 Generate Test Data (25 Episodes)
```bash
DISPLAY=:0.0 python dataset_generator.py   --tasks=stack_cups   --save_path=data/test   --renderer=opengl   --episodes_per_task=25   --processes=1   --variations=1   --all_variations=False

mv data/test/stack_cups/variation0    data/test/stack_cups/all_variations
```

### 2.4 Example Directory Structure
```
data/train/push_buttons/all_variations/episodes/episode0/
data/test/push_buttons/all_variations/episodes/episode0/
```

---

## 3. Building In-Context Learning Demonstrations (ICL Prompts)

1. Open `form_icl_demonstrations.py` in the project root.  
2. Set the `ROOT` variable to the training dataset path:
   ```python
   ROOT = "/to/your/path"
   ```
3. (Optional) Adjust parameters `NUM_DEMOS` and `NUM_PROMPTS`.  
4. Run the script:
   ```bash
   python form_icl_demonstrations.py
   ```
5. Verify output: check `icl_demos/push_buttons_*.txt`. Each line should look like:
   ```json
   {"button": [x, y, z]} > [[action_sequence_1], [action_sequence_2], ...]
   ```

---

## 4. Running Evaluation

### 4.1 Establish Remote Connection to vLLM

Establish a vLLM deployment service on the remote server side:
```bash
cd train_deployment

sbatch qwen_lora_multi.sh
```

Use SSH locally to connect to the login server and compute server on the remote server:
```bash
ssh -fN   -L 8001:lrc-alpha-sg-gpuxx:8001   id@27.54.45.176

lsof -i:8001 # check connection
```

Set vLLM API endpoint:
```bash
export OPENAI_API_BASE="http://localhost:8001/v1"
```

Run evaluation:
```bash
DISPLAY=:0.0 python main.py   model.llm_call_style=openai   model.name=Qwen3-8B   rlbench.tasks=[place_wine_at_rack_location]   rlbench.task_name=place_wine_at_rack_location   rlbench.episode_length=25   rlbench.demo_path=data/test   framework.gpu=0   framework.logdir=/logs/place_wine_at_rack_location_icl   framework.eval_episodes=25   rlbench.headless=False   cinematic_recorder.enabled=false
```

### 4.2 Expected Logs
```
eval_env: Starting episode…
step: X
Evaluating push_buttons | Episode N | Score: X.X | Lang Goal: …
```

### 4.3 Results Directory
```
logs/push_buttons_icl/push_buttons/RoboPrompt/seed0/
```

Contains:
- `score.txt` — per-episode success rate  
- `metrics.json` — detailed statistics  
- `videos/` — if recording enabled

---

## 5. Notes and Common Issues

- Task names **must** match RLBench script file names (e.g., `push_buttons`).  
- If you hit `KeyError` or missing demos: verify `demo_path`, task name, and directory structure.  
- For multi-task evaluation, pass a list:
  ```bash
  rlbench.tasks=[open_drawer,push_buttons]
  ```

---

## 6. End-to-End Workflow

Follow the four stages—environment setup, dataset generation, ICL prompt construction, and evaluation—to complete the RoboPrompt pipeline.
