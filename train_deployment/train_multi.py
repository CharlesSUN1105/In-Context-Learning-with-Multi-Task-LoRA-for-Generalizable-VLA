import os
os.environ["TRANSFORMERS_NO_MLX"] = "1"
import transformers.utils.generic as _gen
_gen.is_mlx_available = lambda *args, **kwargs: False
if hasattr(_gen, "_is_mlx_available"):
    _gen._is_mlx_available = lambda *args, **kwargs: False

import re
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler, DataLoader

import swanlab
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import IntervalStrategy
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback

# =========================
# —— Configuration Area ——
# =========================
MODEL_DIR    = "/home/s84414554/qwen3_finetune/checkpoints/Qwen3-8B"
TRAIN_FILE   = "/home/s84414554/qwen3_finetune/roboprompt-data/combine_10shot.jsonl"
VALID_FILE   = "/home/s84414554/qwen3_finetune/roboprompt-data/combine_eval.jsonl"
OUTPUT_DIR   = "/home/s84414554/qwen3_finetune/roboprompt-data/output/rloraAB2"
MAX_LENGTH   = 2500  # Fix: Increase maximum length to ensure complete action sequences can be accommodated
PROJECT_NAME = "rlora_AB"

# =========================
# —— R-LoRA Key Hyperparameters (Simplified Version, A Matrix Always Training) ——
# =========================
H = 3
ADAPTER_DROPOUT = 0.04  # Optimization: Further reduce adapter dropout to reduce interference with trajectory prediction
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Use shared A + independent B architecture (H branch mode)
# A matrix always trains, using small learning rate
A_LEARNING_RATE_RATIO = 0.2  # Optimization: Further reduce A matrix learning rate to improve trajectory prediction accuracy

H_DROPOUT_P = 0.06      # Optimization: Further reduce H-dropout to reduce interference with trajectory prediction
H_DROPOUT_SEED_SHIFT = 12345
B_INIT_STD = 0.04       # Optimization: Further reduce initialization standard deviation to improve trajectory prediction accuracy
B_INIT_GAMMA = 64     # Optimization: Increase gamma scaling to improve convergence stability
GATE_WARMUP_STEPS = 600 # Optimization: Extend gate warmup for smoother transition
STAGE1_RATIO = 0.5      # Optimization: Extend pure learning stage to let model better learn trajectory patterns
STAGE2_RATIO = 0.3     # Maintain progressive regularization stage
STAGE3_RATIO = 0.2     # Optimization: Further shorten complete regularization stage to avoid over-regularization

# =========================
# 1) SwanLab
# =========================
os.environ["SWANLAB_PROJECT"] = PROJECT_NAME
swanlab.config.update({
    "model": MODEL_DIR,
    "data_max_length": MAX_LENGTH,
})

# =========================
# 2) Load Base Model
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, trust_remote_code=True, local_files_only=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, device_map="auto",
    torch_dtype="auto", local_files_only=True
)

# =========================
# 3) Base LoRA (Shared A Matrix Base Adapter)
# =========================
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,   # Disable built-in dropout, use self-implemented column-level H-dropout
    target_modules=TARGET_MODULES,
)
model = get_peft_model(base_model, lora_cfg)
model.active_adapter = "default"  # This serves as the base adapter for shared A matrix

# =========================
# 4) Data Loading and Task Discovery
# =========================
train_raw = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
val_raw   = load_dataset("json", data_files={"validation": VALID_FILE})["validation"]

# Fix: Shuffle training data order to ensure mixed task training
print("🔀 Shuffling training data order to ensure mixed task training...")
train_raw = train_raw.shuffle(seed=42)
print(f"✅ Training data shuffled, total samples: {len(train_raw)}")

def _discover_tasks(*dss):
    s = set()
    for ds in dss:
        if "task" in ds.column_names:
            for t in ds["task"]:
                s.add(t if isinstance(t, str) and len(t) > 0 else "default")
    if not s:
        s.add("default")
    return sorted(list(s))

TASKS = _discover_tasks(train_raw, val_raw)
TASK2ID = {t: i for i, t in enumerate(TASKS)}

# Create shared A + independent B adapters: Each task creates H independent B matrix adapters (sharing A matrix)
print(f"🔄 Using shared A + independent B architecture (H={H} branches)")
for t in TASKS:
    for h in range(H):
        model.add_adapter(f"{t}_B#{h}", lora_cfg)

# ====== Override B random initialization + scaling (including default) ======
def reinit_lora_B(module, adapter_name):
    # peft's lora_B is ModuleDict: module.lora_B[adapter_name].weight
    if hasattr(module, "lora_B") and adapter_name in getattr(module, "lora_B").keys():
        W = module.lora_B[adapter_name].weight
        if isinstance(W, nn.Parameter):
            nn.init.normal_(W, mean=0.0, std=B_INIT_STD)
            with torch.no_grad():
                W.div_(math.sqrt(B_INIT_GAMMA))  # Scaling

def for_all_lora_modules(fn):
    for m in model.modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
            for name in list(m.lora_B.keys()):
                fn(m, name)

for_all_lora_modules(reinit_lora_B)

# =========================
# Architecture Setup Functions
# =========================
def setup_strict_shared_A_independent_B():
    """
    Setup strict shared A + independent B architecture (shared subspace):
    - All task adapters' A matrices are directly bound to default's A matrix (parameter binding)
    - Keep each task's B matrices independent
    - A matrix always trains, using small learning rate
    """
    print("🔧 Setting up strict shared A + independent B architecture (shared subspace)...")
    
    # Parameter binding: Let all task adapters' A matrices directly point to default's A matrix
    # This structurally makes it one A, shared by all branches, only registered once in optimizer
    for task in TASKS:
        for h in range(H):
            adapter_name = f"{task}_B#{h}"
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and adapter_name in module.lora_A and 'default' in module.lora_A:
                    # Parameter binding: Let task adapter's A matrix directly reference default's A matrix
                    module.lora_A[adapter_name] = module.lora_A['default']
    
    print("✅ Strict shared A + independent B architecture setup complete - parameter binding shared subspace")
    print("📊 All branches' A matrices bound to default, only one A parameter registered in optimizer")
    print(f"🎯 A matrix always trains, learning rate is {A_LEARNING_RATE_RATIO*100:.0f}% of main learning rate")


# Shared A + independent B task manager
class TaskManager:
    def __init__(self, model_ref=None):
        self.current_task = None
        self.current_head = 0
        self.model_ref = model_ref  # Store model reference to avoid using global variables directly
    
    def set_task(self, task_name, head=0):
        self.current_task = task_name
        self.current_head = head
        
        # Use passed model reference, fallback to global model if not provided
        target_model = self.model_ref if self.model_ref is not None else model
        
        if task_name == "default":
            target_model.set_adapter("default")
        else:
            # Shared A + independent B mode (H branches)
            adapter_name = f"{task_name}_B#{head}"
            target_model.set_adapter(adapter_name)
    
    def get_current_task(self):
        return self.current_task
    
    def get_current_head(self):
        return self.current_head

# Setup strict shared A + independent B architecture and create manager
setup_strict_shared_A_independent_B()
task_manager = TaskManager(model_ref=model)

# ====== Set LoRA parameter trainability (shared A + independent B architecture) ======
def enable_strict_shared_A_independent_B_trainable(m: nn.Module):
    """
    Strict shared A + independent B architecture parameter setup:
    - A matrix always trains (strict shared subspace)
    - All branches' A matrices never train (parameter binding)
    - All B matrices train independently
    """
    enabled_params = 0
    a_params = 0
    b_params = 0
    
    for mod in m.modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            # Only set default's A matrix as trainable, don't traverse other adapters' A (they are the same object)
            if 'default' in mod.lora_A:
                for p in mod.lora_A['default'].parameters():
                    p.requires_grad = True
                    a_params += p.numel()
            
            # Enable gradients for all B matrices (each task's B matrices train independently)
            for _name, sub in getattr(mod, "lora_B").items():
                for p in sub.parameters():
                    if not p.requires_grad:
                        p.requires_grad_(True)
                        enabled_params += p.numel()
                        b_params += p.numel()
    
    print(f"✅ Strict shared A + independent B parameter setup complete:")
    print(f"   - Shared A matrix parameters (default): {a_params:,} (always training)")
    print(f"   - Independent B matrix parameters: {b_params:,} (trainable)")
    print(f"   - Current trainable parameters: {enabled_params:,}")

enable_strict_shared_A_independent_B_trainable(model)

# Verify strict shared A matrix setup
def verify_strict_shared_A_setup(m: nn.Module):
    """Verify that strict shared A matrix setup is correct"""
    print("🔍 Verifying strict shared A matrix setup...")
    
    default_a_trainable = 0
    branch_a_trainable = 0
    b_trainable = 0
    
    # Verify parameter binding
    binding_check = {}
    unique_a_params = set()
    
    for mod in m.modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            # Check A matrix status and parameter binding
            for adapter_name, lora_a in getattr(mod, "lora_A").items():
                for p in lora_a.parameters():
                    param_id = id(p)
                    unique_a_params.add(param_id)
                    
                    if adapter_name == 'default':
                        binding_check['default'] = param_id
                        if p.requires_grad:
                            default_a_trainable += 1
                    else:
                        binding_check[adapter_name] = param_id
                        if p.requires_grad:
                            branch_a_trainable += 1
            
            # Check B matrix status
            for _name, sub in getattr(mod, "lora_B").items():
                for p in sub.parameters():
                    if p.requires_grad:
                        b_trainable += 1
    
    # Verify parameter binding is correct
    binding_correct = True
    if 'default' in binding_check:
        default_id = binding_check['default']
        for adapter_name, param_id in binding_check.items():
            if adapter_name != 'default' and param_id != default_id:
                binding_correct = False
                print(f"❌ Parameter binding error: {adapter_name} not bound to default")
                break
    
    # Calculate actual A matrix layer count (each LoRA layer has one A matrix)
    total_lora_layers = len([mod for mod in m.modules() if hasattr(mod, "lora_A")])
    
    # Calculate A parameter binding status for each layer
    layers_with_correct_binding = 0
    for mod in m.modules():
        if hasattr(mod, "lora_A"):
            layer_binding_correct = True
            default_a_id = None
            for adapter_name, lora_a in mod.lora_A.items():
                for param in lora_a.parameters():
                    if adapter_name == 'default':
                        default_a_id = id(param)
                    elif default_a_id is not None and id(param) != default_a_id:
                        layer_binding_correct = False
                        break
            if layer_binding_correct and default_a_id is not None:
                layers_with_correct_binding += 1
    
    print(f"📊 Verification results:")
    print(f"   - LoRA layers: {total_lora_layers}")
    print(f"   - Unique A matrix parameter objects: {len(unique_a_params)} (shared within each layer)")
    print(f"   - Correctly bound layers: {layers_with_correct_binding}/{total_lora_layers}")
    print(f"   - Default A matrix trainable parameters: {default_a_trainable} (should >0, always training)")
    print(f"   - Branch A matrix trainable parameters: {branch_a_trainable} (shared with default when parameter binding)")
    print(f"   - B matrix trainable parameters: {b_trainable} (should >0)")
    print(f"   - Parameter binding status: {'✅ Correct' if binding_correct else '❌ Error'}")
    
    if (layers_with_correct_binding == total_lora_layers and default_a_trainable > 0 and 
        b_trainable > 0 and binding_correct):
        print("✅ Strict shared A matrix setup completely correct: parameter binding effective within each layer, A matrix always training, B matrices trainable")
        print("📝 Note: Branch A matrices show trainable due to parameter binding (sharing same object with default)")
        return True
    else:
        print("❌ Setup error, please check the above issues")
        return False

verify_strict_shared_A_setup(model)

try:
    model.print_trainable_parameters()
except Exception:
    pass

# =========================
# 5) Preprocessing: Only learn "first JSON array" as supervision target (action sequence)
# =========================
def _get_messages(example):
    for k in ("messages", "conversations", "chat"):
        if k in example and isinstance(example[k], list):
            return example[k]
    raise ValueError("Sample missing messages/conversations/chat field, or its type is not list.")

def strip_code_fence(text: str) -> str:
    """Remove 'Prediction:' and ```json fences"""
    t = text.strip()
    t = re.sub(r"^\s*Prediction\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    return t

def extract_first_json_array(text: str):
    """Extract first complete JSON array substring; return None if not found"""
    if not isinstance(text, str):
        text = str(text)
    t = strip_code_fence(text)
    s = t.find("[")
    if s == -1:
        return None
    depth = 0
    for i, ch in enumerate(t[s:], s):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return t[s:i+1]
    return None

def preprocess(ex):
    # 1) Get conversation and last assistant output
    msgs = _get_messages(ex)
    last_asst_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None:
        return {"input_ids": [], "attention_mask": [], "labels": [], "task_id": -1}

    prompt_messages = msgs[:last_asst_idx]
    target_text = msgs[last_asst_idx].get("content", "")
    if not isinstance(target_text, str):
        target_text = str(target_text)

    # 2) Only extract first JSON array; discard failed samples directly
    array_payload = extract_first_json_array(target_text)
    if not array_payload:
        return {"input_ids": [], "attention_mask": [], "labels": [], "task_id": -1}

    # 3) Structure validation
    try:
        arr = json.loads(array_payload)
        if not isinstance(arr, list):
            return {"input_ids": [], "attention_mask": [], "labels": [], "task_id": -1}
    except json.JSONDecodeError:
        return {"input_ids": [], "attention_mask": [], "labels": [], "task_id": -1}

    # 4) Encoding: prompt uses chat template; target only array + EOS
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    target_ids = tokenizer(array_payload, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    # 5) Concatenate and truncate (right-align target, prompt may be truncated)
    max_prompt_len = MAX_LENGTH - len(target_ids)
    if max_prompt_len <= 0:
        # If target is too long, discard this sample directly to avoid learning incomplete sequences
        return {"input_ids": [], "attention_mask": [], "labels": [], "task_id": -1}
    else:
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
        input_ids = prompt_ids + target_ids
        labels    = ([-100] * len(prompt_ids)) + target_ids

    # 6) pad and attention_mask
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if len(input_ids) < MAX_LENGTH:
        pad_len   = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels    = labels    + [-100]   * pad_len
    attention_mask = [0 if tok == pad_id else 1 for tok in input_ids]

    # 7) Map task_id (required for multi-task)
    task_name = ex.get("task", "default")
    if not isinstance(task_name, str) or len(task_name) == 0:
        task_name = "default"
    task_id = TASK2ID.get(task_name, 0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task_id": task_id,
    }

remove_cols_train = [c for c in train_raw.column_names if c not in ("messages", "conversations", "chat", "task")]
remove_cols_val   = [c for c in val_raw.column_names   if c not in ("messages", "conversations", "chat", "task")]
train_tokenized = train_raw.map(preprocess, remove_columns=remove_cols_train, desc="Tokenizing train (array-only)")
val_tokenized   = val_raw.map(preprocess,   remove_columns=remove_cols_val,   desc="Tokenizing valid (array-only)")
train_tokenized = train_tokenized.filter(lambda ex: len(ex["input_ids"]) > 0 and ex["task_id"] >= 0)
val_tokenized   = val_tokenized.filter(lambda ex: len(ex["input_ids"]) > 0 and ex["task_id"] >= 0)

# (Optional) Data health check
def _decode_target(ex):
    tgt = [tid for tid in ex["labels"] if tid != -100]
    return tokenizer.decode(tgt, skip_special_tokens=True)
print("===== Data Health Check (first 3 target decodings) =====")
for i in range(min(3, len(train_tokenized))):
    print(f"Sample {i} target:\n{_decode_target(train_tokenized[i])}\n" + "-"*40)

# =========================
# 6) Collator
# =========================
class SimpleCollator:
    def __call__(self, features):
        batch = {}
        for k in ("input_ids", "attention_mask", "labels"):
            batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
        if "task_id" in features[0]:
            batch["task_id"] = torch.tensor([f["task_id"] for f in features], dtype=torch.long)
        return batch

data_collator = SimpleCollator()

# =========================
# 7) Same Task Batching
# =========================
class SameTaskBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        buckets = {}
        for i in range(len(dataset)):
            tid = int(dataset[i]["task_id"])
            buckets.setdefault(tid, []).append(i)
        self.buckets = [v for _, v in sorted(buckets.items())]

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        buckets = [b[:] for b in self.buckets]
        if self.shuffle:
            for b in buckets:
                rng.shuffle(b)
            rng.shuffle(buckets)
        for b in buckets:
            for i in range(0, len(b), self.batch_size):
                batch = b[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch

    def __len__(self):
        return sum(len(b) // self.batch_size for b in self.buckets)

# =========================
# 8) Training Phase: Branch Gating + H-dropout (Prevent Permanent Zero)
# =========================
def _iter_lora_B_weights_for_adapter(m, adapter_name):
    """yield all lora_B.weight tensors under model for a given adapter"""
    for mod in m.modules():
        if hasattr(mod, "lora_B") and adapter_name in getattr(mod, "lora_B").keys():
            yield mod.lora_B[adapter_name].weight

def _apply_gate_and_hdrop(mask_dict, model, adapter_name, gate, p, seed):
    """
    Apply "temporary" column-wise scaling to specified adapter's B matrix:
      - First multiply entire matrix by gate (gating, 0→1)
      - Then multiply columns by bernoulli keep mask / (1-p) (H-dropout)
    After training step: apply 1/scale inverse scaling to kept columns; write back dropped columns using backup to avoid permanent zero.
    """
    device = next(model.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    mask_dict.clear()
    for W in _iter_lora_B_weights_for_adapter(model, adapter_name):
        r = W.shape[1]
        if p > 0 and model.training:
            keep = torch.bernoulli(torch.full((r,), 1.0 - p, device=W.device, dtype=W.dtype), generator=rng)
            scale = torch.where(keep > 0, 1.0/(1.0-p), torch.zeros_like(keep))
        else:
            scale = torch.ones(r, device=W.device, dtype=W.dtype)

        col_scale = gate * scale  # [r]
        dropped = (col_scale == 0)
        backup = None
        if torch.any(dropped):
            backup = W[:, dropped].detach().clone()  # Only backup dropped columns

        W.data.mul_(col_scale.unsqueeze(0))
        mask_dict[W] = (col_scale, dropped, backup)

def _restore_B(mask_dict):
    """Restore scaled B.weight:
    - For kept columns: apply 1/col_scale inverse scaling to preserve gradient updates;
    - For dropped columns: write back using pre-forward backup to avoid permanent zero.
    """
    for W, (col_scale, dropped, backup) in mask_dict.items():
        inv = torch.where(col_scale > 0, 1.0/col_scale, torch.ones_like(col_scale))
        W.data.mul_(inv.unsqueeze(0))
        if backup is not None:
            W.data[:, dropped] = backup
    mask_dict.clear()

class RLoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tmp_scale_cache = {}  # Store B column scaling and backup, restore after training step
        self._optimizer_param_groups = None  # Store custom parameter groups

    def get_train_dataloader(self):
        train_sampler = SameTaskBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True, seed=42
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Shared A + independent B architecture training step:
        - Use shared A matrix (from default adapter, always training)
        - Use task-specific H independent B matrix branches
        - Three-stage strategy only affects gate / adapter_dropout / h_dropout intensity
        """
        task_ids = inputs.pop("task_id", None)
        if task_ids is not None:
            try:
                task_id = int(task_ids[0].detach().cpu().item())
            except Exception:
                task_id = int(task_ids[0]) if hasattr(task_ids, "__getitem__") else 0
            task = TASKS[task_id]

            step = max(0, int(self.state.global_step))

            # Use trainer's real max_steps to calculate stage boundaries
            if hasattr(self.args, 'max_steps') and self.args.max_steps > 0:
                # If max_steps is set, use it as real total steps
                total_steps = self.args.max_steps
            else:
                # Otherwise get from trainer state or use DataLoader to calculate real steps
                if hasattr(self.state, 'max_steps') and self.state.max_steps > 0:
                    total_steps = self.state.max_steps
                else:
                    # Use DataLoader's actual length to calculate steps (considering SameTaskBatchSampler's tail batch dropping behavior)
                    try:
                        train_dataloader = self.get_train_dataloader()
                        # Fix: DataLoader length is batch count, need to divide by gradient_accumulation_steps to get optimizer steps
                        steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
                        num_epochs = self.args.num_train_epochs
                        total_steps = steps_per_epoch * num_epochs
                        # Only print debug info at training start (avoid repeated output)
                        if step == 0 or (step < 10 and step % 5 == 0):
                            print(f"🔍 Step calculation: DataLoader batches={len(train_dataloader)}, optimizer steps per epoch={steps_per_epoch}, total steps={total_steps}")
                    except Exception:
                        # Fallback to estimated calculation
                        train_samples = len(self.train_dataset) if hasattr(self, 'train_dataset') else 1000
                        per_device_bs = self.args.per_device_train_batch_size
                        grad_accum = self.args.gradient_accumulation_steps
                        num_epochs = self.args.num_train_epochs
                        steps_per_epoch = math.ceil(train_samples / (per_device_bs * grad_accum))
                        total_steps = steps_per_epoch * num_epochs
            
            stage1_steps = int(total_steps * STAGE1_RATIO)
            stage2_steps = int(total_steps * STAGE2_RATIO)

            # Stage 1: Pure learning, no regularization
            if step < stage1_steps:
                gate = 1.0
                current_adapter_dropout = 0.0
                current_h_dropout_p = 0.0

            # Stage 2: Progressive regularization
            elif step < stage1_steps + stage2_steps:
                progress = (step - stage1_steps) / max(1, stage2_steps)
                current_adapter_dropout = ADAPTER_DROPOUT * progress
                current_h_dropout_p = H_DROPOUT_P * progress
                gate = float(min(1.0, step / max(1, GATE_WARMUP_STEPS)))

            # Stage 3: Complete regularization
            else:
                current_adapter_dropout = ADAPTER_DROPOUT
                current_h_dropout_p = H_DROPOUT_P
                gate = 1.0

            # Shared A + independent B H branch rotation logic (unified rotation strategy)
            # First determine whether to use adapter_dropout to fallback to default
            rng = np.random.default_rng(self.state.global_step + 123)
            use_default = current_adapter_dropout > 0 and rng.random() < current_adapter_dropout
            
            if use_default:
                # Fallback to default adapter (regularization)
                task_manager.set_task("default")
                actual_task = "default"
                adapter_name = "default"
            else:
                # Fix: Further reduce H branch rotation frequency to give each branch more learning opportunities
                # Switch head every 50 steps to give complex tasks more learning time
                head = (step // 50) % H
                task_manager.set_task(task, head)
                actual_task = task
                adapter_name = f"{task}_B#{head}"

            # Apply gating and H-dropout
            seed = (self.state.global_step + H_DROPOUT_SEED_SHIFT + (hash(adapter_name) & 0xFFFF)) % (2**31-1)
            _apply_gate_and_hdrop(self._tmp_scale_cache, model, adapter_name, gate, current_h_dropout_p, int(seed))

        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)
        finally:
            if self._tmp_scale_cache:
                _restore_B(self._tmp_scale_cache)

            # Debug: Monitor stage and current LR (read from scheduler)
            debug_step = max(0, int(self.state.global_step))
            if debug_step % 100 == 0:
                # Recalculate debug step count info
                if hasattr(self.args, 'max_steps') and self.args.max_steps > 0:
                    debug_total_steps = self.args.max_steps
                else:
                    try:
                        debug_train_dataloader = self.get_train_dataloader()
                        # Fix: DataLoader length is batch count, need to divide by gradient_accumulation_steps to get optimizer steps
                        debug_steps_per_epoch = math.ceil(len(debug_train_dataloader) / self.args.gradient_accumulation_steps)
                        debug_num_epochs = self.args.num_train_epochs
                        debug_total_steps = debug_steps_per_epoch * debug_num_epochs
                    except Exception:
                        debug_total_steps = 1800  # Default value
                
                debug_stage1_steps = int(debug_total_steps * STAGE1_RATIO)
                debug_stage2_steps = int(debug_total_steps * STAGE2_RATIO)
                if debug_step < debug_stage1_steps:
                    stage = "STAGE1"
                elif debug_step < debug_stage1_steps + debug_stage2_steps:
                    stage = "STAGE2"
                else:
                    stage = "STAGE3"

                current_lr = None
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    try:
                        lrs = self.lr_scheduler.get_last_lr()
                        if lrs:
                            # Display B matrix learning rate (main training parameters), B matrix is now group 0
                            current_lr = lrs[0]  # B matrix learning rate (now group 0)
                    except Exception:
                        pass
                if current_lr is None and hasattr(self, 'optimizer') and self.optimizer and self.optimizer.param_groups:
                    # Also prioritize displaying B matrix learning rate (now group 0)
                    current_lr = self.optimizer.param_groups[0].get('lr', None)

                head = task_manager.get_current_head()
                adapter_info = f"Strict shared A (always training) + independent B (H={H}), task={actual_task}, head={head}"
                
                # Get all parameter group learning rates for debugging
                all_lrs = []
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    try:
                        all_lrs = self.lr_scheduler.get_last_lr()
                    except Exception:
                        pass
                
                try:
                    lr_info = f"lr={current_lr:.2e}"
                    if all_lrs and len(all_lrs) > 1:
                        lr_info += f" (B:{all_lrs[0]:.2e}, A:{all_lrs[1]:.2e})"
                    
                    print(f"[DEBUG] Step {self.state.global_step}: {stage}, {adapter_info}, gate={gate:.3f}, "
                          f"adapter_dropout={current_adapter_dropout:.3f}, h_dropout={current_h_dropout_p:.3f}, "
                          f"{lr_info}, loss={float(loss):.4f}")
                except Exception:
                    print(f"[DEBUG] Step {self.state.global_step}: {stage}, {adapter_info}, gate={gate}, "
                          f"adapter_dropout={current_adapter_dropout}, h_dropout={current_h_dropout_p}, "
                          f"lr={current_lr}, loss={loss}")

        return loss
    
    def set_optimizer_param_groups(self, param_groups):
        """Set custom optimizer parameter groups"""
        self._optimizer_param_groups = param_groups
        print("✅ Custom optimizer parameter groups set")
    
    def create_optimizer(self):
        """Override create_optimizer, use custom parameter groups"""
        print(f"🔍 create_optimizer called, _optimizer_param_groups status: {self._optimizer_param_groups is not None}")
        
        if self._optimizer_param_groups is not None:
            print("🔧 Creating optimizer using custom parameter groups...")
            print(f"📊 Parameter group count: {len(self._optimizer_param_groups)}")
            
            # Import optimizer
            from torch.optim import AdamW
            
            # Create optimizer using custom parameter groups
            optimizer = AdamW(
                self._optimizer_param_groups,
                lr=self.args.learning_rate,  # This will be overridden by lr in parameter groups
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,  # This will be overridden by weight_decay in parameter groups
            )
            
            print(f"✅ Custom optimizer created, param_groups count: {len(optimizer.param_groups)}")
            return optimizer
        else:
            # If no custom parameter groups set, use default method
            print("⚠️ No custom parameter groups set, using default optimizer creation method")
            optimizer = super().create_optimizer()
            print(f"🔍 Default optimizer creation result: {optimizer is not None}")
            if optimizer is None:
                raise RuntimeError("Default optimizer creation failed, please check model parameter settings")
            return optimizer
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Override create_optimizer_and_scheduler, ensure optimizer is not None"""
        print(f"🔍 create_optimizer_and_scheduler called, num_training_steps={num_training_steps}")
        
        # Create optimizer first
        self.optimizer = self.create_optimizer()
        print(f"🔍 Optimizer creation result: {self.optimizer is not None}")
        
        if self.optimizer is None:
            raise RuntimeError("Optimizer creation failed!")
        
        # Then create scheduler
        self.lr_scheduler = self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        print(f"🔍 Scheduler creation result: {self.lr_scheduler is not None}")
        
        return self.optimizer, self.lr_scheduler
    
    def set_eval_head(self, head: int):
        """Set evaluation head branch for optimal branch selection"""
        self._eval_head = head
        print(f"🎯 Set evaluation head to: {head}")
    
    def enable_external_head_control(self, enable: bool = True):
        """Enable/disable external head control for optimal branch selection to avoid being overridden"""
        self._respect_external_head = enable
        print(f"🔧 External head control: {'Enabled' if enable else 'Disabled'}")

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        """
        Evaluation phase using shared A + independent B architecture:
        - Allow using optimal branch for evaluation (not forced head=0)
        - No H-dropout / gating
        """
        inputs = dict(inputs)
        task_ids = inputs.pop("task_id", None)
        if task_ids is not None:
            try:
                task_id = int(task_ids[0].detach().cpu().item())
            except Exception:
                task_id = int(task_ids[0]) if hasattr(task_ids, "__getitem__") else 0
            task = TASKS[task_id]
            
            # During evaluation, respect externally set head (for optimal branch selection)
            # Only use default head=0 when no external settings (mid-evaluation)
            if not hasattr(self, '_respect_external_head') or not self._respect_external_head:
                # Mid-evaluation: use set eval_head or default head=0
                current_head = getattr(self, '_eval_head', 0)
                task_manager.set_task(task, head=current_head)
            # If _respect_external_head=True, don't override external settings, use task_manager current state directly
                
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **kwargs)
        except TypeError:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# =========================
# 9) Training Parameters
# =========================
num_epochs = 12 # Fix: Increase training epochs to ensure all tasks can learn sufficiently
per_device_bs = 2  # Fix: Increase batch size to improve training stability
grad_accum = 15   # Fix: Adjust gradient accumulation to maintain effective batch size
train_samples = len(train_tokenized)
steps_per_epoch = math.ceil(train_samples / (per_device_bs * grad_accum))
EVAL_STEPS = max(1, steps_per_epoch // 2)

print(f"Training samples: {train_samples}, batch size per step * accumulation: {per_device_bs*grad_accum}, steps per epoch: {steps_per_epoch}, eval every {EVAL_STEPS} steps")
total_steps = steps_per_epoch * num_epochs
stage1_steps = int(total_steps * STAGE1_RATIO)
stage2_steps = int(total_steps * STAGE2_RATIO)
print(f"Architecture mode: Strict shared A + independent B (H={H}) - Shared Subspace")
print(f"Stage 1 (0-{stage1_steps}): Intra-task rotation learning (shared A always training, head=step%{H}, gate=1, no dropout)")
print(f"Stage 2 ({stage1_steps}-{stage1_steps+stage2_steps}): Progressive regularization (shared A training)")
print(f"Stage 3 ({stage1_steps+stage2_steps}+): Complete regularization (shared A training)")
print(f"Shared A learning rate ratio: {A_LEARNING_RATE_RATIO*100:.0f}% ({A_LEARNING_RATE_RATIO} times main LR)")
print("📊 Strict sharing: A matrix always training, all branch A matrices forever frozen (parameter binding)")
print("=" * 50)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=5e-5,          # Optimization: Further reduce learning rate to improve trajectory prediction accuracy
    warmup_ratio=0.2,           # Optimization: Increase warmup ratio for smoother learning rate growth
    lr_scheduler_type="cosine",   # ← Unified use of cosine
    weight_decay=0.02,           # Optimization: Increase weight decay to improve generalization
    bf16=True,
    max_grad_norm=0.8,           # Optimization: Further reduce gradient clipping threshold to improve stability

    do_eval=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,

    save_total_limit=2,
    logging_steps=10,
    disable_tqdm=True,
    report_to=["swanlab"],
    run_name="combine_10shot-rlora-AB-stable",
    remove_unused_columns=False,  # Fix: Keep task_id for task routing during evaluation
)

print(f"Training hyperparameters: lr={training_args.learning_rate}, max_grad_norm={training_args.max_grad_norm}, warmup_ratio={training_args.warmup_ratio}")
print(f"Training configuration: epochs={training_args.num_train_epochs}, batch_size={training_args.per_device_train_batch_size}, grad_accum={training_args.gradient_accumulation_steps}")
print("=" * 50)

# =========================
# 10) Setup Optimizer (Simplified Version, Avoid Dynamic Adjustment)
# =========================
def setup_optimizer_with_different_lr(model, base_lr):
    """Setup optimizer with different learning rates for A matrix and B matrix"""
    print("🔧 Starting optimizer parameter group setup...")
    a_matrix_lr = base_lr * A_LEARNING_RATE_RATIO
    
    # Collect parameters
    a_matrix_params = []
    b_matrix_params = []
    other_params = []
    seen_params = set()  # For parameter deduplication
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
            # Use parameter object id for deduplication to avoid duplicate registration of bound parameters
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            
            if 'lora_A' in name:
                a_matrix_params.append(param)
            elif 'lora_B' in name:
                b_matrix_params.append(param)
            else:
                other_params.append(param)
    
    print(f"📊 Parameter statistics: total parameters={total_params}, trainable parameters={trainable_params}")
    print(f"📊 Parameter grouping: A matrix={len(a_matrix_params)}, B matrix={len(b_matrix_params)}, others={len(other_params)}")
    
    # Create parameter groups (B matrix first, so Transformers logs will display B matrix learning rate)
    param_groups = []
    if b_matrix_params:
        param_groups.append({
            'params': b_matrix_params,
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'lora_B'
        })
    if a_matrix_params:
        param_groups.append({
            'params': a_matrix_params,
            'lr': a_matrix_lr,
            'weight_decay': 0.0,
            'name': 'lora_A'
        })
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': 0.01,
            'name': 'other'
        })
    
    # Check if parameter groups are empty
    if not param_groups:
        raise RuntimeError("No trainable parameters found! Please check model settings.")
    
    print(f"📊 Optimizer parameter group setup:")
    print(f"   - A matrix: lr={a_matrix_lr:.2e}, weight_decay=0.0, params={len(a_matrix_params)}")
    print(f"   - B matrix: lr={base_lr:.2e}, weight_decay=0.0, params={len(b_matrix_params)}")
    if other_params:
        print(f"   - Other parameters: lr={base_lr:.2e}, weight_decay=0.01, params={len(other_params)}")
    
    return param_groups

# Setup optimizer parameter groups
optimizer_param_groups = setup_optimizer_with_different_lr(model, training_args.learning_rate)

# =========================
# 11) Start Training
# =========================
swanlab_callback = SwanLabCallback(project=PROJECT_NAME, experiment_name="rlora-array-only-stable")
trainer = RLoRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    callbacks=[swanlab_callback]
)

# Set custom optimizer parameter groups (set before train())
trainer.set_optimizer_param_groups(optimizer_param_groups)

print(f"Starting training, estimated total training steps: {steps_per_epoch * num_epochs}")
trainer.train()

# =========================
# 11) Select optimal branch per task and export shared A + independent B model
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📤 Selecting optimal B branch per task and exporting...")
# Enable external head control to avoid prediction_step overriding external settings
trainer.enable_external_head_control(True)

best_head = {}
for t in TASKS:
    tid = TASK2ID[t]
    val_t = val_tokenized.filter(lambda ex: int(ex["task_id"]) == int(tid))
    if len(val_t) == 0:
        print(f"[Warning] Task {t} has no validation samples, skipping branch selection.")
        continue

    best_h, best_loss = 0, float("inf")
    for h in range(H):
        task_manager.set_task(t, head=h)
        metrics = trainer.evaluate(eval_dataset=val_t)
        loss = metrics.get("eval_loss", 1e9)
        print(f"[Branch Selection] Shared A + independent B, task={t}, head={h}, eval_loss={loss:.4f}")
        if loss < best_loss:
            best_loss, best_h = loss, h

    best_head[t] = best_h
    task_manager.set_task(t, head=best_h)
    export_dir = os.path.join(OUTPUT_DIR, f"{t}")
    
    # Export shared A and optimal B branch
    adapter_name = f"{t}_B#{best_h}"
    model.save_pretrained(export_dir, selected_adapters=["default", adapter_name])
    # Also save tokenizer files to task directory
    tokenizer.save_pretrained(export_dir)
    print(f"✅ Exported task {t}'s shared A + optimal B branch head={best_h} to: {export_dir}")

# Save complete model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training completed (array-only + shared A + independent B (H={H}), A matrix always training, LR managed by cosine + warmup).")
