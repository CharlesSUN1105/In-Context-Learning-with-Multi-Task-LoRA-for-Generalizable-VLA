import os
os.environ["TRANSFORMERS_NO_MLX"] = "1"
import transformers.utils.generic as _gen
_gen.is_mlx_available = lambda *args, **kwargs: False
if hasattr(_gen, "_is_mlx_available"):
    _gen._is_mlx_available = lambda *args, **kwargs: False

import re
import math
import swanlab
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback
from typing import Optional
import json

MODEL_DIR    = "/home/s84414554/qwen3_finetune/checkpoints/Qwen3-8B"
TRAIN_FILE   = "/home/s84414554/qwen3_finetune/roboprompt-data/stack_cup.jsonl"
VALID_FILE   = "/home/s84414554/qwen3_finetune/roboprompt-data/stack_cup_eval.jsonl"
OUTPUT_DIR   = "/home/s84414554/qwen3_finetune/roboprompt-data/output/stack_cup_10shot"
MAX_LENGTH   = 3600
PROJECT_NAME = "stack_cup_10shot"

os.environ["SWANLAB_PROJECT"] = PROJECT_NAME
swanlab.config.update({
    "model": MODEL_DIR,
    "data_max_length": MAX_LENGTH,
})

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
    local_files_only=True
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(base_model, lora_cfg)

def strip_code_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^\s*Prediction\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    return t

def extract_first_json_array(text: str) -> Optional[str]:
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

train_ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
val_ds   = load_dataset("json", data_files={"validation": VALID_FILE})["validation"]

def _get_messages(example):
    for k in ("messages", "conversations", "chat"):
        if k in example and isinstance(example[k], list):
            return example[k]
    raise ValueError("Sample missing messages/conversations/chat field, or its type is not list.")

def preprocess(ex):
    msgs = _get_messages(ex)
    last_asst_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None:
        return {"input_ids": [], "attention_mask": [], "labels":[]}
    prompt_messages = msgs[:last_asst_idx]
    target_text = msgs[last_asst_idx].get("content", "")
    if not isinstance(target_text, str):
        target_text = str(target_text)
    array_payload = extract_first_json_array(target_text)
    if not array_payload:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True
    )
    target_ids = tokenizer(array_payload, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    max_prompt_len = MAX_LENGTH - len(target_ids)
    if max_prompt_len <= 0:
        input_ids = target_ids[-MAX_LENGTH:]
        labels    = input_ids.copy()
    else:
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
        input_ids = prompt_ids + target_ids
        labels    = ([-100] * len(prompt_ids)) + target_ids
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if len(input_ids) < MAX_LENGTH:
        pad_len   = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels    = labels    + [-100]   * pad_len
    attention_mask = [0 if tok == pad_id else 1 for tok in input_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

remove_cols_train = [c for c in train_ds.column_names if c not in ("messages", "conversations", "chat")]
remove_cols_val   = [c for c in val_ds.column_names   if c not in ("messages", "conversations", "chat")]

train_tokenized = train_ds.map(
    preprocess,
    remove_columns=remove_cols_train,
    desc="Tokenizing train (chat format, first-array-only)"
)
val_tokenized = val_ds.map(
    preprocess,
    remove_columns=remove_cols_val,
    desc="Tokenizing valid (chat format, first-array-only)"
)

train_tokenized = train_tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)
val_tokenized   = val_tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)

# ==== Data validation logic (optimized version) ====
print("===== Data Validation =====")

def decode_target(ex):
    """Only decode target part in labels, skip special tokens"""
    target_ids = [tid for tid in ex["labels"] if tid != -100]
    return tokenizer.decode(target_ids, skip_special_tokens=True)

# Print first 3 targets
for i in range(min(3, len(train_tokenized))):
    text = decode_target(train_tokenized[i])
    print(f"Sample {i} target decoding:")
    print(text)
    print("-" * 40)

# Count samples ending with eos
count_total = len(train_tokenized)
count_ok = 0
bad_ids = []
for idx, ex in enumerate(train_tokenized):
    target_text = decode_target(ex)
    if target_text.strip().endswith("]"):  # Use ] to judge end, because we skip_special_tokens=True
        count_ok += 1
    try:
        arr = json.loads(target_text)
        if not isinstance(arr, list) or not all(isinstance(a, list) and len(a) == 7 for a in arr):
            bad_ids.append(idx)
    except json.JSONDecodeError:
        bad_ids.append(idx)

print(f"Samples ending with ']': {count_ok}/{count_total}")
print(f"Invalid array samples: {len(bad_ids)} -> {bad_ids[:10]}")
print("====================")


# ==== Training section ====
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt",
)

swanlab_callback = SwanLabCallback(
    project=PROJECT_NAME,
    experiment_name="put_grocery_10shot"
)
earlystop_callback = EarlyStoppingCallback(
    early_stopping_patience=2
)

num_epochs = 3
per_device_bs = 2
grad_accum = 4
train_samples = len(train_tokenized)
steps_per_epoch = math.ceil(train_samples / (per_device_bs * grad_accum))
EVAL_STEPS = max(1, steps_per_epoch // 3)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=5e-5,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    do_eval=True,
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    logging_steps=10,
    disable_tqdm=True,
    report_to=["swanlab"],
    run_name="put_money_in_safe_10shot",
)

from transformers.trainer_utils import IntervalStrategy
training_args.eval_strategy = IntervalStrategy.STEPS

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    callbacks=[swanlab_callback, earlystop_callback]
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training completed (supervised first JSON array + EOS only), LoRA weights saved to", OUTPUT_DIR)
