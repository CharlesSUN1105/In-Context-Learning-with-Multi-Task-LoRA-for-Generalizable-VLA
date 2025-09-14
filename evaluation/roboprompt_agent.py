from typing import List, Optional
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
from form_icl_demonstrations import create_task_handler, SYSTEM_PROMPT
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, CAMERAS
from openai import OpenAI

# =========================
# Multi-task → LoRA Name Routing Table
# =========================
# Key: Task name in your project (consistent with self.task_name)
# Value: Name registered by --lora-modules when starting vLLM (i.e., id in /v1/models list)
TASK_TO_ADAPTER = {
    "put_money_in_safe": "put_money_adapter",
    "open_drawer": "drawer_adapter",
    "light_bulb_in": "light_bulb_adapter",
    "place_wine_at_rack_location": "place_wine_adapter",
    "stack_cups": "stack_cups_adapter",
    "put_groceries_in_cupboard": "put_groceries_adapter",
    "stack_blocks": "stack_blocks_adapter",
    "close_jar": "close_jar_adapter"
}
# Base model id (i.e., --served-model-name)
BASE_MODEL_NAME = "Qwen3-8B"


def extract_first_json_array(text: str) -> Optional[str]:
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()

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


def openai_call(model_name: str, messages):
    """
    Call vLLM through OpenAI compatible interface.
    Key change: model uses the passed model_name (can be base or a LoRA's name).
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8001/v1"),
    )

    resp = client.chat.completions.create(
        model=model_name,            #  No longer hardcoded as 'adapter', but routed by task
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=512,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return resp.choices[0].message.content.strip()


def huggingface_call(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


class RoboPromptAgent(Agent):
    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.model_config = model_config
        # Complete initialization to prevent access before reset
        self.actions: List[np.ndarray] = []
        self.step = 0
        self.savedir = "outputs"

    def _preprocess(self, obs, step, **kwargs):
        rgb_dict = {}
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}
        for camera in CAMERAS:
            rgb_img = obs[f'{camera}_rgb']
            rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)

            rgb_dict[camera] = rgb_img

            img = Image.fromarray(rgb_img)
            rgb_dir = os.path.join(self.savedir, 'rgb_dir', camera, str(self.episode_id))
            os.makedirs(rgb_dir, exist_ok=True)
            img.save(os.path.join(rgb_dir, f'{self.step}.png'))

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

            mask = obs[f'{camera}_mask']
            mask = mask.squeeze().cpu().numpy()

            mask_dict[camera] = mask

            mask_dir = os.path.join(self.savedir, 'input_masks', camera, str(self.episode_id))
            os.makedirs(mask_dir, exist_ok=True)
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil.save(os.path.join(mask_dir, f'{self.step}.png'))

            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud

        if len(self.actions) == 0:
            user_prompt = self.handler.get_user_prompt(mask_dict, mask_id_to_sim_name, point_cloud_dict)

            print(SYSTEM_PROMPT)
            print()
            print(user_prompt)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            output_text = self.llm_call(messages)

            first_array = extract_first_json_array(output_text)
            if first_array is not None:
                output_text = first_array

            print(f"Prediction:", output_text)
            return output_text

    def _postprocess(self, output_text):
        try:
            payload = extract_first_json_array(output_text) or output_text
            actions = np.array(json.loads(payload))
        except Exception as e:
            actions = [[57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            print(e)
            print('Error when parsing actions')

        if len(np.array(actions).shape) == 1:
            actions = [actions]

        output = []
        for action in actions:
            if len(action) != 7:
                action = [57, 49, 87, 0, 39, 0, 1]
            trans_indicies = np.array(action[:3])
            rot_and_grip_indicies = np.array(action[3:6])
            is_gripper_open = action[6]

            bounds = SCENE_BOUNDS
            res = (bounds[3:] - bounds[:3]) / 100
            attention_coordinate = bounds[:3] + res * trans_indicies + res / 2
            quat = discrete_euler_to_quaternion(rot_and_grip_indicies)

            continuous_action = np.concatenate([
                attention_coordinate,
                quat,
                [is_gripper_open],
                [1],
            ])
            output.append(continuous_action)

        return output[:26]

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        output_text = self._preprocess(observation, step, **kwargs)
        if len(self.actions) == 0 and output_text is not None:
            output = self._postprocess(output_text)
            self.actions = output

        continuous_action = self.actions.pop(0)

        self.step += 1

        copy_obs = {k: v.cpu() for k, v in observation.items()}

        return ActResult(continuous_action,
                         observation_elements=copy_obs,
                         info=None)

    def act_summaries(self) -> List[Summary]:
        return []

    def reset(self):
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.actions = []

    def load_weights(self, savedir: str):
        self.savedir = savedir

        self.handler = create_task_handler(self.task_name)

        # ========= Routing Decision =========
        # Priority: map by task name to LoRA; fallback to name in config if not found; use base if still none
        fallback_name = getattr(self.model_config, "name", None) or BASE_MODEL_NAME
        model_name = TASK_TO_ADAPTER.get(self.task_name, fallback_name)

        if self.model_config.llm_call_style == "openai":
            # Bind to task-routed openai_call
            self.llm_call = lambda messages: openai_call(model_name, messages)

        elif self.model_config.llm_call_style == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("loading model from huggingface")
            model = AutoModelForCausalLM.from_pretrained(
                fallback_name,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            self.llm_call = lambda messages: huggingface_call(model, tokenizer, messages)
        return

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}

    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return