import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, List

from utils.datainfer import DatasetAdapter, ModelResponse, OpenAIInferer

DATASETS_PATH = Path("./data/datasets.json")

# Fill these only for local debugging!
DEBUG_API_KEY = ""
DEBUG_BASE_URL = ""

def encode64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_data_records(data_path: str) -> List[Dict]:
    path = Path(data_path)

    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL in {data_path} at line {line_no}: {e}") from e
        return records

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        raise ValueError(f"Unsupported JSON structure in {data_path}; expected a list of records.")

    raise ValueError(f"Unsupported data file format: {data_path}")


def load_data_map(config_path: Path = DATASETS_PATH) -> Dict[str, Dict[str, str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data_map = json.load(f)

    if not isinstance(data_map, dict):
        raise ValueError(f"Unsupported dataset index structure in {config_path}; expected an object.")

    for name, item in data_map.items():
        if not isinstance(item, dict) or "json_path" not in item:
            raise ValueError(f"Invalid dataset config for {name} in {config_path}.")
        item.setdefault("image_path", "")

    return data_map


class MyDatasetAdapter(DatasetAdapter):
    def __init__(self, data_path: str, image_path: str = "", save_dir: str = "", cot: bool = False):
        self.data = load_data_records(data_path)
        self.data_path = data_path
        self.image_path = image_path
        self.save_dir = save_dir
        self.cot = cot

        self.results = [None] * len(self.data)
        self.total_tokens = 0
        self.total_completion_tokens = 0

    def __len__(self):
        return len(self.data)

    def prepare_input(self, idx: int) -> List[Dict]:
        item = self.data[idx]

        q_type = item.get("question type")
        question = item.get("question", {})

        if isinstance(question, dict):
            question_text = question.get("text", "")
            question_images = question.get("images") or []
        else:
            question_text = str(question)
            question_images = item.get("images") or []

        if isinstance(question_images, str):
            question_images = [question_images]

        # CoT/NoCoT only affects instruct-model prompts.
        if self.cot:
            if q_type == "1":  # multiple-choice
                q_prompt = (
                    'Please answer the question from the given choices and put your final answer in one "\\boxed{}". '
                    'There may be more than one correct option; please fill in all the options you consider correct in the \\boxed{}.\n'
                    "You must think step by step \n"
                )
            elif q_type == "2":  # fill in the blank
                q_prompt = (
                    'Please answer the question using a few words or phrases and put your final answer in one "\\boxed{}".\n'
                    "You must think step by step \n"
                )
            elif q_type == "3":  # open-ended
                q_prompt = (
                    'Please answer the question and summarize your answer concisely in one "\\boxed{}".\n'
                    "You must think step by step \n"
                )
            elif q_type == "4":  # bounding box
                q_prompt = (
                    "First determine the required answer targets according to the task description, and then output bounding boxes only for these targets. \n"
                    "Each bounding box must tightly cover exactly one answer target; do not include multiple objects or large regions in a single box. \n"
                    "You must output exactly the number of bounding boxes specified in the question, no more and no fewer. \n"
                    'Return a single array of bounding boxes in one "\\boxed{}". Each bbox must be in the format [x1, y1, x2, y2], '
                    "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner; different bboxes are separated by semicolons (;).\n"
                    "You must think step by step \n"
                )
            else:
                raise NotImplementedError(f"Unknown question type: {q_type}\nfrom {self.data_path}-{idx}")
        else:
            if q_type == "1":  # multiple-choice
                q_prompt = (
                    'Please answer the question from the given choices and put your final answer in one "\\boxed{}". '
                    'There may be more than one correct option; please fill in all the options you consider correct in the \\boxed{}.\n'
                    "You must output only the final answer. Do not show any reasoning process or explanation.\n"
                )
            elif q_type == "2":  # fill in the blank
                q_prompt = (
                    'Please answer the question using a few words or phrases and put your final answer in one "\\boxed{}".\n'
                    "You must output only the final answer. Do not show any reasoning process or explanation.\n"
                )
            elif q_type == "3":  # open-ended
                q_prompt = (
                    'Please answer the question and summarize your answer concisely in one "\\boxed{}".\n'
                    "You must output only the final answer. Do not show any reasoning process or explanation.\n"
                )
            elif q_type == "4":  # bounding box
                q_prompt = (
                    "First determine the required answer targets according to the task description, and then output bounding boxes only for these targets. \n"
                    "Each bounding box must tightly cover exactly one answer target; do not include multiple objects or large regions in a single box. \n"
                    "You must output exactly the number of bounding boxes specified in the question, no more and no fewer. \n"
                    'Return a single array of bounding boxes in one "\\boxed{}". Each bbox must be in the format [x1, y1, x2, y2], '
                    "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner; different bboxes are separated by semicolons (;).\n"
                    "You must output only the final answer. Do not show any reasoning process or explanation.\n"
                )
            else:
                raise NotImplementedError(f"Unknown question type: {q_type}\nfrom {self.data_path}-{idx}")

        contents = [
            {
                "type": "text",
                "text": f"Question:\n{question_text}",
            }
        ]

        for img in question_images:
            img_path = os.path.join(self.image_path, img)
            contents.append({
                "type": "image",
                "image": img_path,
            })

        return [
            {
                "role": "system",
                "content": "You are a highly intelligent question answering assistant. " + q_prompt,
            },
            {
                "role": "user",
                "content": contents,
            },
        ]

    def handle_output(self, response: ModelResponse):
        idx = response.idx
        self.results[idx] = self.data[idx] | response.model_dump()
        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_completion_tokens += response.usage.completion_tokens

    def finish(self):
        print(f"Processing Complete: {self.data_path}")
        if self.total_tokens > 0:
            print(f"Total tokens used: {self.total_tokens}, the number of completion tokens: {self.total_completion_tokens}")

        os.makedirs(self.save_dir, exist_ok=True)
        new_name = f"{Path(self.data_path).stem}_results.json"
        with open(os.path.join(self.save_dir, new_name), "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name to call.")
    parser.add_argument(
        "prompt_mode",
        nargs="?",
        default="nocot",
        choices=("cot", "nocot"),
        help="Prompt mode: cot or nocot. Defaults to nocot.",
    )
    args = parser.parse_args()
    args.cot = args.prompt_mode == "cot"
    return args


def get_api_config():
    api_key = DEBUG_API_KEY or os.getenv("API_KEY")
    base_url = DEBUG_BASE_URL or os.getenv("BASE_URL")

    if not api_key:
        raise ValueError("Missing API key. Set API_KEY.")
    if not base_url:
        raise ValueError("Missing base URL. Set BASE_URL.")

    return api_key, base_url


def get_model_specific_args(model: str) -> Dict:
    if "gpt-4o" in model:
        return {}
    if "gpt" in model:
        return {
            "reasoning_effort": "medium"
        }
    if "qwen" in model:
        return {
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget": 8192,
            }
        }
    if "gemini-3" in model:
        return {
            "extra_body": {
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingLevel": "low"
                    }
                }
            }
        }
    return {}


if __name__ == "__main__":
    args = parse_args()
    model = args.model
    data_map = load_data_map()
    api_key, base_url = get_api_config()
    generate_kwargs = get_model_specific_args(model)
    save_name = f"{model}_cot" if args.cot else model

    inferencer = OpenAIInferer(
        model=model,
        generate_kwargs=generate_kwargs,
        max_concurrency=6,
        api_key=api_key,
        base_url=base_url,
    )

    for item in data_map.values():
        with MyDatasetAdapter(
            item["json_path"],
            item["image_path"],
            save_dir=f"results/{save_name}",
            cot=args.cot,
        ) as dataset:
            inferencer.run(dataset)
