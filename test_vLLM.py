import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

DATASETS_PATH = Path("./data/datasets.json")


def encode64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_data_url(image_path: str) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    return f"data:{mime_type};base64,{encode64_image(image_path)}"


def load_data_records(data_path: str) -> List[Dict[str, Any]]:
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


def resolve_image_path(image_base_path: str, image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute() or not image_base_path:
        return str(path)
    return str(Path(image_base_path) / path)


def build_question_prompt(q_type: str, cot: bool, data_path: str, idx: int) -> str:
    if q_type == "1":
        q_prompt = (
            'Please answer the question from the given choices and put your final answer in one "\\boxed{}". '
            "There may be more than one correct option; please fill in all the options you consider correct in the \\boxed{}.\n"
        )
    elif q_type == "2":
        q_prompt = 'Please answer the question using a few words or phrases and put your final answer in one "\\boxed{}".\n'
    elif q_type == "3":
        q_prompt = 'Please answer the question and summarize your answer concisely in one "\\boxed{}".\n'
    elif q_type == "4":
        q_prompt = (
            "First determine the required answer targets according to the task description, and then output bounding boxes only for these targets. \n"
            "Each bounding box must tightly cover exactly one answer target; do not include multiple objects or large regions in a single box. \n"
            "You must output exactly the number of bounding boxes specified in the question, no more and no fewer. \n"
            'Return a single array of bounding boxes in one "\\boxed{}". Each bbox must be in the format [x1, y1, x2, y2], '
            "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner; different bboxes are separated by semicolons (;).\n"
        )
    else:
        raise NotImplementedError(f"Unknown question type: {q_type}\nfrom {data_path}-{idx}")

    if cot:
        return q_prompt + "You must think step by step \n"

    return q_prompt + "You must output only the final answer. Do not show any reasoning process or explanation.\n"


class DatasetLoader:
    def __init__(self, json_path: str, image_base_path: str = "", cot: bool = False):
        self.image_base_path = image_base_path
        self.json_path = json_path
        self.cot = cot
        self.raw = load_data_records(json_path)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        item = self.raw[idx]

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

        q_prompt = build_question_prompt(q_type, self.cot, self.json_path, idx)
        user_content = [
            {
                "type": "text",
                "text": f"Question:\n{question_text}",
            }
        ]

        for img in question_images:
            image_path = resolve_image_path(self.image_base_path, img)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image_path)},
                }
            )

        messages = [
            {
                "role": "system",
                "content": "You are a highly intelligent question answering assistant. " + q_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        return messages, item


def build_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def get_model_response(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_retries: int = 2,
    retry_sleep: float = 2.0,
) -> Optional[str]:
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            retries += 1
            print(f"Error: {e}, retrying {retries}/{max_retries}")
            time.sleep(retry_sleep)
    return None


def save_results(output_file: Path, results: List[Dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def run_dataset(
    client: OpenAI,
    model_id: str,
    dataset_name: str,
    paths: Dict[str, str],
    save_dir: Path,
    cot: bool,
) -> None:
    print(f"\nLoading dataset: {dataset_name}")
    json_path = paths["json_path"]
    image_path = paths.get("image_path", "")

    if not os.path.exists(json_path):
        print(f"Skipping {dataset_name}: {json_path} not found.")
        return

    dataset = DatasetLoader(json_path=json_path, image_base_path=image_path, cot=cot)
    print(f"Dataset {dataset_name} loaded successfully, total samples: {len(dataset)}")

    results = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}"):
        messages, item = dataset[i]
        response = get_model_response(
            client=client,
            model_name=model_id,
            messages=messages,
        )

        record = dict(item)
        record["response"] = response
        results.append(record)

    output_file = save_dir / f"{Path(json_path).stem}_results.json"
    save_results(output_file, results)
    print(f"{dataset_name} inference completed, results saved to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="Served model name to call.")
    parser.add_argument(
        "prompt_mode",
        nargs="?",
        default="nocot",
        choices=("cot", "nocot"),
        help="Prompt mode: cot or nocot. Defaults to nocot.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8008)

    args = parser.parse_args()
    args.cot = args.prompt_mode == "cot"
    return args


def main() -> None:
    args = parse_args()
    data_map = load_data_map()

    openai_api_base = build_base_url(args.host, args.port)
    client = OpenAI(
        api_key="EMPTY",
        base_url=openai_api_base,
    )

    model_id = args.model
    save_name = f"{model_id}_cot" if args.cot else model_id
    model_output_path = Path("results") / save_name
    model_output_path.mkdir(parents=True, exist_ok=True)

    print(f"Using vLLM server: {openai_api_base}")
    print(f"Using model: {model_id}")

    for dataset_name, paths in data_map.items():
        run_dataset(
            client=client,
            model_id=model_id,
            dataset_name=dataset_name,
            paths=paths,
            save_dir=model_output_path,
            cot=args.cot,
        )

    print(f"\nAll inference completed. Base URL: {openai_api_base} | Model: {model_id}")
    print(f"Results folder: {model_output_path}")


if __name__ == "__main__":
    main()


"""
Run vLLM server:

CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --trust-remote-code \
  --port 8008 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.8 \
  --served-model-name <model>

Example usage:

# Default prompt mode is nocot. Results are saved to results/<model>.
python test_vLLM.py <model>

# Remote vLLM server. Pass host and port, but keep results under results/<model>.
python test_vLLM.py <model> --host <server_ip> --port 8008
python test_vLLM.py <model> nocot --host <server_ip> --port 8008

# CoT output folder adds _cot to the model name.
python test_vLLM.py <model> cot --host <server_ip> --port 8008
"""
