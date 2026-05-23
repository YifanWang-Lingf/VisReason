import re
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.datainfer import OpenAIInferer
from utils.datainfer import DatasetAdapter, ModelResponse
from utils import bbox_utils


DATASETS_PATH = Path("./data/datasets.json")
BBOX_JUDGE_CLASSES = {"class_1", "class_2"}
BBOX_IOU_THRESHOLD = 0.5

# Fill these only for local debugging!
DEBUG_API_KEY = ""
DEBUG_BASE_URL = ""

template = """
You are a strict evaluator assessing answer correctness. You must output {positive} for fully correct answers and {negative} for any other case.

# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score {positive} if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score {positive} if the prediction matches the answer semantically, it can be in different format.
  * Score {negative} for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
{positive} or {negative}
"""


def get_resp_text(response):
    # datainfer
    d = response.model_dump() if hasattr(response, "model_dump") else {}

    v = d.get("response")
    if isinstance(v, str) and v.strip():
        return v

    for k in ("text", "content", "output_text"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v

    choices = d.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict):
                v = msg.get("content")
                if isinstance(v, str) and v.strip():
                    return v

    v = getattr(response, "text", None)
    if isinstance(v, str) and v.strip():
        return v

    return str(response)

def extract_ans(response) -> str:
    if not isinstance(response, str):
        return ""

    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    match = re.search(pattern, response)

    if match:
        return match.group(1)

    return ""


def read_result_records(data_path: str) -> List[Dict]:
    if data_path.endswith("_results.jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if data_path.endswith("_results.json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported results JSON structure in {data_path}; expected a list.")

    raise ValueError(
        f"Unsupported results file: {data_path}. "
        f"Expect *_results.jsonl or *_results.json"
    )


def get_judge_output_path(data_path: str) -> str:
    if data_path.endswith("_results.jsonl"):
        return data_path.replace("_results.jsonl", "_judge.json")
    if data_path.endswith("_results.json"):
        return data_path.replace("_results.json", "_judge.json")
    return data_path + "_judge.json"


def ordered_box_ious(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> List[float]:
    box_ious = []
    for idx, gt_box in enumerate(gt_boxes):
        if idx < len(pred_boxes):
            box_ious.append(bbox_utils.iou(gt_box, pred_boxes[idx]))
        else:
            box_ious.append(0.0)
    return box_ious


def _normalize_boxes(raw_boxes) -> List[List[float]]:
    boxes = []
    if not isinstance(raw_boxes, list):
        return boxes

    for box in raw_boxes:
        if isinstance(box, list) and len(box) == 4:
            try:
                boxes.append([float(v) for v in box])
            except (TypeError, ValueError):
                pass

    return boxes


def _image_size(item: Dict) -> Tuple[Optional[float], Optional[float]]:
    width = item.get("Width", item.get("Weight"))
    height = item.get("Height")

    try:
        img_w = float(width) if width is not None else None
    except (TypeError, ValueError):
        img_w = None

    try:
        img_h = float(height) if height is not None else None
    except (TypeError, ValueError):
        img_h = None

    return img_w, img_h


def judge_bbox_results(data_path: str, iou_threshold: float = BBOX_IOU_THRESHOLD):
    data = [item for item in read_result_records(data_path) if item]
    results = []

    total_gt = 0
    total_pred = 0
    total_tp = 0
    num_questions_with_gt = 0
    sum_acc_avg = 0.0
    num_parse_fail = 0
    num_clipped_all = 0

    for idx, item in enumerate(data):
        answer = item.get("answer") or {}
        gt_boxes = _normalize_boxes(answer.get("text") if isinstance(answer, dict) else answer)
        img_w, img_h = _image_size(item)

        response = item.get("response", "")
        raw_pred_quads = bbox_utils.parse_pred_quads(response if isinstance(response, str) else "")
        pred_boxes = []
        for quad in raw_pred_quads:
            box = bbox_utils.convert_quad_to_box(quad, img_w, img_h)
            if box is not None:
                pred_boxes.append(box)

        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)
        total_gt += num_gt
        total_pred += num_pred

        if num_gt > 0:
            num_questions_with_gt += 1

        if not raw_pred_quads:
            num_parse_fail += 1
        elif not pred_boxes:
            num_clipped_all += 1

        box_ious = ordered_box_ious(gt_boxes, pred_boxes)
        matched_gt_count = sum(1 for value in box_ious if value >= iou_threshold)
        total_tp += matched_gt_count

        if num_gt > 0:
            acc_avg = matched_gt_count / num_gt
            sum_acc_avg += acc_avg
        else:
            acc_avg = 0.0

        results.append(item | {"judge": box_ious})

    recall = total_tp / total_gt if total_gt > 0 else 0.0
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    acc_avg_dataset = sum_acc_avg / num_questions_with_gt if num_questions_with_gt else 0.0

    summary = {
        "IoU": iou_threshold,
        "Total": total_gt,
        "Pred": total_pred,
        "TP": total_tp,
        "Recall": recall,
        "Precision": precision,
        "ACC-AVG": acc_avg_dataset,
        "Accuracy": acc_avg_dataset,
        "NumSamples": len(data),
        "NumQuestionsWithGT": num_questions_with_gt,
        "NumParseFailNoBbox": num_parse_fail,
        "NumParsedButClippedAll": num_clipped_all,
    }
    results.append({"summary": summary})

    out_path = get_judge_output_path(data_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Processing Complete: {data_path}")
    print(f"BBox IoU@{iou_threshold}: Accuracy={acc_avg_dataset:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    print(f"Judge results saved to: {out_path}")

    return acc_avg_dataset


def load_data_map(config_path: Path = DATASETS_PATH) -> Dict[str, Dict[str, str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data_map = json.load(f)

    if not isinstance(data_map, dict):
        raise ValueError(f"Unsupported dataset index structure in {config_path}; expected an object.")

    return data_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Judge model name to call.")
    parser.add_argument("eval_path", help="Directory containing *_results.json files to judge.")
    return parser.parse_args()


def get_api_config():
    api_key = DEBUG_API_KEY or os.getenv("API_KEY")
    base_url = DEBUG_BASE_URL or os.getenv("BASE_URL")

    if not api_key:
        raise ValueError("Missing API key. Set API_KEY.")
    if not base_url:
        raise ValueError("Missing base URL. Set BASE_URL.")

    return api_key, base_url


class JudgeAdapter(DatasetAdapter):
    def __init__(self, data_path: str):
        self.data_path = data_path

        if data_path.endswith("_results.jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f if line.strip()]
        elif data_path.endswith("_results.json"):
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported results file: {data_path}. "
                f"Expect *_results.jsonl or *_results.json"
            )

        self.data = [item for item in self.data if item]

        self.results = [None] * len(self.data)
        self.correct = 0
        self.total_tokens = 0
        self.total_completion_tokens = 0

        self.force_false = [False] * len(self.data)

    def __len__(self):
        return len(self.data)

    def prepare_input(self, idx: int) -> List[Dict]:
        item = self.data[idx]

        question_raw = item.get("question", "")
        if isinstance(question_raw, dict):
            question = question_raw.get("text", "")
        else:
            question = str(question_raw)

        answer_raw = item.get("answer", "")
        if isinstance(answer_raw, dict):
            answer = answer_raw.get("text", "")
        else:
            answer = str(answer_raw)

        prediction_raw = item.get("response", None)

        positive = "true"
        negative = "false"

        if not isinstance(prediction_raw, str):
            print("Non-str response at idx =", idx, "type =", type(prediction_raw), "value =", prediction_raw)

        prediction = extract_ans(prediction_raw)

        if not answer or not isinstance(prediction, str) or not prediction.strip():
            self.force_false[idx] = True

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": template.format(
                            question=question,
                            answer=answer,
                            prediction=prediction,
                            positive=positive,
                            negative=negative,
                        )
                    }
                ]
            }
        ]

    @staticmethod
    def _safe_get_text(response: ModelResponse) -> str:
        txt = getattr(response, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt

        md = {}
        try:
            md = response.model_dump()
        except Exception:
            md = {}

        for k in ("text", "response", "content", "output_text"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v

        try:
            choices = md.get("choices") or []
            if choices:
                msg = (choices[0].get("message") or {})
                v = msg.get("content")
                if isinstance(v, str):
                    return v
        except Exception:
            pass

        return ""

    def handle_output(self, response: ModelResponse):
        idx = response.idx

        judge_text = self._safe_get_text(response)
        usage = None
        try:
            usage = response.model_dump().get("usage")
        except Exception:
            usage = None

        if self.force_false[idx]:
            judge_text = "false"

        self.results[idx] = self.data[idx] | {"judge": judge_text, "usage": usage}

        if getattr(response, "usage", None):
            try:
                self.total_tokens += response.usage.total_tokens
                self.total_completion_tokens += response.usage.completion_tokens
            except Exception:
                pass

        jt = judge_text.strip().lower()
        if "true" in jt and not self.force_false[idx]:
            self.correct += 1
        elif "false" not in jt and jt:
            print(f"{idx} unknow judge: {judge_text.strip()}")

    def finish(self):
        print(f"Processing Complete：{self.data_path}")
        print(f"✓ Accuracy: {self.correct}/{len(self)}")

        if self.total_tokens > 0:
            print(
                f"Total tokens used: {self.total_tokens}, "
                f"the number of completion tokens: {self.total_completion_tokens}"
            )

        if self.data_path.endswith("_results.jsonl"):
            out_path = self.data_path.replace("_results.jsonl", "_judge.json")
        elif self.data_path.endswith("_results.json"):
            out_path = self.data_path.replace("_results.json", "_judge.json")
        else:
            out_path = self.data_path + "_judge.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

        print(f"✅ Judge results saved to: {out_path}")





if __name__ == "__main__":

    ####========================= API inference ========================####

    args = parse_args()
    data_map = load_data_map()
    inferencer = None
    class_accuracy = {}

    for name in data_map.keys():
        result_path = os.path.join(args.eval_path, f'{name}_results.json')
        if name in BBOX_JUDGE_CLASSES:
            class_accuracy[name] = judge_bbox_results(result_path, BBOX_IOU_THRESHOLD)
            continue

        if inferencer is None:
            api_key, base_url = get_api_config()
            inferencer = OpenAIInferer(
                model=args.model,
                max_concurrency=16,
                api_key=api_key,
                base_url=base_url,
            )

        with JudgeAdapter(result_path) as dataset:
            inferencer.run(dataset)
            class_accuracy[name] = dataset.correct / len(dataset) if len(dataset) > 0 else 0.0

    summary_path = os.path.join(args.eval_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(class_accuracy, f, ensure_ascii=False, indent=4)

    print(f"Summary saved to: {summary_path}")
    
    # python judge.py 2>&1 | tee -a ./run/gemini-3-pro-preview/judge.log
    # python judge.py 2>&1 | tee -a ./run/human1-300-Acc_nocot/judge.log
    # python judge.py 2>&1 | tee -a ./run/gpt-4o/judge.log
    # python judge.py 2>&1 | tee -a ./run/gpt-4o_nocot/judge.log
    # python judge.py 2>&1 | tee -a ./run-mask/qwen3-vl-30b-a3b-thinking-015-VisReason/judge.log 
    # python judge.py 2>&1 | tee -a ./run-mask/qwen3-vl-30b-a3b-thinking-015-MMMU/judge.log 
    # python judge.py qwen2.5-vl-3b-instruct ./results/qwen2.5-vl-3b-instruct
