# Can MLLMs Reason Beyond Language? VisReason: A Comprehensive Benchmark for Vision-Centric Reasoning

This repository provides the official code and evaluation tools for VisReason, including dataset indexing, API and vLLM inference scripts, automatic judging, and result summarization. (ACL 2026 Findings)

## 📢 News


## ✅ TODO

- Release the full dataset on Hugging Face.
- Add the paper link.
- Add citation information.
- Add license information.

<sub>This repository hosts the latest testing version. For the stable official release, please visit [CASIA-IVA-Lab/VisReason](https://github.com/CASIA-IVA-Lab/VisReason).</sub>

## 👀 Overview

VisReason is a benchmark for evaluating vision-centric reasoning in everyday scenarios where perception and inference are tightly coupled. Unlike many STEM-oriented or knowledge-intensive visual reasoning benchmarks, VisReason is designed to test whether MLLMs can reason directly from visual evidence rather than relying mainly on language-mediated abstractions.

VisReason contains 1,505 carefully curated questions across 10 reasoning categories, covering perceptual, structural, and conceptual reasoning. The tasks include visual difference identification, 3D-spatial reasoning, game-board reasoning, and implicit rule inference from visual cues.

Our evaluation shows that VisReason poses a qualitatively different challenge from existing benchmarks, exposing large gaps between humans and current MLLMs and revealing that test-time reasoning strategies such as explicit CoT prompting provide limited gains without stronger visual grounding.

## 📦 Dataset

The full dataset description, field definitions, and download instructions will be provided on Hugging Face.

Hugging Face:

## 🗂️ Repository Structure

```text
VisReason/
  data/
    img_<class_number>/
      datajson_label.<ext>
    ...
    class_1.jsonl
    ...
    datasets.json
  results/
    <model>/
      class_<class_number>_results.json
      class_<class_number>_judge.json
      summary.json
  utils/
    datainfer/
    bbox_utils.py
  test_api_batch.py
  test_vLLM.py
  judge.py
  README.md
```

## 🚀 Quick Start

### ⚙️ Installation

```bash
pip install openai tqdm pydantic datasets rich pillow
```

For local vLLM inference, **install and run vLLM separately** in the environment that serves the model.

### 🧪 Inference

#### **Option A: 🌐 API Inference**

`test_api_batch.py` calls an OpenAI-compatible API endpoint. Set your credentials through environment variables:

```bash
export API_KEY="xxx"
export BASE_URL="xxx"
```

Run inference:

```bash
# thinking models
python test_api_batch.py <model> 
# instruct models
python test_api_batch.py <model> cot
python test_api_batch.py <model> nocot
```

The default prompt mode is `nocot`.

#### **Option B: 🖥️ vLLM Inference**

Start a vLLM OpenAI-compatible server, for example:

```bash
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
```

Run inference:

```bash
# thinking models
python test_vLLM.py <model> --host localhost --port 8008
# instruct models
python test_vLLM.py <model> cot --host localhost --port 8008
python test_vLLM.py <model> nocot --host localhost --port 8008
```

The default prompt mode is `nocot`.

#### **Option C: 🧩 lmms-eval Inference**

Coming soon.


### 📁 Output Format

Inference results are saved under:

```text
results/<model>/class_X_results.json
results/<model>_cot/class_X_results.json
```

Each result file is a JSON list. Each item keeps the original sample fields and adds:

```json
{
  "response": "model output"
}
```

### 🧑‍⚖️ Evaluation

Run judging:

```bash
export API_KEY="xxx"
export BASE_URL="xxx"

python judge.py <judge_model> results/<model>
```

The evaluator writes:

```text
results/<model>/class_X_judge.json
results/<model>/summary.json
```

For `class_1` and `class_2`, bounding-box predictions are evaluated with IoU at threshold 0.5. Other classes are evaluated by the judge model.

## 🖼️ Demos

### Option A: 🌐 API Inference (o4-mini)

Set API credentials:

```bash
export API_KEY="xxx"
export BASE_URL="xxx"
```

Run inference:

```bash
python test_api_batch.py o4-mini
```

The generated results will be saved to:

```text
results/o4-mini/class_1_results.json
results/o4-mini/class_2_results.json
...
results/o4-mini/class_10_results.json
```

Run evaluation:

```bash
python judge.py gpt-5 results/o4-mini
```

The judged outputs and final summary will be saved to:

```text
results/o4-mini/class_1_judge.json
results/o4-mini/class_2_judge.json
...
results/o4-mini/summary.json
```

### Option B: 🖥️ vLLM Inference (Qwen3-VL-235B-A22B-Instruct)

Start the vLLM server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
  --model ../hf_models/Qwen3-VL-235B-A22B-Instruct \
  --trust-remote-code \
  --port 8008 \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --served-model-name Qwen3-VL-235B-A22B-Instruct
```

Run inference:

```bash
python test_vLLM.py Qwen3-VL-235B-A22B-Instruct cot --host 12x.xx.xx.xx --port 8008
```

The generated results will be saved to:

```text
results/Qwen3-VL-235B-A22B-Instruct_cot/class_1_results.json
results/Qwen3-VL-235B-A22B-Instruct_cot/class_2_results.json
...
results/Qwen3-VL-235B-A22B-Instruct_cot/class_10_results.json
```

Run evaluation:

```bash
python judge.py gpt-5 results/Qwen3-VL-235B-A22B-Instruct_cot
```

The judged outputs and final summary will be saved to:

```text
results/Qwen3-VL-235B-A22B-Instruct_cot/class_1_judge.json
results/Qwen3-VL-235B-A22B-Instruct_cot/class_2_judge.json
...
results/Qwen3-VL-235B-A22B-Instruct_cot/summary.json
```

## 📊 Metrics

VisReason reports accuracy at the class level and uses the average class accuracy as the final score.

For `class_1` and `class_2`, predictions are evaluated as bounding-box localization tasks. The evaluator parses the predicted boxes from the model response, matches them with the ground-truth boxes, and counts a box as correct when its IoU is at least 0.5.

For the remaining classes, predictions are judged by an LLM-based evaluator. The judge compares the model response with the ground-truth answer and outputs whether the prediction is correct.

For each class, accuracy is computed as:

```text
accuracy = number of correct samples / number of evaluated samples
```

The final score is the unweighted mean over all class accuracies:

```text
average accuracy = mean(class_1_acc, class_2_acc, ..., class_10_acc)
```

The evaluator writes per-class scores and the final average score to:

```text
results/<model>/summary.json
```

## 🏆 Results




## 📝 Citation

```bibtex

```

## 📄 License
TODO

## 📬 Contact
If you have any questions, please reach out to:

*   Yifan Wang - wangyifan2026@ia.ac.cn
