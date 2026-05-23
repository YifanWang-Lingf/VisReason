import json

from .adapter import DatasetAdapter
from .schema import ModelResponse


class DummyInferer:
    def run(self, dataset: DatasetAdapter):
        for idx in range(len(dataset)):
            item = dataset[idx]
            if idx == 0:
                with open("debug_input.json", "w", encoding="utf-8") as f:
                    json.dump(item.get_openai_messages(), f, indent=4)
            dataset.handle_output(
                ModelResponse(
                    idx=idx,
                    response=json.dumps(item.get_openai_messages(), indent=4),
                )
            )
