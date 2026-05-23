from typing import Dict, List

from .schema import ModelInput, ModelResponse


class DatasetAdapter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def __getitem__(self, idx: int) -> ModelInput:
        return ModelInput(
            idx=idx,
            messages=self.prepare_input(idx),
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def prepare_input(self, idx: int) -> List[Dict]:
        raise NotImplementedError

    def handle_output(self, response: ModelResponse) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError
