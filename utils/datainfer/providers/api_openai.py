import asyncio

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from ..adapter import DatasetAdapter
from ..schema import ModelInput, ModelResponse

lock = asyncio.Lock()


class OpenAIInferer(AsyncOpenAI):
    def __init__(
        self,
        model: str,
        generate_kwargs: dict = {},
        max_concurrency: int = 1,
        **kwargs,
    ):
        self.model = model
        self.generate_kwargs = generate_kwargs
        self.max_concurrency = max_concurrency

        if generate_kwargs.pop("stream", False):
            self.run_task = self.stream_task
            if not self.generate_kwargs.get("stream_options"):
                self.generate_kwargs["stream_options"] = {"include_usage": True}
        else:
            self.run_task = self.task

        super().__init__(**kwargs)

    async def task(self, input: ModelInput):
        completion = await self.chat.completions.create(
            model=self.model,
            messages=input.get_openai_messages(),
            **self.generate_kwargs,
        )

        return ModelResponse(
            idx=input.idx,
            response=completion.choices[0].message.content,
            usage=completion.usage,
            finish_reason=completion.choices[0].finish_reason,
        )

    async def stream_task(self, input: ModelInput):
        async with self.chat.completions.stream(
            model=self.model,
            messages=input.get_openai_messages(),
            **self.generate_kwargs,
        ) as stream:
            content_parts = []
            usage = None
            async for event in stream:
                if hasattr(event, "delta"):
                    content_parts.append(event.delta or "")
                elif hasattr(event, "chunk") and getattr(event.chunk, "usage", None):
                    usage = event.chunk.usage

        return ModelResponse(
            idx=input.idx,
            response="".join(content_parts),
            usage=usage,
            finish_reason="stop",
        )

    async def run_async(self, adapter: DatasetAdapter):
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_with_semaphore(idx) -> None:
            async with semaphore:
                try:
                    response = await self.run_task(adapter[idx])
                    async with lock:
                        adapter.handle_output(response)
                except Exception as e:
                    print(f"\nItem {idx} failed: {e}")

        tasks = [process_with_semaphore(i) for i in range(len(adapter))]
        for coro in async_tqdm.as_completed(tasks, total=len(adapter)):
            await coro

    def run(self, loader: DatasetAdapter):
        return asyncio.run(self.run_async(loader))
