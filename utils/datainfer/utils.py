import base64
from pathlib import Path
from typing import Dict

def encode64_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def convert_image_content_to_openai(content: Dict):
    """
    Convert image content to OpenAI format and replace in-place.
        from `{"type": "image", "image": "..."}`
        to   `{"type": "image_url", "image_url": {"url": "..."}}`
    if input is a local path, convert it to base64 string.
    """
    content["type"] = "image_url"
    url_or_path: str = content.pop("image")
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        content["image_url"] = {"url": url_or_path}
    else:
        ext = Path(url_or_path).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            raise ValueError(f"Unsupported image format: {ext} in path: {url_or_path}")
        # local image path, convert to base64
        base64_str = encode64_file(url_or_path)
        content["image_url"] = {"url": f"data:image/{ext[1:]};base64,{base64_str}"}