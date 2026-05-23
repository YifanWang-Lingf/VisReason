import re
from typing import List, Optional, Tuple


def _box_score(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: Optional[float],
    img_h: Optional[float],
) -> float:
    if x2 <= x1 or y2 <= y1:
        return -1e9

    if not img_w or not img_h:
        return 0.0

    w, h = float(img_w), float(img_h)
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return -1e9

    inside = 0
    for value in (x1, x2):
        if 0 <= value <= w:
            inside += 1
    for value in (y1, y2):
        if 0 <= value <= h:
            inside += 1

    score = inside * 3.0
    ratio = area / (w * h)
    if ratio > 2.0:
        score -= 4.0
    elif ratio > 1.0:
        score -= 2.0
    elif ratio < 1e-4:
        score -= 1.0

    return score


def convert_quad_to_box(
    quad: List[float],
    img_w: Optional[float],
    img_h: Optional[float],
) -> Optional[List[float]]:
    if len(quad) != 4:
        return None

    a, b, c, d = map(float, quad)

    x1_xyxy, y1_xyxy, x2_xyxy, y2_xyxy = a, b, c, d
    if x2_xyxy < x1_xyxy:
        x1_xyxy, x2_xyxy = x2_xyxy, x1_xyxy
    if y2_xyxy < y1_xyxy:
        y1_xyxy, y2_xyxy = y2_xyxy, y1_xyxy

    score_xyxy = _box_score(x1_xyxy, y1_xyxy, x2_xyxy, y2_xyxy, img_w, img_h) + 0.5
    candidates: List[Tuple[float, str, List[float]]] = [
        (score_xyxy, "xyxy", [x1_xyxy, y1_xyxy, x2_xyxy, y2_xyxy])
    ]

    if c > 0 and d > 0:
        cx, cy, width, height = a, b, c, d
        cxcywh_box = [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ]
        candidates.append((
            _box_score(*cxcywh_box, img_w, img_h),
            "cxcywh",
            cxcywh_box,
        ))

        xywh_box = [a, b, a + c, b + d]
        candidates.append((
            _box_score(*xywh_box, img_w, img_h),
            "xywh",
            xywh_box,
        ))

    best_score, best_mode, best_box = max(candidates, key=lambda item: item[0])
    if best_mode != "xyxy" and best_score >= score_xyxy + 3.0:
        x1, y1, x2, y2 = best_box
    else:
        x1, y1, x2, y2 = x1_xyxy, y1_xyxy, x2_xyxy, y2_xyxy

    if img_w and img_h:
        w, h = float(img_w), float(img_h)
        tol = 1e-3
        if x1 >= w - tol and x2 <= 2.0 * w + tol and y1 >= -tol and y2 <= h + tol:
            x1 -= w
            x2 -= w
        elif y1 >= h - tol and y2 <= 2.0 * h + tol and x1 >= -tol and x2 <= w + tol:
            y1 -= h
            y2 -= h

        x1 = min(max(x1, 0.0), w)
        x2 = min(max(x2, 0.0), w)
        y1 = min(max(y1, 0.0), h)
        y2 = min(max(y2, 0.0), h)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def _extract_boxed_contents(text: str) -> List[str]:
    if not text:
        return []

    contents = []
    cursor = 0
    marker = r"\boxed{"
    text_len = len(text)

    while cursor < text_len:
        start_marker = text.find(marker, cursor)
        if start_marker == -1:
            break

        cursor = start_marker + len(marker)
        start_content = cursor
        depth = 1
        while cursor < text_len and depth > 0:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1

        if depth == 0:
            contents.append(text[start_content:cursor - 1])
        else:
            break

    return contents


def parse_pred_quads(response: str) -> List[List[float]]:
    if not response or not response.strip():
        return []

    boxed_contents = _extract_boxed_contents(response)
    content = "\n".join(boxed_contents) if boxed_contents else response
    number = r"-?\d+(?:\.\d+)?"

    quads: List[List[float]] = []
    list4 = re.compile(
        rf"\[\s*({number})\s*,\s*({number})\s*,\s*({number})\s*,\s*({number})\s*\]"
    )
    for a, b, c, d in list4.findall(content):
        quads.append([float(a), float(b), float(c), float(d)])
    if quads:
        return quads

    semicolon_blocks = re.findall(r"\[[^\[\]]*;[^\[\]]*\]", content)
    for block in semicolon_blocks:
        values = [float(value) for value in re.findall(number, block)]
        for idx in range(0, len(values) - 3, 4):
            quads.append(values[idx:idx + 4])
    if quads:
        return quads

    square_pairs = re.findall(rf"\[\s*({number})\s*,\s*({number})\s*\]", content)
    round_pairs = re.findall(rf"\(\s*({number})\s*,\s*({number})\s*\)", content)
    pairs = [(float(a), float(b)) for a, b in (square_pairs + round_pairs)]
    if len(pairs) >= 2 and len(pairs) % 2 == 0:
        for idx in range(0, len(pairs), 2):
            (x1, y1), (x2, y2) = pairs[idx], pairs[idx + 1]
            quads.append([x1, y1, x2, y2])
        if quads:
            return quads

    values = [float(value) for value in re.findall(number, content)]
    for idx in range(0, len(values) - 3, 4):
        quads.append(values[idx:idx + 4])

    return quads


def iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    if area1 <= 0 or area2 <= 0:
        return 0.0

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area
