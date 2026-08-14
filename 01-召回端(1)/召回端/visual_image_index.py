"""Local DINOv2 image index for manual-figure recall.

The index is optional: callers can fall back to caption retrieval whenever the
model, weights, or persisted index are unavailable.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
ROOT = Path(__file__).resolve().parent
DEFAULT_CAPTIONS_PATH = ROOT / "data" / "image_captions_v4_final.json"
DEFAULT_IMAGE_DIR = Path(
    os.getenv("RAYSOURCE_MANUAL_IMAGE_DIR", str(ROOT / "data" / "manual-images"))
)
DEFAULT_INDEX_PATH = ROOT / "data" / "visual_image_index_dinov2.npz"

_LOCK = threading.Lock()
_MODEL: Any = None
_TRANSFORM: Any = None
_INDEX: tuple[np.ndarray, list[dict[str, str]]] | None = None


def _runtime() -> tuple[Any, Any]:
    global _MODEL, _TRANSFORM
    with _LOCK:
        if _MODEL is None or _TRANSFORM is None:
            import timm
            from timm.data import create_transform, resolve_data_config

            model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval()
            transform = create_transform(
                **resolve_data_config(model.pretrained_cfg, model=model),
                is_training=False,
            )
            _MODEL, _TRANSFORM = model, transform
    return _MODEL, _TRANSFORM


def _embed(images: list[Image.Image]) -> np.ndarray:
    import torch

    model, transform = _runtime()
    with torch.inference_mode():
        batch = torch.stack([transform(image.convert("RGB")) for image in images])
        vectors = model(batch).float().cpu().numpy()
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors.astype(np.float32)


def _image_path(image_dir: Path, image_id: str) -> Path | None:
    return next((
        image_dir / f"{image_id}{extension}"
        for extension in (".jpg", ".jpeg", ".png", ".webp")
        if (image_dir / f"{image_id}{extension}").is_file()
    ), None)


def build_index(
    captions_path: Path = DEFAULT_CAPTIONS_PATH,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    index_path: Path = DEFAULT_INDEX_PATH,
    batch_size: int = 16,
) -> dict[str, Any]:
    payload = json.loads(Path(captions_path).read_text(encoding="utf-8"))
    metadata: list[dict[str, str]] = []
    for item in (payload.get("items") or {}).values():
        product = str(item.get("product") or "").strip()
        image_id = str(item.get("image_id") or "").strip()
        path = _image_path(Path(image_dir), image_id)
        if not product or not image_id or path is None:
            continue
        metadata.append({
            "product": product,
            "image_id": image_id,
            "path": str(path),
            "caption": " ".join(
                str(item.get(key) or "") for key in ("short_caption", "content", "reason")
            )[:1200],
        })

    batches: list[np.ndarray] = []
    valid_metadata: list[dict[str, str]] = []
    for offset in range(0, len(metadata), max(1, batch_size)):
        rows = metadata[offset:offset + max(1, batch_size)]
        images: list[Image.Image] = []
        accepted: list[dict[str, str]] = []
        for row in rows:
            try:
                images.append(Image.open(row["path"]).convert("RGB"))
                accepted.append(row)
            except Exception:
                continue
        if images:
            batches.append(_embed(images))
            valid_metadata.extend(accepted)

    vectors = np.concatenate(batches, axis=0) if batches else np.empty((0, 384), dtype=np.float32)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        vectors=vectors.astype(np.float16),
        metadata=np.asarray([json.dumps(row, ensure_ascii=False) for row in valid_metadata]),
        model=np.asarray([MODEL_NAME]),
    )
    global _INDEX
    _INDEX = (vectors, valid_metadata)
    return {"images": len(valid_metadata), "dimensions": int(vectors.shape[1]), "path": str(index_path)}


def load_index(index_path: Path = DEFAULT_INDEX_PATH) -> tuple[np.ndarray, list[dict[str, str]]]:
    global _INDEX
    with _LOCK:
        if _INDEX is None:
            payload = np.load(Path(index_path), allow_pickle=False)
            vectors = payload["vectors"].astype(np.float32)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            metadata = [json.loads(str(item)) for item in payload["metadata"].tolist()]
            _INDEX = (vectors, metadata)
    return _INDEX


def image_from_data_url(data_url: str) -> Image.Image:
    encoded = str(data_url).split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def search(data_url: str, top_k: int = 20) -> list[dict[str, Any]]:
    vectors, metadata = load_index()
    query = _embed([image_from_data_url(data_url)])[0]
    scores = vectors @ query
    order = np.argsort(-scores)[:max(1, top_k)]
    return [{**metadata[int(index)], "visual_score": float(scores[int(index)])} for index in order]


if __name__ == "__main__":
    print(json.dumps(build_index(), ensure_ascii=False))
