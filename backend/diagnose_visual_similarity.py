from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def load_image(value: str) -> np.ndarray:
    if value.lower().startswith(("http://", "https://")):
        request = urllib.request.Request(value, headers={"User-Agent": "RaysourceVisualIndex/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = np.frombuffer(response.read(), dtype=np.uint8)
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(value, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {value}")
    return image


def normalize_gray(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))))
    return gray


def phash(image: np.ndarray) -> np.ndarray:
    gray = normalize_gray(image)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(resized)[:8, :8]
    median = np.median(low[1:, :])
    return (low > median).reshape(-1)


def hist(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(value, value)
    return value


def orb_features(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    detector = cv2.ORB_create(nfeatures=1100, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    return detector.detectAndCompute(normalize_gray(image), None)


def orb_score(
    query_kp: list[cv2.KeyPoint],
    query_desc: np.ndarray | None,
    candidate_kp: list[cv2.KeyPoint],
    candidate_desc: np.ndarray | None,
) -> tuple[float, int, float]:
    if query_desc is None or candidate_desc is None or len(query_desc) < 4 or len(candidate_desc) < 4:
        return 0.0, 0, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(query_desc, candidate_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.74 * second.distance]
    normalized = len(good) / max(12.0, math.sqrt(len(query_desc) * len(candidate_desc)))
    inlier_ratio = 0.0
    if len(good) >= 6:
        src = np.float32([query_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        dst = np.float32([candidate_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_ratio = float(mask.ravel().mean())
    return min(1.0, normalized), len(good), inlier_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("captions", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    target = load_image(args.target)
    target_hash = phash(target)
    target_hist = hist(target)
    target_kp, target_desc = orb_features(target)
    caption_payload = json.loads(args.captions.read_text(encoding="utf-8"))
    caption_items = caption_payload.get("items") or {}

    ranked: list[dict[str, object]] = []
    for path in args.image_dir.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        candidate = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if candidate is None:
            continue
        candidate_hash = phash(candidate)
        hash_similarity = 1.0 - float(np.count_nonzero(target_hash != candidate_hash)) / 64.0
        histogram_similarity = float(cv2.compareHist(target_hist, hist(candidate), cv2.HISTCMP_CORREL))
        candidate_kp, candidate_desc = orb_features(candidate)
        local_score, good_matches, inlier_ratio = orb_score(
            target_kp, target_desc, candidate_kp, candidate_desc
        )
        combined = (
            max(0.0, local_score) * 0.42
            + max(0.0, inlier_ratio) * 0.33
            + max(0.0, hash_similarity) * 0.20
            + max(0.0, histogram_similarity) * 0.05
        )
        image_id = path.stem
        metadata = caption_items.get(image_id) or {}
        ranked.append({
            "image_id": image_id,
            "product": metadata.get("product"),
            "combined": round(combined, 5),
            "orb": round(local_score, 5),
            "good": good_matches,
            "inliers": round(inlier_ratio, 5),
            "phash": round(hash_similarity, 5),
            "hist": round(histogram_similarity, 5),
            "caption": str(metadata.get("short_caption") or metadata.get("content") or "")[:180],
        })

    ranked.sort(key=lambda item: float(item["combined"]), reverse=True)
    print(json.dumps(ranked[: args.top], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
