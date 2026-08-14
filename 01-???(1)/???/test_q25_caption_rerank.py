import asyncio
import importlib.util
import json
import sys
from pathlib import Path


candidate_path = Path(sys.argv[1]).resolve()
target_path = Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location("api_server", candidate_path)
module = importlib.util.module_from_spec(spec)
sys.modules["api_server"] = module
assert spec.loader is not None
spec.loader.exec_module(module)

engine = asyncio.run(module.get_engine())
question = "图片里的两个方形铁盘是什么？分别有什么用途？"
expanded = module._expand_visual_caption_query(question)
products = module._caption_product_candidates(expanded, engine.catalog, limit=5)
candidates = module._caption_image_candidates(
    expanded,
    engine.catalog,
    products=products,
    limit=8,
)
match = module._global_ground_image_to_manual(
    question,
    [module._manual_image_data_url(str(target_path))],
    {
        "objects": "two nested rectangular dark metal trays",
        "focus": "two square metal pans",
        "intent": "identify both accessories and their uses",
        "normalized_question": question,
        "search_terms": ["baking tray", "drip tray", "grease pan"],
    },
    candidates=candidates,
)
print(json.dumps({
    "products": products,
    "candidates": [
        {
            "image_id": item.get("image_id"),
            "product": item.get("product"),
            "heading": item.get("heading"),
            "score": item.get("caption_score"),
        }
        for item in candidates
    ],
    "match": {
        "image_ids": (match or {}).get("image_ids"),
        "product": (match or {}).get("product"),
        "confidence": (match or {}).get("confidence"),
        "heading": (match or {}).get("heading"),
        "reason": (match or {}).get("reason"),
    },
}, ensure_ascii=False))
