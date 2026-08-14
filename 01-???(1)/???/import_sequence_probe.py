import importlib
import time


for module_name in (
    "faiss",
    "jieba",
    "numpy",
    "requests",
    "rank_bm25",
    "tqdm",
    "dotenv",
    "rerank_client",
    "retrieval_engine",
):
    started = time.perf_counter()
    print(f"IMPORT_START {module_name}", flush=True)
    importlib.import_module(module_name)
    print(
        f"IMPORT_OK {module_name} elapsed={time.perf_counter() - started:.3f}s",
        flush=True,
    )
