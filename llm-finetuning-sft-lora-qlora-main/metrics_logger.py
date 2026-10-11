import json
import time
import os
import torch

class ExperimentLogger:
    def __init__(self, method_name, results_path="results.json"):
        self.method_name = method_name
        self.results_path = results_path
        self.start_time = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()

    def log_trainable_params(self, model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.trainable_params = trainable
        self.total_params = total
        self.trainable_pct = round(100 * trainable / total, 4)

    def finish(self, eval_loss=None):
        elapsed = time.time() - self.start_time
        peak_mem_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else None
        )

        result = {
            "method": self.method_name,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_pct": self.trainable_pct,
            "peak_gpu_memory_gb": round(peak_mem_gb, 3) if peak_mem_gb else None,
            "run_time_sec": round(elapsed, 1),
            "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
        }

        all_results = []
        if os.path.exists(self.results_path):
            with open(self.results_path) as f:
                all_results = json.load(f)
        all_results = [r for r in all_results if r["method"] != self.method_name]
        all_results.append(result)
        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n[RESULT] {self.method_name}: {result}\n")
        return result