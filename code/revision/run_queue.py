"""
Runs pipeline_v2.run_dataset for a list of datasets (resumable). Usage:
  .venv/bin/python code/revision/run_queue.py <train_name>:<test_name>:<results_subdir> [...]
Example: tri_rep1:test_triangular:E4_mc_replicates/tri_rep1
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import run_dataset, RESULTS  # noqa: E402

for job in sys.argv[1:]:
    train, test, sub = job.split(":")
    try:
        run_dataset(train, test, os.path.join(RESULTS, sub))
        print("DONE", job, flush=True)
    except Exception:
        print("FAILED", job, flush=True)
        traceback.print_exc()
