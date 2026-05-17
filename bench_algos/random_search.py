import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from nashpobench2api import API
from tqdm import tqdm


@contextmanager
def timing_context():
    """Context manager to measure wall-clock time"""
    start_time = time.time()
    yield
    end_time = time.time()
    return end_time - start_time


def main(args):
    # make save dir
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # make the API instance
    api = API(need_inference_time=False)

    # Collect hyperparameters for timing info
    hyperparams = {"time_budget": args.time_budget, "test_epoch": args.test_epoch}

    # Track total timing
    total_start_time = time.time()
    timing_results = []

    # run the algorithm args.runs times
    for i in tqdm(range(args.runs)):
        run_start_time = time.time()

        # set seed
        random.seed(i)
        # run
        results = random_search(args, api)

        run_end_time = time.time()
        run_duration = run_end_time - run_start_time

        # Add timing info to results
        results_with_timing = {
            "results": results,
            "timing": {
                "wall_clock_time_seconds": run_duration,
                "hyperparameters": hyperparams,
            },
        }
        timing_results.append(run_duration)

        # reset log
        api.reset_logdata()
        # save results as JSON
        save_path = save_dir / f"seed{i}.json"
        save_path.write_text(json.dumps(results_with_timing, indent=2))

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Save timing summary
    timing_summary = {
        "algorithm": "Random Search",
        "hyperparameters": hyperparams,
        "total_runs": args.runs,
        "total_wall_clock_time_seconds": total_duration,
        "average_run_time_seconds": np.mean(timing_results),
        "std_run_time_seconds": np.std(timing_results),
        "min_run_time_seconds": np.min(timing_results),
        "max_run_time_seconds": np.max(timing_results),
        "per_run_times": timing_results,
    }

    timing_path = save_dir / "timing_summary.json"
    timing_path.write_text(json.dumps(timing_summary, indent=2))


# Random Search for Hyper-Parameter Optimization, JMLR 2012
def random_search(args, api):
    while api.get_total_cost() < args.time_budget:
        # get keys(=cellcode, lr, batch_size)
        key = api.get_random_key()
        # query accuracy
        acc, cost = api.query_by_key(**key, epoch=12)
    return api.get_results(epoch=args.test_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RANDOM")
    parser.add_argument(
        "--time_budget",
        type=int,
        default=86400,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="The total runs for evaluation."
    )
    parser.add_argument(
        "--test_epoch",
        default=12,
        help="The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).",
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/random",
        help="Folder to save checkpoints and log.",
    )
    args = parser.parse_args()

    main(args)
