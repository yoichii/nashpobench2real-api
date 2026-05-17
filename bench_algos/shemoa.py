import argparse
import collections
import json
import math
import random
import sys
import time
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from nashpobench2api import API
from tqdm import tqdm


def main(args):
    # make save dir
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # make the API instance
    api = API(
        inference_log_dir="../../inference_time_log",
        device_name=args.device_name,
        need_inference_time=True,
    )

    # Collect hyperparameters for timing info
    hyperparams = {
        "maximum_budget": args.maximum_budget,
        "total_num_eval": args.total_num_eval,
        "population_size": args.population_size,
        "use_proxy": args.use_proxy,
        "time_budget": args.time_budget,
        "test_epoch": args.test_epoch,
        "use_incremental_cost": args.use_incremental_cost,
    }

    # Track total timing
    total_start_time = time.time()
    timing_results = []

    # run the algorithm args.runs times
    for i in tqdm(range(args.runs), desc="Runs"):
        run_start_time = time.time()

        # set seed
        random.seed(i)
        # run
        results = successive_halving_evolutionary_mo(
            args, api, use_incremental_cost=args.use_incremental_cost
        )

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
        "algorithm": "SHEMOA",
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


def mutate(population: dict, api):
    # Select a random individual
    member = random.choice(population)
    new_key = deepcopy(member["key"])

    # Select a random element
    mutate_elements = list(new_key.keys())
    element = random.choice(mutate_elements)

    # Mutate
    while True:
        random_key = api.get_random_key()
        if new_key[element] != random_key[element]:
            break
    new_key[element] = random_key[element]

    return new_key


def non_dominated_sorting(population):
    """Return ranks (1,2,...) via iterative non-dominated fronts.

    修正点:
      1. 支配判定を標準 Pareto (>= / <= かつ片方は厳密) に変更。
      2. (=) だけの点は互いに支配しない。
    """
    n = len(population)
    if n == 0:
        return np.array([])
    dominance = np.zeros((n, n), dtype=bool)
    # ペア毎に支配判定
    for (i_idx, i), (j_idx, j) in combinations(enumerate(population), 2):
        i_dom_j = (
            i["acc"] >= j["acc"] and i["inference_time"] <= j["inference_time"]
        ) and (i["acc"] > j["acc"] or i["inference_time"] < j["inference_time"])
        j_dom_i = (
            j["acc"] >= i["acc"] and j["inference_time"] <= i["inference_time"]
        ) and (j["acc"] > i["acc"] or j["inference_time"] < i["inference_time"])
        if i_dom_j:
            dominance[i_idx, j_idx] = True
        if j_dom_i:
            dominance[j_idx, i_idx] = True

    ranking = np.ones(n, dtype=int)
    dom = dominance.copy()
    current_rank = 1
    remaining = set(range(n))
    while remaining:
        # 現在非支配な index 群
        front = [i for i in remaining if not dom[:, i].any()]
        for i in front:
            ranking[i] = current_rank
        # front を除去
        for i in front:
            dom[i, :] = False
            if i in remaining:
                remaining.remove(i)
        current_rank += 1
    return ranking


try:  # optional dependency (already used elsewhere in repo)
    from pygmo import hypervolume as _pg_hv

    _HAS_PYGMO = True
except Exception:  # noqa
    _HAS_PYGMO = False


def _compute_hv_minimization(points_min, ref_min):
    """Compute hypervolume in minimization space (points_min, ref_min both ndarray).
    Falls back to a simple 2D decomposition if pygmo is unavailable and dim==2.
    """
    if _HAS_PYGMO:
        return _pg_hv(points_min.tolist()).compute(ref_min.tolist())
    # Fallback (2D only): slice accumulation
    if points_min.shape[1] != 2:
        raise RuntimeError("pygmo が無い場合の fallback は2次元のみ対応です。")
    pts = points_min[np.argsort(points_min[:, 0])]  # sort by first objective asc
    hv = 0.0
    prev = ref_min[0]
    # reference は “worst” なので幅 = (ref_min[0] - x) にはならないよう変換前後で注意。
    # ここでは minimization: ref_min は各軸で最大(= worst) を仮定。
    for x, y in pts:
        hv += max(0.0, (ref_min[0] - x)) * max(0.0, (ref_min[1] - y)) - max(
            0.0, (ref_min[0] - prev)
        ) * max(0.0, (ref_min[1] - y))
        prev = x
    return hv


def hypervolume_contributions_max_acc_min_time(acc_time_points):
    """Return per-point HV contributions for points given as (acc,maximize, time,minimize).

    変換: loss1 = 100 - acc (minimize), loss2 = time (minimize)
    ref: 各軸 worst * 1.05 (少し外側)
    """
    if len(acc_time_points) == 0:
        return np.array([]), 0.0
    pts = np.array(acc_time_points, dtype=float)
    losses = np.column_stack([100.0 - pts[:, 0], pts[:, 1]])
    ref = np.max(losses, axis=0) * 1.05 + 1e-9  # small epsilon to ensure strictly worse
    base_hv = _compute_hv_minimization(losses, ref)
    contribs = []
    for i in range(len(losses)):
        mask = np.ones(len(losses), dtype=bool)
        mask[i] = False
        hv_wo = _compute_hv_minimization(losses[mask], ref) if np.any(mask) else 0.0
        contribs.append(base_hv - hv_wo)
    return np.array(contribs), base_hv


def get_budgets(bmin, bmax, eta, max_evals, pop_size):
    """
    Calculate budgets and evaluations per budget according to SH-EMOA paper.
    From the original implementation in the paper's repository.
    """
    # Size of all budgets
    budgets = []
    b = bmax
    while b > bmin:
        budgets.append(b)
        b = math.ceil(b / eta)

    # Number of function evaluations to do per budget
    evals = []
    min_evals = math.ceil(
        (max_evals - pop_size) / sum([eta**i for i in range(len(budgets))])
    )
    for _ in range(len(budgets)):
        evals.append(min_evals)
        min_evals = eta * min_evals

    return np.flip(np.array(budgets)), np.flip(np.array(evals))


def sort_population(population, ranking):
    """
    Return indices of the population sorted by (rank, acc), without copying
    or mutating the original population entries.
    """
    order = sorted(
        range(len(population)), key=lambda i: (ranking[i], population[i]["acc"])
    )
    return order


def successive_halving_evolutionary_mo(args, api, use_incremental_cost=True):
    """
    SH-EMOA with optional incremental cost accounting.

    Args:
        use_incremental_cost: If True, uses proper Successive Halving cost accounting
            where continuing training from a checkpoint only charges the incremental cost.
    """
    # Clear budget cache at the start
    api.clear_budget_cache()

    # Initialization - use the proper SH-EMOA budget calculation
    budgets, evals = get_budgets(
        bmin=1,  # minimum budget
        bmax=args.maximum_budget,
        eta=2,  # standard successive halving parameter
        max_evals=args.total_num_eval,
        pop_size=args.population_size,
    )

    # Choose query function
    query_fn = (
        api.query_by_key_incremental if use_incremental_cost else api.query_by_key
    )

    population = []

    # Init population
    init_pbar = tqdm(total=args.population_size, desc="Initializing population")
    while len(population) < args.population_size:
        key = api.get_random_key()
        # Evaluate with the first (smallest) budget
        acc, inference_time, cost = query_fn(
            **key, epoch=12 if args.use_proxy else 200, iepoch=budgets[0] - 1
        )

        # Add to the population
        population.append(
            {
                "key": key,
                "acc": acc,
                "inference_time": inference_time,
            }
        )
        init_pbar.update(1)
    init_pbar.close()

    # EA with successive halving - controlled by time budget
    total_evals = api.get_total_cost()
    eval_pbar = tqdm(
        initial=total_evals,
        total=args.time_budget,
        desc="Evaluating models",
        unit="cost",
    )

    # Continue cycling through budget levels until time budget is exhausted
    cycle_count = 0
    while api.get_total_cost() < args.time_budget:
        cycle_count += 1

        # Clear budget cache at the start of each cycle so configs are re-evaluated from scratch
        if cycle_count > 1:
            api.clear_budget_cache()

        # Iterate through budgets
        for budget_idx, (budget, n_evals) in enumerate(zip(budgets, evals)):
            # Check if we've exceeded time budget
            if api.get_total_cost() >= args.time_budget:
                break

            # For budgets > 0, re-evaluate population with new budget (incremental cost)
            if budget_idx > 0:
                for member in population:
                    if api.get_total_cost() >= args.time_budget:
                        break
                    acc, inference_time, cost = query_fn(
                        **member["key"],
                        epoch=12 if args.use_proxy else 200,
                        iepoch=budget - 1,
                    )
                    member["acc"] = acc
                    member["inference_time"] = inference_time

                    # Update progress bar
                    new_total = api.get_total_cost()
                    eval_pbar.update(new_total - total_evals)
                    total_evals = new_total

            # Evolution loop for this budget
            for i_eval in range(n_evals):
                # Check if we've exceeded time budget before each evaluation
                if api.get_total_cost() >= args.time_budget:
                    break

                # Generate a new candidate (new candidates start fresh, so use regular query)
                key = mutate(population, api)
                acc, inference_time, cost = query_fn(
                    **key, epoch=12 if args.use_proxy else 200, iepoch=budget - 1
                )
                new_candidate = {
                    "key": key,
                    "acc": acc,
                    "inference_time": inference_time,
                }

                # Select an old candidate using SH-EMOA selection
                ## Sort population by non-dominated sorting
                ranking = non_dominated_sorting(population)
                # Sorted indices by (rank, acc)
                order = sort_population(population, ranking)

                ## Find the worst front (highest rank)
                max_rank = int(max(ranking))
                worst_front_indices = [
                    i for i, r in enumerate(ranking) if int(r) == max_rank
                ]
                worst_front_points = np.array(
                    [
                        [population[i]["acc"], population[i]["inference_time"]]
                        for i in worst_front_indices
                    ]
                )

                if len(worst_front_indices) == 1:
                    # Only one member in worst front, remove it directly
                    remove_idx = worst_front_indices[0]
                else:
                    # Proper HV contribution based removal: remove point with smallest contribution
                    contribs, _ = hypervolume_contributions_max_acc_min_time(
                        worst_front_points
                    )
                    if not np.all(np.isfinite(contribs)):
                        # fallback: remove the one with lowest (acc - time_norm や単純なヒューリスティック)
                        time_norm = (
                            worst_front_points[:, 1] - worst_front_points[:, 1].min()
                        ) / (worst_front_points[:, 1].ptp() + 1e-9)
                        score = worst_front_points[:, 0] - time_norm  # higher better
                        remove_local = int(np.argmin(score))
                    else:
                        remove_local = int(np.argmin(contribs))
                    remove_idx = worst_front_indices[remove_local]

                # Modify population: replace the removed individual with the new candidate
                population.pop(remove_idx)
                population.append(new_candidate)

                # Update progress bar
                new_total = api.get_total_cost()
                eval_pbar.update(new_total - total_evals)
                total_evals = new_total

    eval_pbar.close()
    return api.get_results(epoch=args.test_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHEMOA Search Script")

    # hyperparameters for SHEMOA
    parser.add_argument(
        "--maximum_budget",
        type=int,
        default=12,
        help="The maximum budget (epoch) for evaluating individuals.",
    )
    parser.add_argument(
        "--total_num_eval",
        type=int,
        default=250,
        help="The total number of evaluations",
    )
    parser.add_argument(
        "--population_size", type=int, default=10, help="The population size in SHEMOA."
    )
    parser.add_argument(
        "--use_proxy",
        type=int,
        default=1,
        help="Whether to use the proxy (H0) task or not.",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=86400,
        help="Total time budget for search in seconds.",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="The total runs for evaluation."
    )
    parser.add_argument(
        "--test_epoch",
        default=200,
        help="The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/shemoa",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--use_incremental_cost",
        action="store_true",
        default=True,
        help="Use incremental cost accounting for Successive Halving (default: True)",
    )
    parser.add_argument(
        "--no_incremental_cost",
        dest="use_incremental_cost",
        action="store_false",
        help="Use full cost accounting (legacy behavior)",
    )
    parser.add_argument(
        "--device_name",
        type=str,
        default="raspi-cpu",
        help="Device name for inference time measurement.",
    )
    args = parser.parse_args()

    main(args)
