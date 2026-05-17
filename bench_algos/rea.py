import argparse
import collections
import json
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from math import exp, log
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

    # Extract parameters from args
    population_size = args.population_size
    sample_size = args.sample_size
    time_budget = args.time_budget
    use_proxy = bool(args.use_proxy)
    test_epochs = args.test_epoch

    # Collect hyperparameters for timing info
    hyperparams = {
        "population_size": population_size,
        "sample_size": sample_size,
        "time_budget": time_budget,
        "use_proxy": use_proxy,
        "test_epoch": test_epochs,
    }

    # Track total timing
    total_start_time = time.time()
    timing_results = []

    # run the algorithm args.runs times
    for i in tqdm(range(args.runs), desc="Runs"):
        run_start_time = time.time()

        # set seed
        random.seed(i)
        # run with extracted parameters
        results = regularized_evolution(
            api=api,
            population_size=population_size,
            sample_size=sample_size,
            time_budget=time_budget,
            use_proxy=use_proxy,
            test_epochs=test_epochs,
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
        # save results
        save_path = save_dir / f"seed{i}.json"
        save_path.write_text(json.dumps(results_with_timing, indent=2))

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Save timing summary
    timing_summary = {
        "algorithm": "Regularized Evolution",
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


class Model:
    """Class representing an individual in the population.

    Attributes:
        key (dict): The configuration dictionary for the model
        acc (float): The accuracy achieved by this configuration
    """

    def __init__(self, key: dict, acc: float):
        self.key = key
        self.acc = acc


def mutate(key: dict, api: API) -> dict:
    """Mutate a given configuration by randomly changing one of its parameters.

    Args:
        key: The configuration dictionary to mutate
        api: The API instance for getting random parameters

    Returns:
        dict: A new configuration with one parameter mutated
    """
    # Parameter ranges from reinforce.py
    PARAM_RANGES = {
        "initial_lr": (0.003125, 0.4),  # Learning rate bounds
        "momentum": (0.7, 0.9),  # Momentum bounds
        "weight_decay": (1e-4, 5e-3),  # Weight decay bounds
        "randomflip": (0.25, 0.75),  # Random flip bounds
        "batch_size": [16, 32, 64, 128, 256, 512],  # Batch sizes
        "optimizer": ["SGD", "RMSprop", "Adam"],  # Optimizers
        "cellcode": list(range(4)),  # Cell operations (0-3)
    }

    new_key = deepcopy(key)
    mutate_elements = list(key.keys())
    element = random.choice(mutate_elements)

    # Discrete elements mutation
    if element.startswith("cellcode"):
        candidates = [i for i in PARAM_RANGES["cellcode"] if i != key[element]]
        new_key[element] = random.choice(candidates)
    elif element == "optimizer":
        candidates = [opt for opt in PARAM_RANGES["optimizer"] if opt != key[element]]
        new_key[element] = random.choice(candidates)
    elif element == "batch_size":
        candidates = [bs for bs in PARAM_RANGES["batch_size"] if bs != key[element]]
        new_key[element] = random.choice(candidates)
    # Continuous elements mutation
    else:
        min_val, max_val = PARAM_RANGES[element]
        if element in ["initial_lr", "weight_decay"]:
            # Sample in log space for learning rate and weight decay
            log_min, log_max = log(min_val), log(max_val)
            new_val = exp(random.uniform(log_min, log_max))
        else:
            # Linear space sampling for momentum and randomflip
            new_val = random.uniform(min_val, max_val)
        new_key[element] = float(new_val)

    return new_key


def regularized_evolution(
    api: API,
    population_size: int,
    sample_size: int,
    time_budget: int,
    use_proxy: bool,
    test_epochs: int,
) -> dict:
    """Regularized Evolution Algorithm (aging evolution) for architecture search.

    This implementation follows "Algorithm 1" in Real et al. "Regularized Evolution
    for Image Classifier Architecture Search". The algorithm maintains a population
    of models and evolves them through cycles of tournament selection and mutation.

    Args:
        api: API instance for model evaluation
        population_size: Size of the population to maintain
        sample_size: Number of candidates to sample for tournament selection
        time_budget: Total time budget for search in seconds
        use_proxy: Whether to use the proxy (12 epoch) task or full task
        test_epochs: Number of epochs to use for final evaluation

    Returns:
        dict: Final results from the evolution process
    """
    # Initialize population as a double-ended queue for efficient removal of oldest member
    population = collections.deque()

    # Initialize population with random architectures
    init_pbar = tqdm(total=population_size, desc="Initializing population")
    while len(population) < population_size:
        key = api.get_random_key()
        # Query accuracy using either proxy (12 epochs) or full (200 epochs) task
        epoch = 12 if use_proxy else 200
        acc, cost = api.query_by_key(**key, epoch=epoch)
        population.append(Model(key, acc))
        init_pbar.update(1)
    init_pbar.close()

    # Evolution cycles: each cycle produces a child and removes the oldest model
    total_evals = api.get_total_cost()
    eval_pbar = tqdm(
        initial=total_evals, total=time_budget, desc="Evaluating models", unit="cost"
    )
    while api.get_total_cost() < time_budget:
        # Tournament selection: sample random candidates
        sample = []
        while len(sample) < sample_size:
            # Random sampling is inefficient but clear; performance impact is negligible
            # compared to model training time
            candidate = random.choice(list(population))
            sample.append(candidate)

        # Select the best model from the tournament as parent
        parent = max(sample, key=lambda x: x.acc)

        # Create and evaluate child model through mutation
        child_key = mutate(parent.key, api)
        epoch = 12 if use_proxy else 200
        acc, cost = api.query_by_key(**child_key, epoch=epoch)
        population.append(Model(child_key, acc))

        # Remove oldest model to maintain fixed population size
        population.popleft()

        # Update progress bar
        new_total = api.get_total_cost()
        eval_pbar.update(new_total - total_evals)
        total_evals = new_total

    eval_pbar.close()
    # Return final results at specified test epoch
    return api.get_results(epoch=test_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regularized Evolution Algorithm (REA) Search Script"
    )

    # Basic configuration
    parser.add_argument(
        "--time_budget",
        type=int,
        default=86400,
        help="Total time budget for search in seconds.",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of independent random-seed runs."
    )
    parser.add_argument(
        "--test_epoch",
        type=int,
        default=12,
        help="The test epoch. 12 (trained), 200 (surrogate), or both (12 and 200).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/rea",
        help="Directory to save result files.",
    )

    # REA specific parameters
    parser.add_argument(
        "--population_size",
        type=int,
        default=10,
        help="Size of the population to maintain during evolution.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=3,
        help="Number of candidates to sample for tournament selection.",
    )
    parser.add_argument(
        "--use_proxy",
        type=int,
        default=1,
        help="Whether to use the proxy (12 epoch) task (1) or full task (0).",
    )

    args = parser.parse_args()

    main(args)
