import argparse
import json
import pickle
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nashpobench2api import API
from torch.distributions import Categorical, Normal
from tqdm import tqdm


@contextmanager
def timing_context():
    """Context manager to measure wall-clock time"""
    start_time = time.time()
    yield
    end_time = time.time()
    return end_time - start_time


class PolicyTopology(nn.Module):
    def __init__(
        self,
        num_edges: int,
        batch_sizes: list,
        lr_init: float = 0.025,
        lr_std_init: float = 2.0,
        momentum_init: float = 0.8,
        momentum_std_init: float = 0.1,
        weight_decay_init: float = 1e-3,
        weight_decay_std_init: float = 5.0,
        randomflip_init: float = 0.5,
        randomflip_std_init: float = 0.25,
    ):
        super().__init__()
        self.num_edges = num_edges
        self.ops = [0, 1, 2, 3]
        self.batch_sizes = deepcopy(batch_sizes)
        self.optimizers = ["SGD", "RMSprop", "Adam"]

        self.cell_params = nn.Parameter(
            1e-3 * torch.randn(self.num_edges, len(self.ops))
        )

        # Learning rate (log space)
        self.lr_mean = nn.Parameter(torch.tensor(np.log(lr_init)))
        self.lr_std = nn.Parameter(torch.tensor(np.log(lr_std_init)))

        self.batch_params = nn.Parameter(1e-3 * torch.randn(1, len(self.batch_sizes)))

        self.optimizer_params = nn.Parameter(
            1e-3 * torch.randn(1, len(self.optimizers))
        )

        # Momentum (direct space)
        self.momentum_mean = nn.Parameter(torch.tensor(momentum_init))
        self.momentum_std = nn.Parameter(torch.tensor(momentum_std_init))

        # Weight decay (log space)
        self.weight_decay_mean = nn.Parameter(torch.tensor(np.log(weight_decay_init)))
        self.weight_decay_std = nn.Parameter(
            torch.tensor(np.log(weight_decay_std_init))
        )

        # Random flip (direct space)
        self.randomflip_mean = nn.Parameter(torch.tensor(randomflip_init))
        self.randomflip_std = nn.Parameter(torch.tensor(randomflip_std_init))

    def generate_key(self, actions):
        # Split actions into their components
        cell_actions = actions[: self.num_edges]
        lr = actions[self.num_edges]
        optimizer_idx = actions[self.num_edges + 1]
        momentum = actions[self.num_edges + 2]
        weight_decay = actions[self.num_edges + 3]
        randomflip = actions[self.num_edges + 4]
        batch_idx = actions[self.num_edges + 5]

        # Create the formatted dictionary (values are already clipped in select_action)
        key = {
            "initial_lr": float(lr),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "optimizer": self.optimizers[optimizer_idx],
            "batch_size": self.batch_sizes[batch_idx],
            "randomflip": float(randomflip),
        }

        # Add cellcode components
        for i, action in enumerate(cell_actions):
            key[f"cellcode_{i}"] = int(action)

        return key

    def forward(self):
        cell_prob = F.softmax(self.cell_params, dim=-1)
        optimizer_prob = F.softmax(self.optimizer_params, dim=-1)
        batch_prob = F.softmax(self.batch_params, dim=-1)

        # Return all distribution parameters
        return (
            cell_prob,
            (self.lr_mean, self.lr_std),
            (self.momentum_mean, self.momentum_std),
            (self.weight_decay_mean, self.weight_decay_std),
            (self.randomflip_mean, self.randomflip_std),
            optimizer_prob,
            batch_prob,
        )


class ExponentialMovingAverage:
    """Class that maintains an exponential moving average of observed values."""

    def __init__(self, momentum):
        self._numerator = 0.0
        self._denominator = 0.0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._numerator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        # Return the current moving average value
        return self._numerator / (self._denominator + 1e-8)


def select_action(
    policy,
    lr_min,
    lr_max,
    momentum_min,
    momentum_max,
    weight_decay_min,
    weight_decay_max,
    randomflip_min,
    randomflip_max,
):
    (
        cell_prob,
        (lr_mean, lr_std),
        (momentum_mean, momentum_std),
        (weight_decay_mean, weight_decay_std),
        (randomflip_mean, randomflip_std),
        optimizer_prob,
        batch_prob,
    ) = policy()

    # Sample discrete actions
    cell_d = Categorical(cell_prob)
    optimizer_d = Categorical(optimizer_prob)
    batch_d = Categorical(batch_prob)

    cell_action = cell_d.sample()
    optimizer_action = optimizer_d.sample()
    batch_action = batch_d.sample()

    # Sample continuous actions from Normal distributions
    # Apply softplus to ensure standard deviations are positive
    lr_dist = Normal(lr_mean, F.softplus(lr_std))
    momentum_dist = Normal(
        momentum_mean, F.softplus(momentum_std)
    )  # Direct space for momentum
    weight_decay_dist = Normal(weight_decay_mean, F.softplus(weight_decay_std))
    randomflip_dist = Normal(
        randomflip_mean, F.softplus(randomflip_std)
    )  # Direct space for randomflip

    # Sample in log-space for lr and weight decay
    lr_action_log = lr_dist.sample()
    weight_decay_action_log = weight_decay_dist.sample()

    # Sample momentum and randomflip in original space
    momentum_action = momentum_dist.sample()
    randomflip_action = randomflip_dist.sample()

    # Compute log probabilities first (before clamping)
    cell_log = cell_d.log_prob(cell_action)
    optimizer_log = optimizer_d.log_prob(optimizer_action)
    momentum_log = momentum_dist.log_prob(momentum_action)  # Use original space action
    weight_decay_log = weight_decay_dist.log_prob(weight_decay_action_log)
    randomflip_log = randomflip_dist.log_prob(
        randomflip_action
    )  # Use original space action
    lr_log = lr_dist.log_prob(lr_action_log)
    batch_log = batch_d.log_prob(batch_action)

    # Then transform to appropriate ranges
    lr = torch.exp(lr_action_log).clamp(min=lr_min, max=lr_max)
    momentum = momentum_action.clamp(momentum_min, momentum_max)
    weight_decay = torch.exp(weight_decay_action_log).clamp(
        weight_decay_min, weight_decay_max
    )
    randomflip = randomflip_action.clamp(randomflip_min, randomflip_max)

    # Combine all log probabilities
    log_probs = torch.cat(
        [
            cell_log,
            lr_log.unsqueeze(0),
            optimizer_log,
            momentum_log.unsqueeze(0),
            weight_decay_log.unsqueeze(0),
            randomflip_log.unsqueeze(0),
            batch_log,
        ]
    )

    # Combine all actions in the correct order
    actions = [int(idx) for idx in cell_action.tolist()]
    actions.extend(
        [
            float(lr.item()),
            int(optimizer_action.item()),
            float(momentum.item()),
            float(weight_decay.item()),
            float(randomflip.item()),
            int(batch_action.item()),
        ]
    )

    return log_probs, actions


def reinforce(
    seed,
    api,
    batch_sizes,
    num_edges,
    lr_min,
    lr_max,
    lr_init,
    lr_std_init,
    momentum_min,
    momentum_max,
    momentum_init,
    momentum_std_init,
    weight_decay_min,
    weight_decay_max,
    weight_decay_init,
    weight_decay_std_init,
    randomflip_min,
    randomflip_max,
    randomflip_init,
    randomflip_std_init,
    lr,
    ema_momentum,
    time_budget,
    test_epochs,
):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyTopology(
        num_edges=num_edges,
        batch_sizes=batch_sizes,
        lr_init=lr_init,
        lr_std_init=lr_std_init,
        momentum_init=momentum_init,
        momentum_std_init=momentum_std_init,
        weight_decay_init=weight_decay_init,
        weight_decay_std_init=weight_decay_std_init,
        randomflip_init=randomflip_init,
        randomflip_std_init=randomflip_std_init,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = ExponentialMovingAverage(ema_momentum)
    total_steps = 0

    # Main reinforcement learning loop
    while api.get_total_cost() < time_budget:
        log_prob, action = select_action(
            policy,
            lr_min=lr_min,
            lr_max=lr_max,
            momentum_min=momentum_min,
            momentum_max=momentum_max,
            weight_decay_min=weight_decay_min,
            weight_decay_max=weight_decay_max,
            randomflip_min=randomflip_min,
            randomflip_max=randomflip_max,
        )
        key = policy.generate_key(action)
        reward, cost = api.query_by_key(**key, epoch=12)

        # Update baseline and compute advantage
        baseline.update(reward)
        advantage = reward - baseline.value()
        loss = -(log_prob * advantage).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_steps += 1

    return api.get_results(epoch=test_epochs)


def main(args):
    # make save dir
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # make the API instance
    api = API(need_inference_time=False)

    # Collect hyperparameters for timing info
    hyperparams = {
        "learning_rate": args.learning_rate,
        "EMA_momentum": args.EMA_momentum,
        "time_budget": args.time_budget,
        "test_epoch": args.test_epoch,
        "batch_sizes": args.batch_sizes,
        "num_edges": args.num_edges,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "lr_init": args.lr_init,
        "lr_std_init": args.lr_std_init,
        "momentum_min": args.momentum_min,
        "momentum_max": args.momentum_max,
        "momentum_init": args.momentum_init,
        "momentum_std_init": args.momentum_std_init,
        "weight_decay_min": args.weight_decay_min,
        "weight_decay_max": args.weight_decay_max,
        "weight_decay_init": args.weight_decay_init,
        "weight_decay_std_init": args.weight_decay_std_init,
        "randomflip_min": args.randomflip_min,
        "randomflip_max": args.randomflip_max,
        "randomflip_init": args.randomflip_init,
        "randomflip_std_init": args.randomflip_std_init,
    }

    # Track total timing
    total_start_time = time.time()
    timing_results = []

    # run the algorithm args.runs times
    for i in tqdm(range(args.runs), desc="Runs"):
        run_start_time = time.time()

        # set seed
        results = reinforce(
            seed=i,
            api=api,
            batch_sizes=args.batch_sizes,
            num_edges=args.num_edges,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            lr_init=args.lr_init,
            momentum_min=args.momentum_min,
            momentum_max=args.momentum_max,
            momentum_init=args.momentum_init,
            weight_decay_min=args.weight_decay_min,
            weight_decay_max=args.weight_decay_max,
            weight_decay_init=args.weight_decay_init,
            randomflip_min=args.randomflip_min,
            randomflip_max=args.randomflip_max,
            randomflip_init=args.randomflip_init,
            lr_std_init=args.lr_std_init,
            momentum_std_init=args.momentum_std_init,
            weight_decay_std_init=args.weight_decay_std_init,
            randomflip_std_init=args.randomflip_std_init,
            lr=args.learning_rate,
            ema_momentum=args.EMA_momentum,
            time_budget=args.time_budget,
            test_epochs=args.test_epoch,
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
        "algorithm": "REINFORCE",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE Search Script")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--EMA_momentum",
        type=float,
        default=0.9,
        help="Momentum factor for the baseline EMA.",
    )
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
        default=12,
        help="The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/reinforce",
        help="Directory to save result pickles.",
    )
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Candidate batch sizes to explore.",
    )
    parser.add_argument(
        "--num_edges",
        type=int,
        default=6,
        help="Number of edges in the cell architecture.",
    )
    # Learning rate bounds and initialization
    parser.add_argument(
        "--lr_min",
        type=float,
        default=0.003125,
        help="Minimum allowable learning rate.",
    )
    parser.add_argument(
        "--lr_max", type=float, default=0.4, help="Maximum allowable learning rate."
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.025,
        help="Initial learning rate mean in log space.",
    )
    parser.add_argument(
        "--lr_std_init",
        type=float,
        default=2.0,
        help="Initial learning rate std multiplier (value of 2.0 means range of mean/2 to mean*2).",
    )
    # Momentum bounds and initialization
    parser.add_argument(
        "--momentum_min", type=float, default=0.7, help="Minimum allowable momentum."
    )
    parser.add_argument(
        "--momentum_max", type=float, default=0.9, help="Maximum allowable momentum."
    )
    parser.add_argument(
        "--momentum_init", type=float, default=0.8, help="Initial momentum mean."
    )
    parser.add_argument(
        "--momentum_std_init",
        type=float,
        default=0.1,
        help="Initial momentum standard deviation.",
    )
    # Weight decay bounds and initialization
    parser.add_argument(
        "--weight_decay_min",
        type=float,
        default=1e-4,
        help="Minimum allowable weight decay.",
    )
    parser.add_argument(
        "--weight_decay_max",
        type=float,
        default=5e-3,
        help="Maximum allowable weight decay.",
    )
    parser.add_argument(
        "--weight_decay_init",
        type=float,
        default=1e-3,
        help="Initial weight decay mean in log space.",
    )
    parser.add_argument(
        "--weight_decay_std_init",
        type=float,
        default=5.0,
        help="Initial weight decay std multiplier (value of 5.0 means range of mean/5 to mean*5).",
    )
    # Random flip bounds and initialization
    parser.add_argument(
        "--randomflip_min",
        type=float,
        default=0.25,
        help="Minimum allowable random flip probability.",
    )
    parser.add_argument(
        "--randomflip_max",
        type=float,
        default=0.75,
        help="Maximum allowable random flip probability.",
    )
    parser.add_argument(
        "--randomflip_init",
        type=float,
        default=0.5,
        help="Initial random flip probability mean.",
    )
    parser.add_argument(
        "--randomflip_std_init",
        type=float,
        default=0.25,
        help="Initial random flip probability standard deviation.",
    )
    args = parser.parse_args()

    main(args)
