from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data_pipeline import HousingDataProcessor
from experiment import (
    HomotopyBOHB,
    evaluate_config,
    seed_everything,
    train_final_and_evaluate,
)


SEEDS = [42, 123, 2024, 3407, 5189]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-agent ER-TRPO house-price prediction pipeline")
    parser.add_argument("--data-path", required=True, help="Path to HouseTS or UKHP table (CSV/Parquet/Excel).")
    parser.add_argument("--dataset", required=True, choices=["house_ts", "ukhp"])
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--date-column", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--labeled-fraction", type=float, default=0.30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--search-initial-configs", type=int, default=9)
    parser.add_argument("--max-search-rows", type=int, default=50000)
    parser.add_argument("--single-seed", type=int, default=None, help="Run one seed instead of the five paper seeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.single_seed] if args.single_seed is not None else SEEDS
    frame = HousingDataProcessor.load_table(args.data_path)

    run_results = []
    for seed in seeds:
        seed_everything(seed)
        processor = HousingDataProcessor(
            target_column=args.target_column,
            date_column=args.date_column,
            group_column=args.group_column,
            sequence_length=args.sequence_length,
        )
        split = processor.prepare(frame, seed=seed)

        def objective(config, budget, rho):
            return evaluate_config(
                config=config,
                split=split,
                budget=budget,
                rho=rho,
                device=args.device,
                seed=seed,
                labeled_fraction=args.labeled_fraction,
                max_search_rows=args.max_search_rows,
            )

        optimizer = HomotopyBOHB(
            evaluator=objective,
            seed=seed,
            min_budget=10,
            max_budget=90,
            eta=3,
            initial_configs=args.search_initial_configs,
        )
        best_config = optimizer.optimize()
        trainer, metrics = train_final_and_evaluate(
            best_config,
            split,
            device=args.device,
            seed=seed,
            labeled_fraction=args.labeled_fraction,
        )

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.state_dict(), seed_dir / "price_feature_model.pt")
        torch.save(trainer.agent1.policy.state_dict(), seed_dir / "agent1_policy.pt")
        torch.save(trainer.agent2.policy.state_dict(), seed_dir / "agent2_policy.pt")
        with open(seed_dir / "best_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(best_config), f, indent=2)
        with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)
        run_results.append(asdict(metrics))
        print(f"Seed {seed}: {asdict(metrics)}")

    keys = ["rmse", "mape", "rmsle", "r2"]
    summary = {
        key: {
            "mean": float(np.mean([r[key] for r in run_results])),
            "std": float(np.std([r[key] for r in run_results], ddof=1)) if len(run_results) > 1 else 0.0,
        }
        for key in keys
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
