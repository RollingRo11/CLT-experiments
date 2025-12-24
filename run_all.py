#!/usr/bin/env python
"""Run all CLT cross-layer experiments."""

import sys
import os

# Add experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))

import torch


def main():
    torch.set_grad_enabled(False)

    print("\n" + "=" * 70)
    print("CLT CROSS-LAYER CONNECTIONS: FULL EXPERIMENT SUITE")
    print("=" * 70 + "\n")

    # Run experiments
    from exp1_weight_analysis import main as exp1_main
    from exp2_ablation import main as exp2_main
    from exp3_feature_analysis import main as exp3_main
    from exp4_distance_analysis import main as exp4_main

    results = {}

    print("\n[1/4] Running Experiment 1: Weight Analysis...")
    results["exp1"] = exp1_main()

    print("\n[2/4] Running Experiment 2: Ablation Study...")
    results["exp2"] = exp2_main()

    print("\n[3/4] Running Experiment 3: Feature Analysis...")
    results["exp3"] = exp3_main()

    print("\n[4/4] Running Experiment 4: Distance Analysis...")
    results["exp4"] = exp4_main()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("\nFigures saved to: figures/")
    print("See REPORT.md for full analysis")

    return results


if __name__ == "__main__":
    main()
