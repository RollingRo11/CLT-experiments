#!/usr/bin/env python
"""Run all CLT cross-layer experiments."""

import os
import sys

# Add experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))

import torch
from exp1_weight_analysis import main as exp1_main
from exp2_ablation import main as exp2_main
from exp3_feature_analysis import main as exp3_main
from exp4_distance_analysis import main as exp4_main
from exp5_timetravel import main as exp5_main
from exp6_shortcut import main as exp6_main
from exp7_targeting import main as exp7_main
from exp8_gradient import main as exp8_main


def main():
    torch.set_grad_enabled(False)

    print("\n" + "=" * 70)
    print("CLT CROSS-LAYER CONNECTIONS: FULL EXPERIMENT SUITE")
    print("=" * 70 + "\n")

    results = {}

    print("\n[1/8] Running Experiment 1: Weight Analysis...")
    results["exp1"] = exp1_main()

    print("\n[2/8] Running Experiment 2: Ablation Study...")
    results["exp2"] = exp2_main()

    print("\n[3/8] Running Experiment 3: Feature Analysis...")
    results["exp3"] = exp3_main()

    print("\n[4/8] Running Experiment 4: Distance Analysis...")
    results["exp4"] = exp4_main()

    print("\n[5/8] Running Experiment 5: Time Travel Logit Lens...")
    results["exp5"] = exp5_main()

    print("\n[6/8] Running Experiment 6: Residual Alignment...")
    results["exp6"] = exp6_main()

    print("\n[7/8] Running Experiment 7: Component Targeting...")
    results["exp7"] = exp7_main()

    print("\n[8/8] Running Experiment 8: Gradient Alignment...")
    # Exp 8 needs grad enabled, handled inside its main or we can set it here
    torch.set_grad_enabled(True)
    results["exp8"] = exp8_main()
    torch.set_grad_enabled(False)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("\nFigures saved to: figures/")
    print("See REPORT.md for full analysis")

    return results


if __name__ == "__main__":
    main()
