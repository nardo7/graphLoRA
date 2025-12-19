import glob
import json
import os

import matplotlib.pyplot as plt
import torch


def get_best_roc_auc_huggingface(checkpoint_dir):
    """
    Extract the best ROC-AUC from HuggingFace trainer_state.json files.
    """
    best_roc_auc = -1.0
    best_checkpoint = None

    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

    if not checkpoint_dirs:
        return None, None

    for ckpt_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)

            # Look through log_history for best eval_roc_auc
            for entry in trainer_state.get("log_history", []):
                if "eval_roc_auc" in entry:
                    roc_auc = entry["eval_roc_auc"]
                    if roc_auc > best_roc_auc:
                        best_roc_auc = roc_auc
                        best_checkpoint = ckpt_dir

    return best_roc_auc, best_checkpoint


def get_best_roc_auc_own_code(checkpoint_dir):
    """
    Extract the best ROC-AUC from own_code trainer_state.pt files.
    """
    best_roc_auc = -1.0
    best_checkpoint = None

    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch-*"))

    if not checkpoint_dirs:
        return None, None

    for ckpt_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(ckpt_dir, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            training_state = torch.load(trainer_state_path, map_location="cpu")
            roc_auc = training_state.get("best_roc_auc", -1.0)

            # Also check in metrics if available
            if "metrics" in training_state and "roc_auc" in training_state["metrics"]:
                roc_auc = max(roc_auc, training_state["metrics"]["roc_auc"])

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_checkpoint = ckpt_dir

    return best_roc_auc, best_checkpoint


def main():
    # Define all runs to analyze
    runs = {
        "HF Full": "./graph-classification/full/molhiv",
        "HF LoRA": "./graph-classification/lora/molhiv",
        "Own Code Full": "./graph-classification/own_code/full/molhiv",
        "Own Code Regularized": "./graph-classification/own_code/regularized/molhiv",
    }

    # Collect best ROC-AUC for each run
    results = {}

    for run_name, run_path in runs.items():
        if not os.path.exists(run_path):
            print(f"Warning: Path does not exist: {run_path}")
            continue

        # Determine which extraction method to use based on path
        if "own_code" in run_path:
            best_roc_auc, best_checkpoint = get_best_roc_auc_own_code(run_path)
        else:
            best_roc_auc, best_checkpoint = get_best_roc_auc_huggingface(run_path)

        if best_roc_auc is not None and best_roc_auc > 0:
            results[run_name] = best_roc_auc
            print(f"{run_name}: {best_roc_auc:.4f} (from {best_checkpoint})")
        else:
            print(f"{run_name}: No valid ROC-AUC found")

    # Create bar chart
    if not results:
        print("No results to plot!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    run_names = list(results.keys())
    roc_aucs = list(results.values())

    # Create bars with different colors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(range(len(run_names)), roc_aucs, color=colors[: len(run_names)])

    # Customize plot
    ax.set_xlabel("Training Run", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC-AUC Score", fontsize=12, fontweight="bold")
    ax.set_title("Best ROC-AUC Score by Training Run", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(run_names)))
    ax.set_xticklabels(run_names, rotation=45, ha="right")
    ax.set_ylim([0, 1.0])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, roc_aucs)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("best_roc_auc_comparison.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved as 'best_roc_auc_comparison.png'")
    plt.show()


if __name__ == "__main__":
    main()
