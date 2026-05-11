import pandas as pd
import matplotlib.pyplot as plt

# paths to results.csv for each model
models = {
    "Tag Detection": "archive/runs/tag_detection/train4/results.csv",
    "Direction Classifier": "archive/runs/angle_detection/train/results.csv",
    "Digit Recognition": "archive/runs/number_detection/bee_digit_v3/results.csv",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, path) in zip(axes, models.items()):
    df = pd.read_csv(path)
    
    # find the loss columns
    train_col = [c for c in df.columns if "train" in c and "loss" in c][0]
    val_col = [c for c in df.columns if "val" in c and "loss" in c][0]
    
    ax.plot(df["epoch"], df[train_col], label="Train Loss", color="#1f77b4")
    ax.plot(df["epoch"], df[val_col], label="Val Loss", color="#ff7f0e", linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()