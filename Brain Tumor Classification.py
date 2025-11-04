# Brain Tumor Classification (robust) — RGB force + graphs + logs
import os, sys, json, subprocess, traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------- PATHS / CONFIG -----------------
DATA_ROOT   = "/Users/anshulshukla/Downloads/archive"
TRAIN_DIR   = os.path.join(DATA_ROOT, "Training")
TEST_DIR    = os.path.join(DATA_ROOT, "Testing")

OUT_DIR = "/Users/anshulshukla/python/Computer vision"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_SIZE  = (224, 224)
BATCH_SIZE  = 8
EPOCHS      = 1           # keep 1 to verify outputs quickly; increase later
LR          = 1e-4
NUM_WORKERS = 0           # macOS
PERSISTENT  = False

STAMP      = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_PREFIX = os.path.join(OUT_DIR, f"brain_tumor_{STAMP}")
# ---------------------------------------------------

def save_and_open(path):
    abs_path = os.path.abspath(path)
    print(f"📁 Saved: {abs_path}")
    if sys.platform == "darwin":
        try: subprocess.run(["open", abs_path], check=False)
        except Exception as e: print(f"   (Could not auto-open: {e})")

def smoke_test():
    txt_path = os.path.join(OUT_DIR, "write_check.txt")
    with open(txt_path, "w") as f:
        f.write("If you can see this file, OUT_DIR is writable.\n")
    save_and_open(txt_path)

    plt.figure()
    plt.plot([0,1,2],[0,1,0]); plt.title("Path Test Plot")
    png_path = os.path.join(OUT_DIR, "path_test.png")
    plt.savefig(png_path, bbox_inches="tight"); plt.close()
    save_and_open(png_path)

def build_loaders():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Force every image to RGB to avoid 1-channel errors
    to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))

    tf_train = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        to_rgb,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_test = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tf_train)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=tf_test)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, persistent_workers=PERSISTENT)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, persistent_workers=PERSISTENT)
    return train_ds, test_ds, train_dl, test_dl

def build_model(num_classes, device):
    model = models.mobilenet_v3_large(weights=None)  # no downloads
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model.to(device)

def plot_training_curves(history, out_prefix):
    if len(history["loss"]) == 0:
        history["loss"] = [0.0]
        history["acc"]  = [0.0]
    xs = np.arange(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(xs, history["loss"]); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss per Epoch"); plt.grid(True, linestyle="--", alpha=0.4)
    p = f"{out_prefix}_loss.png"; plt.savefig(p, bbox_inches="tight"); plt.close(); save_and_open(p)

    plt.figure()
    plt.plot(xs, history["acc"]); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Training Accuracy per Epoch"); plt.grid(True, linestyle="--", alpha=0.4)
    p = f"{out_prefix}_acc.png"; plt.savefig(p, bbox_inches="tight"); plt.close(); save_and_open(p)

def plot_confusion_matrix(cm, classes, out_prefix):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            plt.text(j, i, str(val),
                     ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout()
    p = f"{out_prefix}_confusion_matrix.png"; plt.savefig(p, bbox_inches="tight"); plt.close(); save_and_open(p)

def train(model, train_dl, device):
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=LR)
    history = {"loss": [], "acc": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = cor = 0; loss_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out  = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            cor      += (out.argmax(1) == y).sum().item()
            tot      += y.size(0)

        epoch_loss = (loss_sum / tot) if tot > 0 else 0.0
        epoch_acc  = (cor / tot) if tot > 0 else 0.0
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        print(f"Epoch {ep}/{EPOCHS} — loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")
    return history

def evaluate(model, test_dl, classes, device):
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            out = model(x)
            yp.extend(out.argmax(1).cpu().tolist())
            yt.extend(y.tolist())
    if len(yt) == 0:
        return {"accuracy": 0.0}, np.zeros((len(classes), len(classes)), dtype=int)
    report = classification_report(yt, yp, target_names=classes, output_dict=True)
    cm = confusion_matrix(yt, yp)
    return report, cm

def save_report(report, out_prefix):
    jpath = f"{out_prefix}_report.json"
    tpath = f"{out_prefix}_report.txt"
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    lines = ["Per-class metrics (precision / recall / f1 / support):\n"]
    for k, v in report.items():
        if k in ("accuracy", "macro avg", "weighted avg"): continue
        if isinstance(v, dict) and {"precision","recall","f1-score","support"} <= set(v.keys()):
            lines.append(f"{k:>12s}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1-score']:.3f} N={int(v['support'])}")
    lines.append("\nAverages:")
    if "accuracy" in report:
        lines.append(f"accuracy   : {report['accuracy']:.3f}")
    for name in ("macro avg", "weighted avg"):
        if name in report:
            v = report[name]
            lines.append(f"{name:>12s}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1-score']:.3f} N={int(v['support'])}")
    with open(tpath, "w") as f:
        f.write("\n".join(lines))
    save_and_open(jpath); save_and_open(tpath)

def main():
    smoke_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    train_ds, test_ds, train_dl, test_dl = build_loaders()
    classes = train_ds.classes
    print("Classes:", classes)
    print(f"Train images: {len(train_ds)} | Test images: {len(test_ds)}")
    print(f"Train batches: {len(train_dl)} | Test batches: {len(test_dl)}")

    model = build_model(len(classes), device)

    history = {"loss": [], "acc": []}
    try:
        history = train(model, train_dl, device)
    except Exception as e:
        print("❌ Training error:")
        traceback.print_exc()

    # Save model regardless
    mpath = f"{OUT_PREFIX}.pt"
    try:
        torch.save(model.state_dict(), mpath)
        save_and_open(mpath)
    except Exception as e:
        print("❌ Could not save model:", e)

    # Save history csv regardless
    try:
        csv_path = f"{OUT_PREFIX}_history.csv"
        with open(csv_path, "w") as f:
            f.write("epoch,loss,acc\n")
            for i, (l, a) in enumerate(zip(history.get("loss", []), history.get("acc", [])), start=1):
                f.write(f"{i},{float(l):.6f},{float(a):.6f}\n")
        save_and_open(csv_path)
    except Exception as e:
        print("❌ Could not save history CSV:", e)

    # Curves (even if empty, we save placeholder curves)
    try:
        plot_training_curves(history, OUT_PREFIX)
    except Exception as e:
        print("❌ Could not plot curves:", e)

    # Evaluation + CM
    try:
        report, cm = evaluate(model, test_dl, classes, device)
        save_report(report, OUT_PREFIX)
        plot_confusion_matrix(cm, classes, OUT_PREFIX)
    except Exception as e:
        print("❌ Evaluation/CM error:")
        traceback.print_exc()

    print("\n✅ All outputs saved (or attempted) under:")
    print(os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
