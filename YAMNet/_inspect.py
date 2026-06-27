import numpy as np
import collections

def acc(yt, yp):
    return float(np.mean(npt := np.array(yt) == np.array(yp)))

def confusion(yt, yp, k):
    yt = np.array(yt); yp = np.array(yp)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return cm

print("=== test_predictions.npz ===")
d = np.load("outputs/yamnet/test_predictions.npz", allow_pickle=True)
classes = list(d["classes"])
print("classes:", classes)
yt = d["y_true"]; yp = d["y_pred"]
print("y_true:", yt.tolist())
print("y_pred:", yp.tolist())
print("acc:", float(np.mean(yt == yp)))
cm = confusion(yt, yp, len(classes))
print("confusion matrix (rows=true, cols=pred), labels:", classes)
print(cm)
# per-class P/R/F1
print("per-class (precision/recall/f1):")
for i, c in enumerate(classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"  {c}: P={p:.3f} R={r:.3f} F1={f:.3f}  (support={cm[i,:].sum()})")

print("\n=== noise_results.npz ===")
n = np.load("outputs/yamnet/noise_results.npz", allow_pickle=True)
print("snr_tiers:", list(n["snr_tiers"]))
print("acc per tier:", list(n["acc"]))
print("y_true:", n["y_true"].tolist())
for k in ["preds_clean", "preds_5dB", "preds_0dB", "preds_n5dB"]:
    preds = n[k]
    yt2 = n["y_true"]
    print(f"  {k}: preds={preds.tolist()}  acc={float(np.mean(preds == yt2)):.4f}")

print("\n=== embeddings.npz ===")
e = np.load("outputs/yamnet/embeddings.npz", allow_pickle=True)
print("X shape:", e["X"].shape, "y shape:", e["y"].shape)
print("per-class counts:", dict(collections.Counter(e["y"].tolist())))
