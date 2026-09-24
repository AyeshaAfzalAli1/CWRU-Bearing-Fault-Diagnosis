import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# ======================
# Configuration
# ======================
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

WINDOW_SIZE = 1024
STEP_SIZE = 1024

LABEL_MAP = {
    "Normal": 0,
    "Ball": 1,
    "Inner Race": 2,
    "Outer Race": 3,
}

os.makedirs(OUT_DIR, exist_ok=True)


# ======================
# Utility Functions
# ======================

def find_de_signal(mat_dict):
    """
    Find the Drive-End vibration signal from a CWRU .mat file.
    CWRU Drive-End vibration variables typically end with 'DE_time'.
    """
    for key, value in mat_dict.items():
        if key.endswith("DE_time"):
            signal = np.squeeze(value)

            if signal.ndim == 1 and signal.size >= WINDOW_SIZE:
                return signal.astype(np.float32)

    return None


def slice_signal(signal, window_size, step_size):
    """
    Divide a vibration signal into fixed-length windows.
    """
    segments = []

    for start in range(
        0,
        len(signal) - window_size + 1,
        step_size
    ):
        segment = signal[start:start + window_size]
        segments.append(segment)

    return segments


# ======================
# Main Processing
# ======================

X_all = []
y_all = []


# ---------- Normal ----------
normal_dir = os.path.join(RAW_DIR, "CWRU_Normal")

print("[INFO] Loading Normal data...")

for fname in tqdm(sorted(os.listdir(normal_dir))):

    if not fname.endswith(".mat"):
        continue

    file_path = os.path.join(normal_dir, fname)
    mat = sio.loadmat(file_path)

    signal = find_de_signal(mat)

    if signal is None:
        print(f"[WARN] No Drive-End signal found in {fname}")
        continue

    segments = slice_signal(
        signal,
        WINDOW_SIZE,
        STEP_SIZE
    )

    X_all.extend(segments)
    y_all.extend(
        [LABEL_MAP["Normal"]] * len(segments)
    )


# ---------- Fault Data ----------
fault_root = os.path.join(
    RAW_DIR,
    "CWRU_12K_DE"
)

for fault_type in [
    "Ball",
    "Inner Race",
    "Outer Race"
]:

    print(f"[INFO] Loading {fault_type} data...")

    fault_label = LABEL_MAP[fault_type]
    fault_dir = os.path.join(
        fault_root,
        fault_type
    )

    for root, _, files in os.walk(fault_dir):

        for fname in sorted(files):

            if not fname.endswith(".mat"):
                continue

            file_path = os.path.join(
                root,
                fname
            )

            mat = sio.loadmat(file_path)

            signal = find_de_signal(mat)

            if signal is None:
                print(
                    f"[WARN] No Drive-End signal found in {fname}"
                )
                continue

            segments = slice_signal(
                signal,
                WINDOW_SIZE,
                STEP_SIZE
            )

            X_all.extend(segments)

            y_all.extend(
                [fault_label] * len(segments)
            )


# ======================
# Convert to NumPy Arrays
# ======================

X_all = np.array(
    X_all,
    dtype=np.float32
)

y_all = np.array(
    y_all,
    dtype=np.int64
)


# ======================
# Save Processed Dataset
# ======================

np.save(
    os.path.join(OUT_DIR, "X.npy"),
    X_all
)

np.save(
    os.path.join(OUT_DIR, "y.npy"),
    y_all
)


# ======================
# Dataset Summary
# ======================

print("\n========== Dataset Summary ==========")

print(f"X shape: {X_all.shape}")
print(f"y shape: {y_all.shape}")
print(f"Total samples: {len(y_all)}")

for name, idx in LABEL_MAP.items():
    count = (y_all == idx).sum()
    print(f"{name:<12}: {count}")

print("=====================================")
