import os
import numpy as np
from sklearn.model_selection import train_test_split

# =====================
# Fixed random seed
# =====================
SEED = 42

# =====================
# Load processed dataset
# =====================
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

os.makedirs("data/splits", exist_ok=True)

# =====================
# 1. Create test set (10%)
# =====================
X_remain, X_test, y_remain, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=SEED,
    stratify=y
)

# =====================
# 2. Split remaining data into train and validation sets
# Final ratio: Train 70%, Validation 20%, Test 10%
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X_remain,
    y_remain,
    test_size=2/9,
    random_state=SEED,
    stratify=y_remain
)

# =====================
# Save dataset splits
# =====================
np.save("data/splits/X_train.npy", X_train)
np.save("data/splits/y_train.npy", y_train)

np.save("data/splits/X_val.npy", X_val)
np.save("data/splits/y_val.npy", y_val)

np.save("data/splits/X_test.npy", X_test)
np.save("data/splits/y_test.npy", y_test)

print("Dataset split completed.")
print(f"Train: {len(X_train)}")
print(f"Val  : {len(X_val)}")
print(f"Test : {len(X_test)}")
