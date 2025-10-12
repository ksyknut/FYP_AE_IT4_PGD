import os
import glob
import pandas as pd
import numpy as np

# Paths
COMPONENTS_DIR = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\components"
INDEX_CSV      = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\sp500_index.csv"

def load_dataframe(path, date_format="%Y-%m-%d"):
    """
    Load a CSV with the first column as date, convert all other columns to numeric,
    drop rows with NaNs, and return a DataFrame indexed by datetime.
    """
    # Read everything as string
    df = pd.read_csv(path, dtype=str, header=0)
    # Parse first column as datetime index
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format=date_format, errors="coerce")
    df = df.set_index(df.columns[0])
    # Convert all other columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with any NaN
    df = df.dropna(how="any")
    return df

# 1. Load and process index data
idx_df = load_dataframe(INDEX_CSV, date_format="%Y-%m-%d")
idx_df["log_ret"] = np.log(idx_df["Close"] / idx_df["Close"].shift(1))
y_series = idx_df["log_ret"].dropna()

# 2. Load and process component stock data
file_list = glob.glob(os.path.join(COMPONENTS_DIR, "*.csv"))
rets = []
symbols = []
for file_path in file_list:
    df = load_dataframe(file_path, date_format="%Y-%m-%d")
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    rets.append(df["log_ret"])
    symbols.append(os.path.splitext(os.path.basename(file_path))[0])

# 3. Align dates and build return matrix A
A_df = pd.concat(rets, axis=1)
A_df.columns = symbols
# Ensure numeric dtypes for future versions
A_df = A_df.infer_objects()
data = pd.concat([y_series, A_df], axis=1).dropna().infer_objects()
y = data["log_ret"]
A = data[symbols].values  # shape (T, n)
T, n = A.shape

# 4. Compute initial reference portfolio (equal weights)
x0 = np.ones(n) / n

# 5. Attention computation
def compute_attention(A, y, x_ref, tau=1.0):
    # Absolute residuals
    r = np.abs(A.dot(x_ref) - y.values)
    # Stabilize & scale
    z = (r / tau) - np.max(r / tau)
    # Inverted sigmoid mapping to (0,1)
    return 1 - np.exp(z) / (np.exp(z) + 1)

# 6. AE-IT-GD algorithm
def ae_it_gd(A, y, x0, eta=0.01, tau=0.5, M=10, max_iter=200, tol=1e-6):
    x = x0.copy()
    alpha = compute_attention(A, y, x, tau)
    for t in range(max_iter):
        resid = A.dot(x) - y.values
        grad = 2.0 * A.T.dot(alpha * resid)
        x_new = x - eta * grad
        x_new = np.maximum(x_new, 0)           # clip to x>=0
        if x_new.sum() > 0:
            x_new = x_new / x_new.sum()       # re-normalize to sum = 1
        else:
            x_new = x.copy()                  # 若全为0，退回旧值以避免除0

        # Dynamic attention update
        if (t + 1) % M == 0:
            alpha = compute_attention(A, y, x_new, tau)
        # Convergence check
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, alpha

# 7. Run AE-IT-GD
x_final, alpha_final = ae_it_gd(A, y, x0, eta=0.01, tau=0.5, M=10, max_iter=200, tol=1e-6)

# 8. Display results
print("Final portfolio weights:")
for sym, weight in zip(symbols, x_final):
    print(f"{sym}: {weight:.6f}")

print("\nTop 5 lowest-attention dates (likely noisy):")
alpha_series = pd.Series(alpha_final, index=y.index)
print(alpha_series.nsmallest(5))
