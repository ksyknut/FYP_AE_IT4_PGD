import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# Load and preprocess data: convert to datetime index, calculate log returns, handle index weights
def load_and_convert_data(data_path, index_weights_path=None):
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    if df.empty:
        raise ValueError("matrix_data.csv is empty! Check data generator")
    if 'index' not in df.columns:
        raise KeyError("Missing 'index' column (benchmark returns)")
    
    df_log_return = np.log1p(df)
    df_log_return = df_log_return.replace([np.inf, -np.inf], np.nan).dropna()
    
    index_weights = None
    if index_weights_path:
        try:
            df_weights = pd.read_csv(index_weights_path, index_col=0)
            df_weights.index = pd.to_datetime(df_weights.index)
            index_weights = df_weights.reindex(df_log_return.index).fillna(method='ffill')
            stock_cols = [col for col in df_log_return.columns if col != 'index']
            index_weights = index_weights.reindex(columns=stock_cols).fillna(0.0)
        except:
            index_weights = None  # Use default initialization if load fails
    
    return df_log_return, index_weights


# Initialize parameters: aligned with paper's IT4-PGD design
data_path = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\matrix_data.csv"
index_weights_path = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\index_weights.csv"
df_log_return, index_weights = load_and_convert_data(data_path, index_weights_path)

# Split data into features (X) and target (y, benchmark returns)
dates_full = df_log_return.index
y_full = df_log_return['index'].values.reshape(-1, 1)
X_full = df_log_return.drop(columns=['index']).values
stock_cols = [col for col in df_log_return.columns if col != 'index']
D, N = X_full.shape

# Define test period range
test_start_date = pd.to_datetime('2023-01-01')
test_end_date = pd.to_datetime('2025-10-10')
test_start_idx = np.searchsorted(dates_full, test_start_date, side='left')
test_end_idx = np.searchsorted(dates_full, test_end_date, side='right')
if test_start_idx >= test_end_idx:
    raise ValueError(f"No data in test period! Data range: {dates_full.min().date()} - {dates_full.max().date()}")


# Core parameters: paper-aligned hyperparameters
LOOKBACK_WINDOW = 300    # Training window size
REBALANCE_WINDOW = 60    # Test window size (rebalancing period)
K = 120                  # Sparsity constraint (number of selected assets)
rho = 1.0                # Penalty for sum-to-1 constraint
delta = 0.04             # Turnover constraint (max weight change per period)
h = 0.1                 # Max weight for single asset
max_iter = 2000          # Max iterations for optimization
tol = 1e-9               # Convergence tolerance


# Baseline solver: unconstrained least squares (for comparison)
def solve_baseline(X, y, w_prev=None):
    if len(X) < 10:  # Handle small dataset
        w = np.zeros(X.shape[1])
        w[np.random.choice(X.shape[1], 150, replace=False)] = 1.0 / 150
        return w
    
    D, N = X.shape
    # Objective: minimize prediction error
    def objective(w):
        return np.sum((X @ w - y.flatten()) **2)
    
    # Constraints: sum to 1, non-negative weights
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: w},
    ]
    
    w0 = np.ones(N) / N  # Initial guess: uniform weights
    res = minimize(objective, w0, method='SLSQP', constraints=constraints,
                   options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
    return res.x if res.success else w0


# Base class for IT4-PGD solvers: implements core optimization logic
class PGDSolverBase:
    def __init__(self, X, y, train_end_idx, dates_full, index_weights,
                 K=K, rho=rho, delta=delta, h=h, max_iter=max_iter, tol=tol, x_prev=None):
        self.X = X  # Feature matrix (asset returns)
        self.y = y.flatten()  # Target (benchmark returns)
        self.K = K  # Sparsity: number of non-zero weights
        self.rho = rho  # Penalty for sum(w) != 1
        self.delta = delta  # Max weight change from previous period
        self.h = h  # Upper bound for single asset weight
        self.max_iter = max_iter  # Max optimization iterations
        self.tol = tol  # Convergence threshold
        self.train_end_idx = train_end_idx  # End index of training period
        self.dates_full = dates_full  # Full date list
        self.index_weights = index_weights  # Index constituent weights (for initialization)
        
        self.T, self.N = X.shape  # T: time steps; N: number of assets
        # Initialize weights: use previous weights or baseline if first run
        self.x_prev = x_prev if x_prev is not None else self._get_baseline_init()
        
        # Calculate learning rate: based on Hessian's max eigenvalue
        ATA = self.X.T @ self.X
        ones_matrix = np.ones((self.N, self.N))
        hessian = 2 * ATA + self.rho * ones_matrix  # Hessian of objective function
        eigenvalues = np.linalg.eigvalsh(hessian)
        self.L = eigenvalues.max()  # Lipschitz constant
        self.eta = 0.8 / self.L  # Learning rate (within (0, 1/L])

    # Initialize weights: use top-K assets from index weights or correlation
    def _get_baseline_init(self):
        if self.index_weights is not None and self.train_end_idx < len(self.dates_full):
            train_end_date = self.dates_full[self.train_end_idx]
            if train_end_date in self.index_weights.index:
                w_init = self.index_weights.loc[train_end_date].values
                top_k_idx = np.argsort(w_init)[-self.K:]
                w_init[~np.isin(np.arange(len(w_init)), top_k_idx)] = 0.0
                w_init = w_init / np.sum(w_init) if np.sum(w_init) > 1e-8 else np.ones(self.N)/self.N
                return w_init
        
        # Fallback: top-K assets by correlation with benchmark
        corr_with_index = np.corrcoef(self.X.T, self.y)[-1, :-1]
        top_k_idx = np.argsort(corr_with_index)[-self.K:]
        w_init = np.zeros(self.N) 
        w_init[top_k_idx] = 1.0 / self.K
        return w_init

    # Compute gradient of objective function
    def compute_gradient(self, x):
        residual = self.X @ x - self.y  # Prediction error
        grad_tracking = 2.0 * (self.X.T @ residual)  # Gradient of tracking error
        grad_invest = self.rho * (np.sum(x) - 1.0) * np.ones(self.N)  # Gradient of sum constraint
        return grad_tracking + grad_invest

    # Compute objective function value
    def compute_objective(self, x):
        residual = self.X @ x - self.y
        tracking_error = np.sum(residual** 2)  # Tracking error term
        invest_penalty = (self.rho / 2.0) * (np.sum(x) - 1.0) ** 2  # Sum constraint penalty
        return tracking_error + invest_penalty

    # Project weights to feasible set (satisfy sparsity, turnover, and bound constraints)
    def project_to_constraint(self, z):
        x_proj = np.zeros(self.N)
        # Turnover constraints: zeta = max(0, previous_weight - delta), nu = min(h, previous_weight + delta)
        zeta = np.maximum(0, self.x_prev - self.delta)
        nu = np.minimum(self.h, self.x_prev + self.delta)
        
        # Step 1: Keep assets with zeta > 0 (from previous period)
        Omega = np.where(zeta > 1e-8)[0]
        K_prime = len(Omega)
        for i in Omega:
            x_proj[i] = np.clip(z[i], zeta[i], nu[i])
        
        # Step 2: Add K'' = K - K' new assets (highest correlation with benchmark)
        K_double_prime = self.K - K_prime
        if K_double_prime > 0:
            corr_with_index = np.corrcoef(self.X.T, self.y)[-1, :-1]
            candidate_idx = np.where(np.isclose(x_proj, 0, atol=1e-8))[0]  # Assets not in Omega
            candidate_corr = corr_with_index[candidate_idx]
            sorted_candidate_idx = candidate_idx[np.argsort(-candidate_corr)]  # Sort by correlation
            selected_idx = sorted_candidate_idx[:K_double_prime]
            for i in selected_idx:
                x_proj[i] = np.clip(z[i], 0, self.h)
        
        # Normalize to ensure sum(w) = 1
        sum_proj = np.sum(x_proj)
        if sum_proj > 1e-8:
            x_proj = x_proj / sum_proj
        else:  # Fallback if sum is zero
            top_k_idx = np.argsort(np.corrcoef(self.X.T, self.y)[-1, :-1])[-self.K:]
            x_proj[top_k_idx] = 1.0 / self.K
        
        return x_proj

    # Solve optimization problem via projected gradient descent
    def solve(self, verbose=False):
        x = self.x_prev.copy()
        best_obj = self.compute_objective(x)
        best_x = x.copy()
        
        for iteration in range(self.max_iter):
            grad = self.compute_gradient(x)  # Get gradient
            z = x - self.eta * grad  # Gradient descent step
            x_new = self.project_to_constraint(z)  # Project to feasible set
            
            # Track best solution
            current_obj = self.compute_objective(x_new)
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x_new.copy()
            
            # Check convergence
            if np.sum((x_new - x) **2) <= self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            x = x_new
        
        return best_x


# Paper IT4-PGD solver: inherits base class (original paper logic)
class PaperIT4PGDSolver(PGDSolverBase):
    pass

def solve_paper_it4pgd(X, y, train_end_idx, dates_full, index_weights, x_prev=None):
    solver = PaperIT4PGDSolver(
        X, y, train_end_idx, dates_full, index_weights,
        x_prev=x_prev
    )
    return solver.solve(verbose=False)


# Masked IT4-PGD solver: direction-sensitive gradient (penalizes opposite-direction errors more)
class MaskedIT4PGDSolver(PGDSolverBase):
    def compute_gradient(self, x):
        residual = self.X @ x - self.y  # Prediction error
        # Direction mask: 1 if residual and benchmark same direction, -1 otherwise
        direction_mask = np.sign(residual) * np.sign(self.y)
        # Weight errors: 3x penalty for opposite direction, 0.6x for same direction
        error_weight = np.where(direction_mask < 0, 3.0, 0.6)  
        weighted_residual = residual * error_weight  # Apply weights
        
        grad_tracking = 2.0 * (self.X.T @ weighted_residual)  # Weighted tracking error gradient
        grad_invest = self.rho * (np.sum(x) - 1.0) * np.ones(self.N)
        return grad_tracking + grad_invest

def solve_masked_it4pgd(X, y, train_end_idx, dates_full, index_weights, x_prev=None):
    solver = MaskedIT4PGDSolver(
        X, y, train_end_idx, dates_full, index_weights,
        x_prev=x_prev
    )
    return solver.solve(verbose=False)


# Run rolling window comparison: evaluate all strategies across test periods
def run_comparison():
    # Initialize storage for results
    y_pred_baseline_windows = []
    y_pred_paper_it4_windows = []
    y_pred_masked_it4_windows = []
    mdte_baseline_windows = []
    mdte_paper_it4_windows = []
    mdte_masked_it4_windows = []
    atr_baseline_windows = []
    atr_paper_it4_windows = []
    atr_masked_it4_windows = []
    w_baseline_windows = []
    w_paper_it4_windows = []
    w_masked_it4_windows = []
    window_indices = []
    
    # Initialize previous weights for ATR calculation
    prev_w_baseline = None
    prev_w_paper_it4 = None
    prev_w_masked_it4 = None
    
    # Total ATR (sum over windows)
    total_atr_baseline = 0.0
    total_atr_paper_it4 = 0.0
    total_atr_masked_it4 = 0.0
    
    # Calculate initial training window start
    train_start_current = test_start_idx - LOOKBACK_WINDOW
    if train_start_current < 0:
        raise ValueError(f"Insufficient training data! Need {LOOKBACK_WINDOW} days, got {test_start_idx}")
    
    print("Running rolling window comparison...\n")
    print(f"{'Window':<5} {'Baseline MDTE(bps)':<18} {'Paper IT4 MDTE(bps)':<20} {'Masked-IT4 MDTE(bps)':<22} {'Improvement(%)':<12}")
    print("-" * 85)
    
    # Iterate over windows
    for win in range(100):
        train_start = train_start_current + win * REBALANCE_WINDOW
        train_end = train_start + LOOKBACK_WINDOW
        test_start = train_end
        test_end = test_start + REBALANCE_WINDOW
        
        # Stop if exceed data range
        if test_end > len(dates_full):
            break
        if dates_full[test_end-1] > test_end_date:
            test_end = np.searchsorted(dates_full, test_end_date, side='right')
            if test_start >= test_end:
                break
        
        # Get train/test data
        X_train = X_full[train_start:train_end]
        y_train = y_full[train_start:train_end]
        X_test_w = X_full[test_start:test_end]
        y_test_w = y_full[test_start:test_end]
        T_test = len(y_test_w)
        if T_test < 1:
            break
        
        # Compute weights for each strategy
        w_baseline = solve_baseline(X_train, y_train)
        x_prev_it4 = w_paper_it4_windows[-1] if w_paper_it4_windows else None
        w_paper_it4 = solve_paper_it4pgd(X_train, y_train, train_end, dates_full, index_weights, x_prev=x_prev_it4)
        x_prev_masked = w_masked_it4_windows[-1] if w_masked_it4_windows else None
        w_masked_it4 = solve_masked_it4pgd(X_train, y_train, train_end, dates_full, index_weights, x_prev=x_prev_masked)
        
        # Predict test returns
        y_pred_baseline_w = X_test_w @ w_baseline.reshape(-1, 1)
        y_pred_paper_it4_w = X_test_w @ w_paper_it4.reshape(-1, 1)
        y_pred_masked_it4_w = X_test_w @ w_masked_it4.reshape(-1, 1)
        
        # Calculate ATR (turnover)
        if win == 0:  # First window: no previous weights, ATR=0
            atr_baseline = 0.0
            atr_paper_it4 = 0.0
            atr_masked_it4 = 0.0
        else:  # ATR = sum of absolute weight changes from previous window
            atr_baseline = np.sum(np.abs(w_baseline - prev_w_baseline))
            atr_paper_it4 = np.sum(np.abs(w_paper_it4 - prev_w_paper_it4))
            atr_masked_it4 = np.sum(np.abs(w_masked_it4 - prev_w_masked_it4))
        
        # Update total ATR
        total_atr_baseline += atr_baseline
        total_atr_paper_it4 += atr_paper_it4
        total_atr_masked_it4 += atr_masked_it4
        
        # Calculate MDTE (Mean Daily Tracking Error in basis points)
        error_baseline = y_pred_baseline_w.flatten() - y_test_w.flatten()
        error_paper_it4 = y_pred_paper_it4_w.flatten() - y_test_w.flatten()
        error_masked_it4 = y_pred_masked_it4_w.flatten() - y_test_w.flatten()
        mdte_baseline = (np.linalg.norm(error_baseline, ord=2) / T_test) * 1e4
        mdte_paper_it4 = (np.linalg.norm(error_paper_it4, ord=2) / T_test) * 1e4
        mdte_masked_it4 = (np.linalg.norm(error_masked_it4, ord=2) / T_test) * 1e4
        
        # Store results
        y_pred_baseline_windows.append(y_pred_baseline_w.flatten())
        y_pred_paper_it4_windows.append(y_pred_paper_it4_w.flatten())
        y_pred_masked_it4_windows.append(y_pred_masked_it4_w.flatten())
        w_baseline_windows.append(w_baseline)
        w_paper_it4_windows.append(w_paper_it4)
        w_masked_it4_windows.append(w_masked_it4)
        window_indices.append((train_start, train_end, test_start, test_end))
        mdte_baseline_windows.append(mdte_baseline)
        mdte_paper_it4_windows.append(mdte_paper_it4)
        mdte_masked_it4_windows.append(mdte_masked_it4)
        atr_baseline_windows.append(atr_baseline)
        atr_paper_it4_windows.append(atr_paper_it4)
        atr_masked_it4_windows.append(atr_masked_it4)
        
        # Update previous weights for next window
        prev_w_baseline = w_baseline.copy()
        prev_w_paper_it4 = w_paper_it4.copy()
        prev_w_masked_it4 = w_masked_it4.copy()
        
        # Print window stats
        improvement = (mdte_paper_it4 - mdte_masked_it4)/mdte_paper_it4*100 if mdte_paper_it4>1e-8 else 0
        print(f"{win:<5} {mdte_baseline:<18.2f} {mdte_paper_it4:<20.2f} {mdte_masked_it4:<22.2f} {improvement:>+10.2f}%")
    
    # Concatenate results across windows
    y_pred_baseline_all = np.concatenate(y_pred_baseline_windows) if y_pred_baseline_windows else np.array([])
    y_pred_paper_it4_all = np.concatenate(y_pred_paper_it4_windows) if y_pred_paper_it4_windows else np.array([])
    y_pred_masked_it4_all = np.concatenate(y_pred_masked_it4_windows) if y_pred_masked_it4_windows else np.array([])
    
    # Get test dates and actual returns
    first_test_idx = window_indices[0][2] if window_indices else 0
    last_test_idx = window_indices[-1][3] if window_indices else 0
    y_test_all = y_full[first_test_idx:last_test_idx] if first_test_idx < last_test_idx else np.array([])
    dates_test_all = dates_full[first_test_idx:last_test_idx] if first_test_idx < last_test_idx else pd.DatetimeIndex([])
    
    # Ensure equal length
    min_len = min(len(y_test_all), len(y_pred_baseline_all), len(y_pred_paper_it4_all), len(y_pred_masked_it4_all))
    
    return {
        'y_test': y_test_all[:min_len].flatten(),
        'y_pred_baseline': y_pred_baseline_all[:min_len],
        'y_pred_paper_it4': y_pred_paper_it4_all[:min_len],
        'y_pred_masked_it4': y_pred_masked_it4_all[:min_len],
        'dates': dates_test_all[:min_len],
        'mdte_baseline_windows': mdte_baseline_windows,
        'mdte_paper_it4_windows': mdte_paper_it4_windows,
        'mdte_masked_it4_windows': mdte_masked_it4_windows,
        'atr_baseline_windows': atr_baseline_windows,
        'atr_paper_it4_windows': atr_paper_it4_windows,
        'atr_masked_it4_windows': atr_masked_it4_windows,
        'total_atr_baseline': total_atr_baseline,
        'total_atr_paper_it4': total_atr_paper_it4,
        'total_atr_masked_it4': total_atr_masked_it4,
        'num_windows': len(window_indices),
        'w_paper_it4': w_paper_it4_windows,
    }


# Execute comparison and visualize results
print("="*90)
print("Paper-aligned IT4-PGD Comparison")
print("="*90 + "\n")
results = run_comparison()

# Calculate overall metrics
mdte_baseline_avg = np.mean(results['mdte_baseline_windows']) if results['mdte_baseline_windows'] else 0
mdte_paper_it4_avg = np.mean(results['mdte_paper_it4_windows']) if results['mdte_paper_it4_windows'] else 0
mdte_masked_it4_avg = np.mean(results['mdte_masked_it4_windows']) if results['mdte_masked_it4_windows'] else 0
avg_sparsity_it4 = np.mean([np.count_nonzero(w) for w in results['w_paper_it4']]) if results['w_paper_it4'] else 0

print("\n" + "="*90)
print("Overall Results")
print("="*90 + "\n")
print(f"{'Metric':<25} {'Baseline':<20} {'Paper IT4-PGD':<20} {'Masked-IT4-PGD':<20}")
print("-" * 85)
print(f"{'Avg MDTE (bps)':<25} {mdte_baseline_avg:<20.2f} {mdte_paper_it4_avg:<20.2f} {mdte_masked_it4_avg:<20.2f}")
print(f"{'Total ATR':<25} {results['total_atr_baseline']:<20.4f} {results['total_atr_paper_it4']:<20.4f} {results['total_atr_masked_it4']:<20.4f}")
print(f"{'Avg Sparsity':<25} {'-':<20} {avg_sparsity_it4:<20.1f} {'-':<20}")
print()

# Calculate improvement rates
paper_vs_baseline_mdte = (mdte_baseline_avg - mdte_paper_it4_avg)/mdte_baseline_avg*100 if mdte_baseline_avg>1e-8 else 0
masked_vs_paper_mdte = (mdte_paper_it4_avg - mdte_masked_it4_avg)/mdte_paper_it4_avg*100 if mdte_paper_it4_avg>1e-8 else 0
print(f"Paper IT4-PGD vs Baseline: MDTE reduced by {paper_vs_baseline_mdte:+.2f}%")
print(f"Masked-IT4-PGD vs Paper IT4-PGD: MDTE reduced by {masked_vs_paper_mdte:+.2f}%")
print(f"Number of windows analyzed: {results['num_windows']}\n")


# Plot results
print("Generating visualizations...\n")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Figure 1: Cumulative returns
ax = axes[0]
if len(results['dates']) > 0:
    cum_index = np.exp(np.cumsum(results['y_test'])) - 1
    cum_baseline = np.exp(np.cumsum(results['y_pred_baseline'])) - 1
    cum_paper_it4 = np.exp(np.cumsum(results['y_pred_paper_it4'])) - 1
    cum_masked_it4 = np.exp(np.cumsum(results['y_pred_masked_it4'])) - 1
    
    ax.plot(results['dates'], cum_index, label='Index (Benchmark)', linewidth=3, color='black', alpha=0.9)
    ax.plot(results['dates'], cum_baseline, label='Baseline', linewidth=2.5, linestyle='--', color='red', alpha=0.8)
    ax.plot(results['dates'], cum_paper_it4, label='Paper IT4-PGD', linewidth=2.5, linestyle='-', color='steelblue', alpha=0.8)
    ax.plot(results['dates'], cum_masked_it4, label='Masked-IT4-PGD', linewidth=2.5, linestyle='-.', color='green', alpha=0.8)

ax.set_title('Figure 1: Cumulative Return Tracking (Paper Logic)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return', fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Figure 2: Window-level MDTE
ax = axes[1]
window_ids = np.arange(results['num_windows'])
if len(window_ids) > 0:
    ax.plot(window_ids, results['mdte_baseline_windows'], 'o-', label='Baseline', linewidth=2.5, markersize=6, color='red')
    ax.plot(window_ids, results['mdte_paper_it4_windows'], 's-', label='Paper IT4-PGD', linewidth=2.5, markersize=6, color='steelblue')
    ax.plot(window_ids, results['mdte_masked_it4_windows'], '^-', label='Masked-IT4-PGD', linewidth=2.5, markersize=6, color='green')
    ax.axhline(y=mdte_baseline_avg, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=mdte_paper_it4_avg, color='steelblue', linestyle='--', alpha=0.5)
    ax.axhline(y=mdte_masked_it4_avg, color='green', linestyle='--', alpha=0.5)

ax.set_title('Figure 2: Window-level MDTE Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Window ID', fontsize=12)
ax.set_ylabel('MDTE (Basis Points)', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# Figure 3: Window-level ATR
ax = axes[2]
if len(window_ids) > 0:
    ax.plot(window_ids, results['atr_baseline_windows'], 'o-', label='Baseline', linewidth=2.5, markersize=6, color='red')
    ax.plot(window_ids, results['atr_paper_it4_windows'], 's-', label='Paper IT4-PGD', linewidth=2.5, markersize=6, color='steelblue')
    ax.plot(window_ids, results['atr_masked_it4_windows'], '^-', label='Masked-IT4-PGD', linewidth=2.5, markersize=6, color='green')
    ax.text(0.02, 0.95, f'Total ATR:\nBaseline: {results["total_atr_baseline"]:.2f}\nPaper IT4: {results["total_atr_paper_it4"]:.2f}\nMasked-IT4: {results["total_atr_masked_it4"]:.2f}',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top', fontsize=10)

ax.set_title('Figure 3: Window-level ATR Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Window ID', fontsize=12)
ax.set_ylabel('ATR (Turnover Rate)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = r"F:\University\FYP\FYP_AE_IT4_PGD\test_results\init_results.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✓ Results saved to: {save_path}\n")
plt.close()

print("="*90)
print("✓ Execution completed")
print("="*90)