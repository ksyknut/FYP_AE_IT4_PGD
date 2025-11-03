"""
=====================================================================
AE-IT4-PGD vs IT4-PGD vs Index 对比分析
- 数据加载: 全部加载 (所有历史数据)
- 模型训练: 使用全部数据训练
- 图表显示: 自动检测实际数据范围（从最早日期开始）
- 累积收益: 直接显示 Accumulated Return (不normalize)
- MDTE/波动率: 按月显示
- 6个子图合并为一张
=====================================================================
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import warnings
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 第一部分：数据加载与预处理
# =====================================================================

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, components_dir: str, index_csv: str):
        self.components_dir = components_dir
        self.index_csv = index_csv
    
    @staticmethod
    def load_dataframe(path: str, date_format: str = "%Y-%m-%d") -> pd.DataFrame:
        """加载CSV文件"""
        df = pd.read_csv(path, dtype=str, header=0)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format=date_format, errors="coerce")
        df = df.set_index(df.columns[0])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="any")
        return df
    
    def load_data(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, pd.DatetimeIndex]:
        """加载全部数据"""
        print("=" * 80)
        print("加载数据 (全部历史数据)...")
        print("=" * 80)
        
        # 加载指数
        idx_df = self.load_dataframe(self.index_csv, date_format="%Y-%m-%d")
        idx_df["log_ret"] = np.log(idx_df["Close"] / idx_df["Close"].shift(1))
        y_series = idx_df["log_ret"].dropna()
        print(f"✓ 指数: {len(y_series)} 行 ({y_series.index[0].date()} 至 {y_series.index[-1].date()})")
        
        # 加载成分股
        file_list = glob.glob(os.path.join(self.components_dir, "*.csv"))
        print(f"✓ 发现 {len(file_list)} 个文件")
        
        rets = []
        symbols = []
        for i, file_path in enumerate(file_list):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(file_list)}")
            
            df = self.load_dataframe(file_path, date_format="%Y-%m-%d")
            df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
            rets.append(df["log_ret"])
            symbols.append(os.path.splitext(os.path.basename(file_path))[0])
        
        print(f"✓ 成分股: {len(symbols)} 个资产")
        
        # 对齐日期
        A_df = pd.concat(rets, axis=1)
        A_df.columns = symbols
        A_df = A_df.infer_objects()
        
        data = pd.concat([y_series, A_df], axis=1).dropna().infer_objects()
        
        y_all = data["log_ret"].values
        A_all = data[symbols].values
        dates_all = data.index
        
        print(f"✓ 全部数据: {len(y_all)} 天")
        print(f"✓ 日期范围: {dates_all[0].date()} 至 {dates_all[-1].date()}")
        
        # 标准化
        if normalize:
            scaler_A = StandardScaler()
            A_std = scaler_A.fit_transform(A_all)
        else:
            A_std = A_all.copy()
        
        print()
        return A_all, A_std, y_all, symbols, dates_all


# =====================================================================
# 第二部分：改进的AE-IT4-PGD
# =====================================================================

class AEIT4PGD:
    """AE-IT4-PGD Algo"""
    
    def __init__(
        self, A: np.ndarray, y: np.ndarray, K: int = 30, h: float = 0.10,
        epsilon: float = 0.05, rho: float = 20, eta: float = None,
        tau: float = None, M: int = 10, max_iter: int = 500, tol: float = 1e-8
    ):
        self.A = A
        self.y = np.asarray(y).flatten()
        self.T, self.n = A.shape
        
        self.K = K
        self.h = h
        self.epsilon = epsilon
        self.rho = rho
        self.M = M
        self.max_iter = max_iter
        self.tol = tol
        
        ATA = 2 * self.A.T @ self.A
        ones = np.ones(self.n)
        hess = ATA + self.rho * np.outer(ones, ones)
        self.L = np.max(np.linalg.eigvalsh(hess))
        
        if eta is None:
            self.eta = 0.9 / self.L
        else:
            self.eta = min(eta, 0.9 / self.L)
        
        if tau is None:
            x_init = np.ones(self.n) / self.n
            residuals = np.abs(self.A @ x_init - self.y)
            self.tau = np.std(residuals)
            if self.tau < 1e-6:
                self.tau = 0.5
        else:
            self.tau = tau
        
        self.x_ref = np.ones(self.n) / self.n
        self.alpha = self._compute_attention(self.x_ref)
        self.x_history = []
        self.f_history = []
        self.alpha_history = []
    
    def _compute_attention(self, x: np.ndarray) -> np.ndarray:
        resid = np.abs(self.A @ x - self.y)
        z = (resid / self.tau) - np.max(resid / self.tau)
        alpha = 1 - np.exp(z) / (np.exp(z) + 1)
        return np.clip(alpha, 0, 1)
    
    def _compute_objective(self, x: np.ndarray) -> float:
        resid = self.A @ x - self.y
        loss = np.sum(resid ** 2)
        penalty = 0.5 * self.rho * (np.sum(x) - 1) ** 2
        return loss + penalty
    
    def _compute_gradient(self, x: np.ndarray) -> np.ndarray:
        resid = self.A @ x - self.y
        weighted_resid = self.alpha * resid
        grad_A = 2 * self.A.T @ weighted_resid
        grad_penalty = self.rho * (np.sum(x) - 1) * np.ones(self.n)
        return grad_A + grad_penalty
    
    def _compute_bounds(self, x_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        zeta = np.maximum(0, x_prev - self.epsilon)
        upsilon = np.minimum(self.h, x_prev + self.epsilon)
        return zeta, upsilon
    
    def _project_onto_constraints(
        self, z: np.ndarray, zeta: np.ndarray, upsilon: np.ndarray
    ) -> np.ndarray:
        x = np.zeros(self.n)
        omega = np.where(zeta > 0)[0]
        K_prime = len(omega)
        
        for i in omega:
            x[i] = np.clip(z[i], zeta[i], upsilon[i])
        
        if K_prime >= self.K:
            x_sum = np.sum(x)
            return x / x_sum if x_sum > 0 else x
        
        K_double = self.K - K_prime
        gamma = np.where(zeta == 0)[0]
        
        if len(gamma) > 0:
            phi_values = np.array([z[i] for i in gamma])
            sorted_indices = np.argsort(-np.abs(phi_values))[:min(K_double, len(gamma))]
            phi_prime_indices = gamma[sorted_indices]
            
            for i in phi_prime_indices:
                x[i] = np.clip(z[i], 0, upsilon[i])
        else:
            phi_prime_indices = np.array([], dtype=int)
        
        candidate_indices = np.concatenate([omega, phi_prime_indices])
        
        if len(candidate_indices) > self.K:
            delta = np.zeros(len(candidate_indices))
            for j, i in enumerate(candidate_indices):
                delta[j] = (x[i] - z[i]) ** 2 - z[i] ** 2
            
            sorted_delta_idx = np.argsort(delta)[:self.K]
            selected_indices = candidate_indices[sorted_delta_idx]
            
            x_new = np.zeros(self.n)
            for i in selected_indices:
                x_new[i] = np.clip(z[i], max(0, zeta[i]), upsilon[i])
            x = x_new
        
        x = np.clip(x, 0, self.h)
        x_sum = np.sum(x)
        
        if x_sum > 1e-10:
            x = x / x_sum
        else:
            x = np.ones(self.n) / self.n
        
        return x
    
    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.x_ref.copy()
        x_prev = x.copy()
        
        for t in range(self.max_iter):
            grad = self._compute_gradient(x)
            z = x - self.eta * grad
            
            zeta, upsilon = self._compute_bounds(x_prev)
            x = self._project_onto_constraints(z, zeta, upsilon)
            
            if (t + 1) % self.M == 0:
                self.alpha = self._compute_attention(x)
            
            f_val = self._compute_objective(x)
            self.f_history.append(f_val)
            self.x_history.append(x.copy())
            self.alpha_history.append(self.alpha.copy())
            
            norm_diff = np.linalg.norm(x - x_prev) ** 2
            sparsity = np.sum(np.abs(x) > 1e-6)
            
            if sparsity > 0:
                convergence_metric = norm_diff / sparsity
            else:
                convergence_metric = norm_diff
            
            if convergence_metric < self.tol:
                break
            
            x_prev = x.copy()
        
        self.x_final = x
        self.alpha_final = self.alpha
        return x, self.alpha_final


# =====================================================================
# 第三部分：IT4-PGD
# =====================================================================

class IT4PGD:
    """标准IT4-PGD"""
    
    def __init__(
        self, A: np.ndarray, y: np.ndarray, K: int = 30, h: float = 0.10,
        epsilon: float = 0.05, rho: float = 20, eta: float = None,
        max_iter: int = 500, tol: float = 1e-8
    ):
        self.A = A
        self.y = np.asarray(y).flatten()
        self.T, self.n = A.shape
        
        self.K = K
        self.h = h
        self.epsilon = epsilon
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        
        ATA = 2 * self.A.T @ self.A
        ones = np.ones(self.n)
        hess = ATA + self.rho * np.outer(ones, ones)
        self.L = np.max(np.linalg.eigvalsh(hess))
        
        if eta is None:
            self.eta = 0.9 / self.L
        else:
            self.eta = min(eta, 0.9 / self.L)
        
        self.x_ref = np.ones(self.n) / self.n
        self.x_history = []
        self.f_history = []
    
    def _compute_objective(self, x: np.ndarray) -> float:
        resid = self.A @ x - self.y
        loss = np.sum(resid ** 2)
        penalty = 0.5 * self.rho * (np.sum(x) - 1) ** 2
        return loss + penalty
    
    def _compute_gradient(self, x: np.ndarray) -> np.ndarray:
        resid = self.A @ x - self.y
        grad_A = 2 * self.A.T @ resid
        grad_penalty = self.rho * (np.sum(x) - 1) * np.ones(self.n)
        return grad_A + grad_penalty
    
    def _compute_bounds(self, x_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        zeta = np.maximum(0, x_prev - self.epsilon)
        upsilon = np.minimum(self.h, x_prev + self.epsilon)
        return zeta, upsilon
    
    def _project_onto_constraints(
        self, z: np.ndarray, zeta: np.ndarray, upsilon: np.ndarray
    ) -> np.ndarray:
        x = np.zeros(self.n)
        omega = np.where(zeta > 0)[0]
        K_prime = len(omega)
        
        for i in omega:
            x[i] = np.clip(z[i], zeta[i], upsilon[i])
        
        if K_prime >= self.K:
            x_sum = np.sum(x)
            return x / x_sum if x_sum > 0 else x
        
        K_double = self.K - K_prime
        gamma = np.where(zeta == 0)[0]
        
        if len(gamma) > 0:
            phi_values = np.array([z[i] for i in gamma])
            sorted_indices = np.argsort(-np.abs(phi_values))[:min(K_double, len(gamma))]
            phi_prime_indices = gamma[sorted_indices]
            
            for i in phi_prime_indices:
                x[i] = np.clip(z[i], 0, upsilon[i])
        else:
            phi_prime_indices = np.array([], dtype=int)
        
        candidate_indices = np.concatenate([omega, phi_prime_indices])
        
        if len(candidate_indices) > self.K:
            delta = np.zeros(len(candidate_indices))
            for j, i in enumerate(candidate_indices):
                delta[j] = (x[i] - z[i]) ** 2 - z[i] ** 2
            
            sorted_delta_idx = np.argsort(delta)[:self.K]
            selected_indices = candidate_indices[sorted_delta_idx]
            
            x_new = np.zeros(self.n)
            for i in selected_indices:
                x_new[i] = np.clip(z[i], max(0, zeta[i]), upsilon[i])
            x = x_new
        
        x = np.clip(x, 0, self.h)
        x_sum = np.sum(x)
        
        if x_sum > 1e-10:
            x = x / x_sum
        else:
            x = np.ones(self.n) / self.n
        
        return x
    
    def fit(self) -> np.ndarray:
        x = self.x_ref.copy()
        x_prev = x.copy()
        
        for t in range(self.max_iter):
            grad = self._compute_gradient(x)
            z = x - self.eta * grad
            
            zeta, upsilon = self._compute_bounds(x_prev)
            x = self._project_onto_constraints(z, zeta, upsilon)
            
            f_val = self._compute_objective(x)
            self.f_history.append(f_val)
            self.x_history.append(x.copy())
            
            norm_diff = np.linalg.norm(x - x_prev) ** 2
            sparsity = np.sum(np.abs(x) > 1e-6)
            
            if sparsity > 0:
                convergence_metric = norm_diff / sparsity
            else:
                convergence_metric = norm_diff
            
            if convergence_metric < self.tol:
                break
            
            x_prev = x.copy()
        
        self.x_final = x
        return x









# =====================================================================
# 第四部分：性能指标与月度聚合
# =====================================================================

class PerformanceMetrics:
    """性能指标"""
    
    @staticmethod
    def compute_portfolio_returns(A: np.ndarray, x: np.ndarray) -> np.ndarray:
        return A @ x
    
    @staticmethod
    def compute_accumulated_return(returns: np.ndarray) -> float:
        return np.exp(np.sum(returns)) - 1
    
    @staticmethod
    def compute_mdte(portfolio_returns: np.ndarray, index_returns: np.ndarray) -> float:
        tracking_error = portfolio_returns - index_returns
        return np.mean(np.abs(tracking_error)) * 10000  # bps
    
    @staticmethod
    def group_by_month(dates, values):
        """按月份聚合数据"""
        df = pd.DataFrame({'date': dates, 'value': values})
        df['yearmonth'] = df['date'].dt.to_period('M')
        monthly = df.groupby('yearmonth')['value'].mean()
        month_labels = [str(p) for p in monthly.index]
        return month_labels, monthly.values








# =====================================================================
# 第五部分：可视化
# =====================================================================

class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_all_in_one(
        dates, index_ret, it4_ret, ae_ret,
        output_dir="."
    ):
        """所有图表合并为一张 (3x2 子图)，使用实际数据范围"""
        
        # 使用实际数据范围（从最早日期到最后日期）
        dates_display = dates
        index_ret_display = index_ret
        it4_ret_display = it4_ret
        ae_ret_display = ae_ret
        
        print(f"✓ 显示日期范围: {dates_display[0].date()} 至 {dates_display[-1].date()}")
        
        fig = plt.figure(figsize=(18, 14))
        
        # ===== 第1行 =====
        # 1.1 累积收益（直接显示，不normalize）
        ax1 = plt.subplot(3, 2, 1)
        index_acc = np.exp(np.cumsum(index_ret_display)) - 1
        it4_acc = np.exp(np.cumsum(it4_ret_display)) - 1
        ae_acc = np.exp(np.cumsum(ae_ret_display)) - 1
        
        ax1.plot(dates_display, index_acc, label='Index', linewidth=2.5, color='blue', linestyle='--')
        ax1.plot(dates_display, it4_acc, label='IT4-PGD', linewidth=2.5, color='black')
        ax1.plot(dates_display, ae_acc, label='AE-IT4-PGD', linewidth=2.5, color='green')
        ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Accumulated Return', fontsize=11, fontweight='bold')
        ax1.set_title(f'(a) Accumulated Returns ({dates_display[0].strftime("%Y-%m")} to {dates_display[-1].strftime("%Y-%m")})', 
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 1.2 日收益
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(dates_display, index_ret_display*100, label='Index', linewidth=1, color='blue', alpha=0.7, linestyle='--')
        ax2.plot(dates_display, it4_ret_display*100, label='IT4-PGD', linewidth=1, color='black', alpha=0.7)
        ax2.plot(dates_display, ae_ret_display*100, label='AE-IT4-PGD', linewidth=1, color='green', alpha=0.7)
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Daily Return (%)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Daily Returns', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # ===== 第2行 =====
        # 2.1 MDTE (按月显示)
        ax3 = plt.subplot(3, 2, 3)
        it4_error = it4_ret_display - index_ret_display
        ae_error = ae_ret_display - index_ret_display
        
        it4_mdte_vals = np.abs(it4_error) * 10000
        ae_mdte_vals = np.abs(ae_error) * 10000
        
        month_labels, it4_mdte_monthly = PerformanceMetrics.group_by_month(dates_display, it4_mdte_vals)
        month_labels, ae_mdte_monthly = PerformanceMetrics.group_by_month(dates_display, ae_mdte_vals)
        
        x_pos = np.arange(len(month_labels))
        width = 0.35
        
        ax3.bar(x_pos - width/2, it4_mdte_monthly, width, label='IT4-PGD', color='red', alpha=0.7)
        ax3.bar(x_pos + width/2, ae_mdte_monthly, width, label='AE-IT4-PGD', color='green', alpha=0.7)
        ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax3.set_ylabel('MDTE (bps)', fontsize=11, fontweight='bold')
        ax3.set_title('(c) MDTE per Month', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 2.2 月度波动率
        ax4 = plt.subplot(3, 2, 4)
        
        month_vol_it4 = []
        month_vol_ae = []
        for month in pd.DatetimeIndex(dates_display).to_period('M').unique():
            month_mask = pd.DatetimeIndex(dates_display).to_period('M') == month
            if np.sum(month_mask) > 0:
                month_vol_it4.append(np.mean(np.abs(it4_ret_display[month_mask])) * 10000)
                month_vol_ae.append(np.mean(np.abs(ae_ret_display[month_mask])) * 10000)
        
        x_pos2 = np.arange(len(month_vol_it4))
        ax4.bar(x_pos2 - width/2, month_vol_it4, width, label='IT4-PGD', color='red', alpha=0.7)
        ax4.bar(x_pos2 + width/2, month_vol_ae, width, label='AE-IT4-PGD', color='green', alpha=0.7)
        ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Average |Return| (bps)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Monthly Volatility', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos2)
        ax4.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # ===== 第3行 =====
        # 3.1 总收益对比
        ax5 = plt.subplot(3, 2, 5)
        index_total = PerformanceMetrics.compute_accumulated_return(index_ret_display)
        it4_total = PerformanceMetrics.compute_accumulated_return(it4_ret_display)
        ae_total = PerformanceMetrics.compute_accumulated_return(ae_ret_display)
        
        names = ['Index', 'IT4-PGD', 'AE-IT4-PGD']
        values = [index_total*100, it4_total*100, ae_total*100]
        colors = ['blue', 'red', 'green']
        bars = ax5.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax5.set_ylabel('Accumulated Return (%)', fontsize=11, fontweight='bold')
        ax5.set_title(f'(e) Total Accumulated Returns ({dates_display[0].strftime("%Y-%m")} to {dates_display[-1].strftime("%Y-%m")})', 
                      fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3.2 改进对比
        ax6 = plt.subplot(3, 2, 6)
        it4_mdte_mean = np.mean(it4_mdte_vals)
        ae_mdte_mean = np.mean(ae_mdte_vals)
        mdte_improvement = (it4_mdte_mean - ae_mdte_mean) / it4_mdte_mean * 100 if it4_mdte_mean > 0 else 0
        
        acc_improvement = (ae_total - it4_total) / abs(it4_total) * 100 if it4_total != 0 else 0
        
        improvements = [mdte_improvement, acc_improvement]
        labels = ['MDTE\nImprovement', 'Return\nImprovement']
        colors_imp = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax6.bar(labels, improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax6.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax6.set_title('(f) AE-IT4-PGD vs IT4-PGD Improvement', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            label = f'{val:+.2f}%'
            va = 'bottom' if height > 0 else 'top'
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va=va, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_results.png'), dpi=300, bbox_inches='tight')
        print("✓ 图表已保存: all_results.png")
        plt.close()


# =====================================================================
# 主程序
# =====================================================================

if __name__ == "__main__":
    
    COMPONENTS_DIR = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\components"
    INDEX_CSV      = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\sp500_index.csv"
    OUTPUT_DIR     = "."
    K_SPARSITY     = 30
    
    print("\n" + "=" * 80)
    print("【完整回测】AE-IT4-PGD vs IT4-PGD vs 指数")
    print("(全部数据训练，自动检测显示范围)")
    print("=" * 80 + "\n")
    
    # 步骤1：加载全部数据
    loader = DataLoader(COMPONENTS_DIR, INDEX_CSV)
    A_original, A_std, y_original, symbols, dates = loader.load_data(normalize=True)
    
    # 步骤2：训练模型（使用全部数据）
    print("=" * 80)
    print("步骤2: 训练模型 (使用全部数据)")
    print("=" * 80 + "\n")
    
    print("训练IT4-PGD...")
    model_it4 = IT4PGD(A_std, y_original, K=K_SPARSITY, h=0.10, epsilon=0.05, rho=20)
    x_it4 = model_it4.fit()
    
    print("训练AE-IT4-PGD...")
    model_ae = AEIT4PGD(A_std, y_original, K=K_SPARSITY, h=0.10, epsilon=0.05, rho=20)
    x_ae, alpha_ae = model_ae.fit()
    
    x_index = np.ones(len(symbols)) / len(symbols)
    print("✓ 指数基准: 等权投资组合\n")
    
    # 步骤3：计算全部收益
    print("=" * 80)
    print("步骤3: 计算收益")
    print("=" * 80 + "\n")
    
    index_ret = A_original @ x_index
    it4_ret = A_original @ x_it4
    ae_ret = A_original @ x_ae
    
    # 计算全部时间段的性能指标
    index_acc_full = PerformanceMetrics.compute_accumulated_return(index_ret)
    it4_acc_full = PerformanceMetrics.compute_accumulated_return(it4_ret)
    ae_acc_full = PerformanceMetrics.compute_accumulated_return(ae_ret)
    
    it4_mdte_full = PerformanceMetrics.compute_mdte(it4_ret, index_ret)
    ae_mdte_full = PerformanceMetrics.compute_mdte(ae_ret, index_ret)
    
    print(f"【全部时间段性能】({dates[0].date()} 至 {dates[-1].date()})")
    print(f"{'指标':<20} {'指数':<18} {'IT4-PGD':<18} {'AE-IT4-PGD':<18}")
    print("-" * 80)
    print(f"{'Acc.Return (%)':<20} {index_acc_full*100:<18.4f} "
          f"{it4_acc_full*100:<18.4f} "
          f"{ae_acc_full*100:<18.4f}")
    print(f"{'MDTE (bps)':<20} {'N/A':<18} "
          f"{it4_mdte_full:<18.2f} "
          f"{ae_mdte_full:<18.2f}")
    print()
    
    mdte_improvement = (it4_mdte_full - ae_mdte_full) / it4_mdte_full * 100
    acc_improvement = (ae_acc_full - it4_acc_full) / abs(it4_acc_full) * 100
    
    print(f"MDTE 改进: {mdte_improvement:+.2f}%")
    print(f"累积收益改进: {acc_improvement:+.2f}%\n")
    
    # 步骤4：生成图表（使用实际数据范围）
    print("=" * 80)
    print("步骤4: 生成图表")
    print("=" * 80 + "\n")
    
    Visualizer.plot_all_in_one(dates, index_ret, it4_ret, ae_ret, OUTPUT_DIR)
    
    print("=" * 80)
    print("【完成】")
    print("=" * 80)