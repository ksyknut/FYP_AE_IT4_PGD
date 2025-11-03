"""
=====================================================================
【最终完整版】AE-IT4-PGD vs IT4-PGD vs Index 对比分析
- 所有图表合并为一张 (3x2 子图)
- 累积收益图使用 Accumulated Return (exp of log return) 而非 log return
- 无CSV导出，直接print输出
- 仅输出权重信息
- 简洁版本
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
        """加载指数和成分股数据"""
        print("=" * 80)
        print("加载数据...")
        print("=" * 80)
        
        # 加载指数
        idx_df = self.load_dataframe(self.index_csv, date_format="%Y-%m-%d")
        idx_df["log_ret"] = np.log(idx_df["Close"] / idx_df["Close"].shift(1))
        y_series = idx_df["log_ret"].dropna()
        print(f"✓ 指数: {len(y_series)} 行")
        
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
        
        y = data["log_ret"].values
        A = data[symbols].values
        dates = data.index
        
        print(f"✓ 日期: {dates[0].date()} 至 {dates[-1].date()}")
        
        # 标准化
        if normalize:
            scaler_A = StandardScaler()
            A_std = scaler_A.fit_transform(A)
        else:
            A_std = A.copy()
        
        print()
        return A, A_std, y, symbols, dates


# =====================================================================
# 第二部分：改进的AE-IT4-PGD
# =====================================================================

class ImprovedAEIT4PGD:
    """改进的AE-IT4-PGD"""
    
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
        print("=" * 80)
        print("训练改进AE-IT4-PGD...")
        print("=" * 80)
        
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
            
            if (t + 1) % 50 == 0:
                print(f"Iter {t+1:3d} | f={f_val:.6e} | Δx={norm_diff:.6e} | 非零={int(sparsity)}")
            
            if convergence_metric < self.tol:
                print(f"✓ 收敛! 迭代: {t+1}")
                break
            
            x_prev = x.copy()
        
        self.x_final = x
        self.alpha_final = self.alpha
        print()
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
        print("=" * 80)
        print("训练标准IT4-PGD...")
        print("=" * 80)
        
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
            
            if (t + 1) % 50 == 0:
                print(f"Iter {t+1:3d} | f={f_val:.6e} | Δx={norm_diff:.6e} | 非零={int(sparsity)}")
            
            if convergence_metric < self.tol:
                print(f"✓ 收敛! 迭代: {t+1}")
                break
            
            x_prev = x.copy()
        
        self.x_final = x
        print()
        return x


# =====================================================================
# 第四部分：性能指标
# =====================================================================

class PerformanceMetrics:
    """性能指标"""
    
    @staticmethod
    def compute_portfolio_returns(A: np.ndarray, x: np.ndarray) -> np.ndarray:
        return A @ x
    
    @staticmethod
    def compute_cumulative_return(returns: np.ndarray) -> float:
        return np.sum(returns)
    
    @staticmethod
    def compute_accumulated_return(returns: np.ndarray) -> float:
        """计算 accumulated return: exp(sum(log_returns)) - 1"""
        return np.exp(np.sum(returns)) - 1
    
    @staticmethod
    def compute_mdte(portfolio_returns: np.ndarray, index_returns: np.ndarray) -> float:
        tracking_error = portfolio_returns - index_returns
        return np.mean(np.abs(tracking_error))
    
    @staticmethod
    def compute_atr(x_portfolio: np.ndarray, x_prev: np.ndarray = None) -> float:
        if x_prev is None:
            x_prev = np.ones(len(x_portfolio)) / len(x_portfolio)
        return np.sum(np.abs(x_portfolio - x_prev))
    
    @staticmethod
    def compute_metrics_summary(
        A_original: np.ndarray, y_original: np.ndarray, x: np.ndarray, name: str
    ) -> Dict:
        
        portfolio_ret = PerformanceMetrics.compute_portfolio_returns(A_original, x)
        cum_ret = PerformanceMetrics.compute_cumulative_return(portfolio_ret)
        acc_ret = PerformanceMetrics.compute_accumulated_return(portfolio_ret)
        mdte = PerformanceMetrics.compute_mdte(portfolio_ret, y_original)
        atr = PerformanceMetrics.compute_atr(x)
        
        nonzero = np.sum(x > 1e-6)
        
        return {
            'Name': name,
            'CumulativeReturn': cum_ret,
            'AccumulatedReturn': acc_ret,
            'MDTE': mdte,
            'MDTE_bps': mdte * 10000,
            'ATR': atr,
            'Nonzero_Assets': nonzero,
            'Portfolio_Returns': portfolio_ret
        }


# =====================================================================
# 第五部分：单张图表可视化
# =====================================================================

class Visualizer:
    """单张图表可视化"""
    
    @staticmethod
    def plot_all_in_one(
        dates, index_ret, it4_ret, ae_ret, it4_error, ae_error,
        results, symbols, x_it4, x_ae, alpha_ae, output_dir="."
    ):
        """所有图表合并为一张 (3x2 子图)"""
        
        fig = plt.figure(figsize=(18, 14))
        
        # ===== 第1行 =====
        # 1.1 累积收益 (Accumulated Return)
        ax1 = plt.subplot(3, 2, 1)
        index_acc = np.exp(np.cumsum(index_ret)) - 1
        it4_acc = np.exp(np.cumsum(it4_ret)) - 1
        ae_acc = np.exp(np.cumsum(ae_ret)) - 1
        
        ax1.plot(dates, index_acc * 100, label='Index', linewidth=2.5, color='blue')
        ax1.plot(dates, it4_acc * 100, label='IT4-PGD', linewidth=2.5, color='red')
        ax1.plot(dates, ae_acc * 100, label='AE-IT4-PGD', linewidth=2.5, color='green')
        ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Accumulated Return (%)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Accumulated Returns Comparison', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 1.2 日追踪误差
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(dates, np.abs(it4_error), label='IT4-PGD', linewidth=1.5, color='red', alpha=0.7)
        ax2.plot(dates, np.abs(ae_error), label='AE-IT4-PGD', linewidth=1.5, color='green', alpha=0.7)
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Absolute Tracking Error', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Daily Tracking Error Comparison', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # ===== 第2行 =====
        # 2.1 性能指标对比 - 累积收益
        ax3 = plt.subplot(3, 2, 3)
        names = list(results.keys())
        values = [results[name]['AccumulatedReturn'] * 100 for name in names]
        colors = ['blue', 'red', 'green']
        bars = ax3.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Accumulated Return (%)', fontsize=11, fontweight='bold')
        ax3.set_title('(c) Accumulated Returns', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # 2.2 MDTE对比
        ax4 = plt.subplot(3, 2, 4)
        values = [results[name]['MDTE_bps'] for name in names]
        bars = ax4.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('MDTE (bps)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Mean Daily Tracking Error', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # ===== 第3行 =====
        # 3.1 注意力分布
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(alpha_ae, linewidth=1.5, color='purple', alpha=0.7)
        ax5.set_xlabel('Trading Day', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Attention Weight', fontsize=11, fontweight='bold', color='purple')
        ax5.tick_params(axis='y', labelcolor='purple')
        ax5.grid(True, alpha=0.3)
        ax5.set_title('(e) Attention Weights Over Time', fontsize=12, fontweight='bold')
        
        # 3.2 投资组合权重对比 (Top 10)
        ax6 = plt.subplot(3, 2, 6)
        
        nonzero_idx = np.where(x_ae > 1e-6)[0]
        top_idx = nonzero_idx[np.argsort(-x_ae[nonzero_idx])[:10]]
        top_symbols = [symbols[i] for i in top_idx]
        top_weights_ae = x_ae[top_idx]
        
        y_pos = np.arange(len(top_symbols))
        ax6.barh(y_pos, top_weights_ae, color='green', alpha=0.7, edgecolor='black')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(top_symbols, fontsize=9)
        ax6.set_xlabel('Weight', fontsize=11, fontweight='bold')
        ax6.set_title('(f) AE-IT4-PGD Top 10 Assets', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_results.png'), dpi=300, bbox_inches='tight')
        print("✓ 所有图表已合并: all_results.png")
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
    print("【完整解决方案】AE-IT4-PGD vs IT4-PGD vs 指数")
    print("=" * 80 + "\n")
    
    # ===================================================================
    # 步骤1：加载数据
    # ===================================================================
    loader = DataLoader(COMPONENTS_DIR, INDEX_CSV)
    A_original, A_std, y_original, symbols, dates = loader.load_data(normalize=True)
    
    # ===================================================================
    # 步骤2: 训练模型
    # ===================================================================
    print("\n" + "=" * 80)
    print("步骤2: 训练模型")
    print("=" * 80 + "\n")
    
    model_ae = ImprovedAEIT4PGD(
        A=A_std, y=y_original, K=K_SPARSITY, h=0.10, epsilon=0.05,
        rho=20, max_iter=500, tol=1e-8
    )
    x_ae, alpha_ae = model_ae.fit()
    
    model_it4 = IT4PGD(
        A=A_std, y=y_original, K=K_SPARSITY, h=0.10, epsilon=0.05,
        rho=20, max_iter=500, tol=1e-8
    )
    x_it4 = model_it4.fit()
    
    x_index = np.ones(len(symbols)) / len(symbols)
    print("=" * 80)
    print("指数基准: 等权投资组合")
    print("=" * 80 + "\n")
    
    # ===================================================================
    # 步骤3: 计算性能指标
    # ===================================================================
    print("=" * 80)
    print("步骤3: 性能指标")
    print("=" * 80 + "\n")
    
    metrics_index = PerformanceMetrics.compute_metrics_summary(
        A_original, y_original, x_index, "Index"
    )
    metrics_it4 = PerformanceMetrics.compute_metrics_summary(
        A_original, y_original, x_it4, "IT4-PGD"
    )
    metrics_ae = PerformanceMetrics.compute_metrics_summary(
        A_original, y_original, x_ae, "AE-IT4-PGD"
    )
    
    results = {
        'Index': metrics_index,
        'IT4-PGD': metrics_it4,
        'AE-IT4-PGD': metrics_ae
    }
    
    # ===================================================================
    # 步骤4: 打印性能对比报告
    # ===================================================================
    print("=" * 80)
    print("性能对比表 (使用Accumulated Return)")
    print("=" * 80 + "\n")
    
    print(f"{'指标':<20} {'指数':<18} {'IT4-PGD':<18} {'AE-IT4-PGD':<18}")
    print("-" * 80)
    print(f"{'Acc.Return (%)':<20} {results['Index']['AccumulatedReturn']*100:<18.4f} "
          f"{results['IT4-PGD']['AccumulatedReturn']*100:<18.4f} "
          f"{results['AE-IT4-PGD']['AccumulatedReturn']*100:<18.4f}")
    print(f"{'MDTE (bps)':<20} {results['Index']['MDTE_bps']:<18.4f} "
          f"{results['IT4-PGD']['MDTE_bps']:<18.4f} "
          f"{results['AE-IT4-PGD']['MDTE_bps']:<18.4f}")
    print(f"{'ATR':<20} {results['Index']['ATR']:<18.6f} "
          f"{results['IT4-PGD']['ATR']:<18.6f} "
          f"{results['AE-IT4-PGD']['ATR']:<18.6f}")
    print(f"{'持仓资产数':<20} {results['Index']['Nonzero_Assets']:<18} "
          f"{results['IT4-PGD']['Nonzero_Assets']:<18} "
          f"{results['AE-IT4-PGD']['Nonzero_Assets']:<18}")
    print()
    
    # ===================================================================
    # 步骤5: 打印改进分析
    # ===================================================================
    print("=" * 80)
    print("AE-IT4-PGD 改进分析")
    print("=" * 80 + "\n")
    
    it4_mdte = results['IT4-PGD']['MDTE_bps']
    ae_mdte = results['AE-IT4-PGD']['MDTE_bps']
    mdte_improvement = (it4_mdte - ae_mdte) / it4_mdte * 100
    
    it4_acc = results['IT4-PGD']['AccumulatedReturn']
    ae_acc = results['AE-IT4-PGD']['AccumulatedReturn']
    acc_improvement = (ae_acc - it4_acc) / abs(it4_acc) * 100 if it4_acc != 0 else 0
    
    it4_atr = results['IT4-PGD']['ATR']
    ae_atr = results['AE-IT4-PGD']['ATR']
    atr_improvement = (it4_atr - ae_atr) / it4_atr * 100 if it4_atr != 0 else 0
    
    print(f"MDTE 改进: {mdte_improvement:+.2f}%")
    print(f"累积收益改进: {acc_improvement:+.2f}%")
    print(f"ATR 改进: {atr_improvement:+.2f}%")
    print()
    
    # ===================================================================
    # 步骤6: 打印投资组合权重
    # ===================================================================
    print("=" * 80)
    print("IT4-PGD 投资组合权重 (非零资产)")
    print("=" * 80 + "\n")
    
    nonzero_indices_it4 = np.where(x_it4 > 1e-6)[0]
    sorted_indices_it4 = nonzero_indices_it4[np.argsort(-x_it4[nonzero_indices_it4])]
    
    print(f"{'排名':<6} {'Symbol':<10} {'权重':<15} {'百分比':<10}")
    print("-" * 50)
    for i, idx in enumerate(sorted_indices_it4, 1):
        print(f"{i:<6} {symbols[idx]:<10} {x_it4[idx]:<15.6f} {x_it4[idx]*100:<10.2f}%")
    print()
    
    print("=" * 80)
    print("AE-IT4-PGD 投资组合权重 (非零资产)")
    print("=" * 80 + "\n")
    
    nonzero_indices_ae = np.where(x_ae > 1e-6)[0]
    sorted_indices_ae = nonzero_indices_ae[np.argsort(-x_ae[nonzero_indices_ae])]
    
    print(f"{'排名':<6} {'Symbol':<10} {'权重':<15} {'百分比':<10}")
    print("-" * 50)
    for i, idx in enumerate(sorted_indices_ae, 1):
        print(f"{i:<6} {symbols[idx]:<10} {x_ae[idx]:<15.6f} {x_ae[idx]*100:<10.2f}%")
    print()
    
    # ===================================================================
    # 步骤7: 生成单张综合图表
    # ===================================================================
    print("=" * 80)
    print("步骤7: 生成图表")
    print("=" * 80 + "\n")
    
    Visualizer.plot_all_in_one(
        dates,
        results['Index']['Portfolio_Returns'],
        results['IT4-PGD']['Portfolio_Returns'],
        results['AE-IT4-PGD']['Portfolio_Returns'],
        results['IT4-PGD']['Portfolio_Returns'] - y_original,
        results['AE-IT4-PGD']['Portfolio_Returns'] - y_original,
        results, symbols, x_it4, x_ae, alpha_ae, OUTPUT_DIR
    )
    
    # ===================================================================
    # 完成
    # ===================================================================
    print("=" * 80)
    print("【完整解决方案执行完成】")
    print("=" * 80 + "\n")
    print("输出文件:")
    print(f"  - all_results.png (6个子图合并)")
    print(f"\n")