"""
GLOBAL STAT ARB: RMT-CLEANED AVELLANEDA-LEE
═══════════════════════════════════════════

WHAT IT DOES
Standard Avellaneda-Lee runs PCA on an empirical correlation matrix
that is ~90% noise (Laloux et al., 1999). This script fixes that by
denoising with the Marchenko-Pastur law first — producing more stable
eigenportfolios and cleaner s-scores — and applies the framework to a
global universe: Indian equities, US equities, and commodity/FX ETFs.

KEY FORMULA
For N assets over T days, eigenvalues of a random correlation matrix
follow the Marchenko-Pastur distribution with upper bound:
    λ_max = σ²(1 + √(N/T))²
Any empirical eigenvalue below λ_max is noise. We clip and rebuild.

PIPELINE
1. Fetch global universe → log returns
2. Empirical correlation C → RMT denoise → C_clean
3. PCA on C_clean → eigenportfolios (systematic factors)
4. Regress each asset on factors → idiosyncratic residuals ε_i
5. Fit OU process to cumulative ε_i → s-score
6. Trade: long s < -1.25, short s > +1.25, exit at ±0.50
7. Half-Kelly position sizing from OU parameters
8. Walk-forward backtest (252-day train, 21-day test)
9. Compare RMT vs plain PCA baseline

REFERENCES
[1] Marchenko & Pastur (1967). Math. USSR-Sbornik, 1(4).
[2] Laloux et al. (1999). Phys. Rev. Letters, 83(7), 1467–1470.
[3] Bouchaud & Potters (2009). arXiv:0910.1205 [q-fin.ST].
[4] Avellaneda & Lee (2010). Quantitative Finance, 10(7), 761–782.
[5] Kelly (1956). Bell System Technical Journal, 35(4), 917–926.


"""


from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

warnings.filterwarnings("ignore")
console = Console()





@dataclass
class Config:
   

    START_DATE: str = "2018-01-01"
    END_DATE: str = "2024-12-31"
    ESTIMATION_WINDOW: int = 252          
    TEST_WINDOW: int = 21                 
    N_FACTORS_MAX: int = 15              
    S_SCORE_OPEN: float = 1.25           
    S_SCORE_CLOSE: float = 0.50          
    HALF_KELLY: float = 0.50              
    TRANSACTION_COST_BP: int = 10         
    MIN_HALF_LIFE_DAYS: int = 1           
    MAX_HALF_LIFE_DAYS: int = 40         
    MIN_COVERAGE: float = 0.85            
    MAX_CONSECUTIVE_FILL: int = 5         
    OU_WINDOW: int = 60                   
    MIN_KAPPA: float = 0.01              


INDIAN_TICKERS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "WIPRO.NS",
]

# US equities — NYSE / NASDAQ large-cap
US_TICKERS: List[str] = [
    "AAPL", "MSFT", "JPM", "BAC", "XOM", "JNJ", "WMT", "GS", "COP", "LLY",
    "HD", "CAT", "PFE", "CVX", "MRK",
]

# Commodity + FX ETFs — cross-asset diversification
COMMODITY_FX_TICKERS: List[str] = [
    "GLD", "SLV", "GDX", "USO", "XLE", "UUP", "EEM", "TLT", "IEI", "FXI",
]

ALL_TICKERS: List[str] = INDIAN_TICKERS + US_TICKERS + COMMODITY_FX_TICKERS




def fetch_universe_data(
    tickers: List[str],
    start: str,
    end: str,
    config: Config,
) -> pd.DataFrame:
    
    console.print("[bold cyan][1/6] Fetching price data …[/bold cyan]")

    try:
        raw: pd.DataFrame = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = tickers[:1]
    except Exception as exc:
        console.print(f"[yellow]Bulk download failed ({exc}); fetching per-ticker …[/yellow]")
        prices = pd.DataFrame()

    # Per-ticker fallback for missing columns
    existing_cols = set(prices.columns) if not prices.empty else set()
    missing = [t for t in tickers if t not in existing_cols]

    if missing:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("Per-ticker download", total=len(missing))
            for tkr in missing:
                try:
                    tmp = yf.download(
                        tkr, start=start, end=end,
                        auto_adjust=True, progress=False,
                    )
                    if not tmp.empty:
                        col = tmp["Close"] if "Close" in tmp.columns else tmp.iloc[:, 0]
                        prices[tkr] = col
                except Exception:
                    console.print(f"[red]  ✗ {tkr} — skipped[/red]")
                prog.update(task, advance=1)

    if prices.empty:
        raise RuntimeError("No price data retrieved — check network / tickers.")

    # Coverage filter
    coverage = prices.notna().mean()
    keep = coverage[coverage >= config.MIN_COVERAGE].index.tolist()
    dropped = [c for c in prices.columns if c not in keep]
    if dropped:
        console.print(f"[yellow]  Dropped low-coverage tickers: {dropped}[/yellow]")
    prices = prices[keep]

    # Fill small gaps
    prices = (
        prices
        .fillna(method="ffill", limit=config.MAX_CONSECUTIVE_FILL)
        .fillna(method="bfill", limit=config.MAX_CONSECUTIVE_FILL)
    )
    prices.dropna(inplace=True)

    console.print(
        f"[green]  ✓ {prices.shape[1]} tickers × {prices.shape[0]} trading days[/green]"
    )
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.DataFrame
        Clean adjusted-close price panel.

    Returns
    -------
    pd.DataFrame
        Log-return panel (first row dropped).
    """
    log_ret: pd.DataFrame = np.log(prices / prices.shift(1))
    log_ret.dropna(how="all", inplace=True)
    log_ret.dropna(axis=1, how="all", inplace=True)
    return log_ret




def marchenko_pastur_pdf(
    x: np.ndarray,
    q: float,
    sigma_sq: float,
) -> np.ndarray:
    """Evaluate the Marchenko-Pastur probability density function.

    f(λ) = √[(λ_max − λ)(λ − λ_min)] / (2π σ² q λ)
    λ_min = σ²(1 − √q)²,  λ_max = σ²(1 + √q)²    (ref [1])

    Parameters
    ----------
    x : np.ndarray
        Eigenvalue grid.
    q : float
        Ratio N/T.
    sigma_sq : float
        Noise variance parameter.

    Returns
    -------
    np.ndarray
        MP density at each point in *x*.
    """
    lambda_min: float = sigma_sq * (1.0 - np.sqrt(q)) ** 2
    lambda_max: float = sigma_sq * (1.0 + np.sqrt(q)) ** 2

    pdf = np.zeros_like(x, dtype=np.float64)
    mask = (x >= lambda_min) & (x <= lambda_max)
    xm = x[mask]

    # Ref [1]: MP density numerator & denominator
    numerator = np.sqrt((lambda_max - xm) * (xm - lambda_min))
    denominator = 2.0 * np.pi * sigma_sq * q * xm
    pdf[mask] = numerator / denominator
    return pdf


def _kl_divergence_mp(
    sigma_sq: float,
    eigenvalues: np.ndarray,
    q: float,
    n_bins: int = 100,
) -> float:
    
    lambda_min = sigma_sq * (1.0 - np.sqrt(q)) ** 2
    lambda_max = sigma_sq * (1.0 + np.sqrt(q)) ** 2

    lo = max(eigenvalues.min() * 0.8, lambda_min * 0.5)
    hi = max(eigenvalues.max() * 1.2, lambda_max * 1.5)
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    emp_counts, _ = np.histogram(eigenvalues, bins=edges, density=False)
    emp_prob = emp_counts.astype(np.float64) / max(emp_counts.sum(), 1)

    mp_vals = marchenko_pastur_pdf(centres, q, sigma_sq)
    mp_total = mp_vals.sum()
    if mp_total < 1e-12:
        return 1e6

    mp_prob = mp_vals / mp_total

    eps = 1e-12
    emp_prob = emp_prob + eps
    mp_prob = mp_prob + eps
    emp_prob /= emp_prob.sum()
    mp_prob /= mp_prob.sum()

    kl = np.sum(emp_prob * np.log(emp_prob / mp_prob))
    return float(kl)


def fit_marchenko_pastur(
    eigenvalues: np.ndarray,
    q: float,
) -> Tuple[float, float, float]:
    
    cutoff = np.percentile(eigenvalues, 95)
    noise_eigs = eigenvalues[eigenvalues <= cutoff]
    if len(noise_eigs) < 3:
        noise_eigs = eigenvalues

    result = minimize_scalar(
        _kl_divergence_mp,
        bounds=(0.5, 2.0),
        args=(noise_eigs, q),
        method="bounded",
    )
    sigma_sq: float = float(result.x)
    lambda_min: float = sigma_sq * (1.0 - np.sqrt(q)) ** 2
    lambda_max: float = sigma_sq * (1.0 + np.sqrt(q)) ** 2

    return sigma_sq, lambda_min, lambda_max


def denoise_correlation_matrix(
    C: np.ndarray,
    q: float,
    n_factors_max: int,
) -> Tuple[np.ndarray, int, float]:
    
    N = C.shape[0]

    eigenvalues_asc, eigenvectors_asc = np.linalg.eigh(C)
    eigenvalues = eigenvalues_asc[::-1].copy()
    eigenvectors = eigenvectors_asc[:, ::-1].copy()

    sigma_sq, lambda_min, lambda_max = fit_marchenko_pastur(eigenvalues, q)

    signal_mask = eigenvalues > lambda_max
    n_signal = int(signal_mask.sum())
    n_factors = min(n_signal, n_factors_max)

    eigenvalues_clean = eigenvalues.copy()
    noise_mask = ~signal_mask
    if noise_mask.sum() > 0:
        noise_mean = eigenvalues[noise_mask].mean()
        eigenvalues_clean[noise_mask] = noise_mean
    if n_signal > n_factors_max:
        extra_mean = eigenvalues_clean[noise_mask].mean() if noise_mask.sum() > 0 else 0.0
        eigenvalues_clean[n_factors_max:n_signal] = extra_mean

    # Reconstruct:  C_clean = Q diag(λ_clean) Qᵀ
    C_clean = (eigenvectors * eigenvalues_clean[np.newaxis, :]) @ eigenvectors.T

    # Rescale diagonal to 1 (valid correlation matrix)
    d = np.sqrt(np.diag(C_clean))
    d[d < 1e-12] = 1.0
    D_inv = np.diag(1.0 / d)
    C_clean = D_inv @ C_clean @ D_inv
    C_clean = 0.5 * (C_clean + C_clean.T)  # enforce symmetry

    console.print(
        f"  [cyan]RMT[/cyan]: σ²={sigma_sq:.4f}  λ_max={lambda_max:.4f}  "
        f"signal={n_signal}  factors_kept={n_factors}"
    )
    return C_clean, n_factors, lambda_max




def build_factor_model(
    returns: pd.DataFrame,
    C_clean: np.ndarray,
    n_factors: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
   
    R: np.ndarray = returns.values
    T, N = R.shape

    # Standardise column-wise
    mu = R.mean(axis=0)
    sigma = R.std(axis=0, ddof=1)
    sigma[sigma < 1e-12] = 1.0
    R_std = (R - mu) / sigma

    # Eigenvectors of C_clean (descending order)
    evals_asc, evecs_asc = np.linalg.eigh(C_clean)
    idx_desc = np.argsort(evals_asc)[::-1]
    W = evecs_asc[:, idx_desc][:, :n_factors]          # N × k

    # Factor returns  F = R_std @ W  (ref [4], Eq. 3)
    F = R_std @ W                                      # T × k

    # OLS:  β = (FᵀF)⁻¹ Fᵀ R_std  via lstsq
    betas, _, _, _ = np.linalg.lstsq(F, R_std, rcond=None)
    betas = betas.T                                    # N × k

    R_hat = F @ betas.T                                # T × N
    epsilon = R_std - R_hat                            # T × N  idiosyncratic residuals

    # Cumulative residuals — the mean-reverting spread
    cum_eps = np.cumsum(epsilon, axis=0)

    residuals_df = pd.DataFrame(epsilon, index=returns.index, columns=returns.columns)
    cum_residuals_df = pd.DataFrame(cum_eps, index=returns.index, columns=returns.columns)

    return betas, residuals_df, cum_residuals_df





def fit_ou_process(X: np.ndarray) -> Tuple[float, float, float, float]:
  
    DT: float = 1.0 / 252.0

    if len(X) < 10:
        return 0.0, 0.0, 1.0, float("inf")

    Y = X[1:]       # X_t
    Z = X[:-1]      # X_{t-1}

    # OLS:  Y = a + b·Z
    n = len(Y)
    Z_aug = np.column_stack([np.ones(n), Z])
    coeffs, _, _, _ = np.linalg.lstsq(Z_aug, Y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    # Stationarity check
    if b <= 0.0 or b >= 1.0:
        return 0.0, 0.0, 1.0, float("inf")

    # OU mapping  (ref [4], Eq. 8–11)
    kappa: float = -np.log(b) / DT
    m: float = a / (1.0 - b)

    eta = Y - (a + b * Z)
    sigma_eta: float = float(np.std(eta, ddof=1))
    sigma_eq: float = sigma_eta / np.sqrt(1.0 - b ** 2)

    # half-life in trading days
    half_life: float = np.log(2.0) / kappa

    if sigma_eq < 1e-12:
        sigma_eq = 1e-12

    return kappa, m, sigma_eq, half_life


def compute_s_scores(
    cum_residuals: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    T, N = cum_residuals.shape
    window: int = config.OU_WINDOW
    cols = cum_residuals.columns
    idx = cum_residuals.index

    s_arr = np.full((T, N), np.nan)
    k_arr = np.full((T, N), np.nan)
    hl_arr = np.full((T, N), np.nan)
    seq_arr = np.full((T, N), np.nan)

    vals = cum_residuals.values

    for t in range(window, T):
        for j in range(N):
            segment = vals[t - window: t + 1, j]
            kappa, m, sigma_eq, half_life = fit_ou_process(segment)
            k_arr[t, j] = kappa
            hl_arr[t, j] = half_life
            seq_arr[t, j] = sigma_eq
            s_arr[t, j] = (vals[t, j] - m) / sigma_eq

    return (
        pd.DataFrame(s_arr, index=idx, columns=cols),
        pd.DataFrame(k_arr, index=idx, columns=cols),
        pd.DataFrame(hl_arr, index=idx, columns=cols),
        pd.DataFrame(seq_arr, index=idx, columns=cols),
    )




def generate_signals(
    s_scores: pd.DataFrame,
    kappa_df: pd.DataFrame,
    half_life_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
   
    T, N = s_scores.shape
    signals = np.zeros((T, N), dtype=np.float64)

    s = s_scores.values
    k = kappa_df.values
    hl = half_life_df.values

    for t in range(1, T):
        for j in range(N):
            si = s[t, j]
            ki = k[t, j]
            hli = hl[t, j]
            prev = signals[t - 1, j]

            if np.isnan(si) or np.isnan(ki) or np.isnan(hli):
                signals[t, j] = 0.0
                continue

            # OU validity filter
            ou_valid = (
                ki >= config.MIN_KAPPA
                and config.MIN_HALF_LIFE_DAYS <= hli <= config.MAX_HALF_LIFE_DAYS
            )

            if not ou_valid:
                signals[t, j] = 0.0
                continue

            # Exit condition
            if abs(si) < config.S_SCORE_CLOSE:
                signals[t, j] = 0.0
                continue

            # Entry (only from flat)
            if prev == 0.0:
                if si < -config.S_SCORE_OPEN:
                    signals[t, j] = 1.0     # long — reversion up expected
                elif si > config.S_SCORE_OPEN:
                    signals[t, j] = -1.0    # short — reversion down expected
                else:
                    signals[t, j] = 0.0
            else:
                signals[t, j] = prev        # hold existing position

    return pd.DataFrame(signals, index=s_scores.index, columns=s_scores.columns)


def compute_position_sizes(
    signals: pd.DataFrame,
    s_scores: pd.DataFrame,
    kappa_df: pd.DataFrame,
    sigma_eq_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
  
    DT: float = 1.0 / 252.0

    s_vals = np.abs(s_scores.values)
    k_vals = kappa_df.values
    seq_vals = sigma_eq_df.values
    sig_vals = signals.values

    # Kelly fraction: f* = |s| × (1 − e^{−κΔt}) / σ_eq  (ref [5])
    with np.errstate(over="ignore", invalid="ignore"):
        kelly = s_vals * (1.0 - np.exp(-k_vals * DT)) / np.where(seq_vals > 1e-12, seq_vals, 1.0)

    kelly = np.nan_to_num(kelly, nan=0.0, posinf=0.0, neginf=0.0)
    f_half = config.HALF_KELLY * np.clip(kelly, 0.0, 1.0)

    positions = sig_vals * f_half

    # Dollar-neutral normalisation
    row_abs_sum = np.abs(positions).sum(axis=1, keepdims=True)
    row_abs_sum = np.where(row_abs_sum < 1e-12, 1.0, row_abs_sum)
    positions = positions / row_abs_sum

    return pd.DataFrame(positions, index=signals.index, columns=signals.columns)




def run_walk_forward_backtest(
    returns: pd.DataFrame,
    config: Config,
    use_rmt: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
   
    label = "RMT" if use_rmt else "Baseline"
    T_total, N = returns.shape
    est = config.ESTIMATION_WINDOW
    tst = config.TEST_WINDOW

    all_pnl: List[pd.Series] = []
    fold_records: List[Dict] = []

    n_folds = (T_total - est) // tst
    if n_folds < 1:
        raise ValueError("Not enough data for even one fold.")

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold]{label}[/bold] walk-forward"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} folds"),
        console=console,
    ) as prog:
        task = prog.add_task("Backtesting", total=n_folds)

        for fold_idx in range(n_folds):
            train_start = fold_idx * tst
            train_end = train_start + est
            test_start = train_end
            test_end = min(test_start + tst, T_total)

            if test_end <= test_start:
                prog.update(task, advance=1)
                continue

            train_ret = returns.iloc[train_start:train_end]
            test_ret = returns.iloc[test_start:test_end]

            # ---- Estimation phase ----------------------------------------
            C_emp = np.corrcoef(train_ret.values, rowvar=False)
            q = N / est

            if use_rmt:
                C_used, n_factors, lam_max = denoise_correlation_matrix(
                    C_emp, q, config.N_FACTORS_MAX
                )
            else:
                n_factors = min(5, N - 1)
                C_used = C_emp
                lam_max = 0.0

            if n_factors < 1:
                n_factors = 1

            betas, train_resid, train_cum = build_factor_model(
                train_ret, C_used, n_factors
            )

           
            ou_params: Dict[str, Tuple[float, float, float, float]] = {}
            for col in train_cum.columns:
                seg = train_cum[col].values[-config.OU_WINDOW:]
                ou_params[col] = fit_ou_process(seg)

            R_test = test_ret.values
            mu_train = train_ret.values.mean(axis=0)
            sig_train = train_ret.values.std(axis=0, ddof=1)
            sig_train[sig_train < 1e-12] = 1.0
            R_test_std = (R_test - mu_train) / sig_train

            evals_asc, evecs_asc = np.linalg.eigh(C_used)
            idx_desc = np.argsort(evals_asc)[::-1]
            W = evecs_asc[:, idx_desc][:, :n_factors]

            F_test = R_test_std @ W
            R_hat_test = F_test @ betas.T
            eps_test = R_test_std - R_hat_test

            last_cum = train_cum.values[-1, :]
            cum_test = np.cumsum(eps_test, axis=0) + last_cum

            # ---- S-scores on test window ----------
            T_test = cum_test.shape[0]
            s_test = np.zeros((T_test, N))
            k_test = np.zeros((T_test, N))
            hl_test = np.zeros((T_test, N))
            seq_test = np.zeros((T_test, N))

            for j, col in enumerate(test_ret.columns):
                kappa, m, sigma_eq, half_life = ou_params[col]
                k_test[:, j] = kappa
                hl_test[:, j] = half_life
                seq_test[:, j] = sigma_eq
                s_test[:, j] = (cum_test[:, j] - m) / max(sigma_eq, 1e-12)

            s_df = pd.DataFrame(s_test, index=test_ret.index, columns=test_ret.columns)
            k_df = pd.DataFrame(k_test, index=test_ret.index, columns=test_ret.columns)
            hl_df = pd.DataFrame(hl_test, index=test_ret.index, columns=test_ret.columns)
            seq_df = pd.DataFrame(seq_test, index=test_ret.index, columns=test_ret.columns)

            # ---- Signals & positions -----
            sigs = generate_signals(s_df, k_df, hl_df, config)
            pos = compute_position_sizes(sigs, s_df, k_df, seq_df, config)

            # ---- P&L = positions × next-day returns − costs 
            pos_vals = pos.values[:-1, :]
            ret_vals = test_ret.values[1:, :]

            if pos_vals.shape[0] == 0:
                prog.update(task, advance=1)
                continue

            daily_gross = (pos_vals * ret_vals).sum(axis=1)

            # Transaction costs on turnover
            if pos_vals.shape[0] > 1:
                delta_pos = np.diff(pos.values[:pos_vals.shape[0] + 1], axis=0)
                turnover = np.abs(delta_pos).sum(axis=1)
            else:
                turnover = np.abs(pos_vals).sum(axis=1)

            tc = turnover * (config.TRANSACTION_COST_BP / 10_000.0)
            daily_net = daily_gross - tc[:len(daily_gross)]

            pnl_idx = test_ret.index[1: 1 + len(daily_net)]
            pnl_series = pd.Series(daily_net, index=pnl_idx)
            all_pnl.append(pnl_series)

            fold_records.append({
                "fold": fold_idx,
                "train_start": train_ret.index[0],
                "test_start": test_ret.index[0],
                "test_end": test_ret.index[-1],
                "n_factors": n_factors,
                "lambda_max": lam_max,
            })

            prog.update(task, advance=1)

    if not all_pnl:
        raise RuntimeError(f"No P&L produced for {label} backtest.")

    combined = pd.concat(all_pnl).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    meta = pd.DataFrame(fold_records)
    return combined, meta





def compute_performance_metrics(pnl: pd.Series) -> Dict[str, float]:
   
    equity = (1.0 + pnl).cumprod()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    n_days = len(pnl)
    ann = 252.0

    ann_return = float((1.0 + total_return) ** (ann / max(n_days, 1)) - 1.0)
    ann_vol = float(pnl.std() * np.sqrt(ann))
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0

    downside = pnl[pnl < 0]
    ds_vol = float(downside.std() * np.sqrt(ann)) if len(downside) > 1 else 1e-12
    sortino = ann_return / ds_vol if ds_vol > 1e-12 else 0.0

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min())

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    win_rate = float((pnl > 0).sum() / max(len(pnl), 1))

    gp = float(pnl[pnl > 0].sum()) if (pnl > 0).any() else 0.0
    gl = float(abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 1e-12
    pf = gp / gl if gl > 1e-12 else 0.0

    var_95 = float(np.percentile(pnl, 5))
    cvar_95 = float(pnl[pnl <= var_95].mean()) if (pnl <= var_95).any() else var_95

    return {
        "total_return": total_return,
        "annualised_return": ann_return,
        "annualised_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def plot_eigenspectrum(
    eigenvalues: np.ndarray,
    q: float,
    sigma_sq: float,
    lambda_max: float,
    save_path: str = "fig1_eigenspectrum.png",
) -> None:
   
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(eigenvalues, bins=50, density=True, alpha=0.6,
            color="steelblue", edgecolor="white", label="Empirical eigenvalues")

    x_grid = np.linspace(0.01, max(eigenvalues) * 1.1, 500)
    mp_curve = marchenko_pastur_pdf(x_grid, q, sigma_sq)
    ax.plot(x_grid, mp_curve, "r-", lw=2.0, label="Marchenko-Pastur PDF")

    ax.axvline(lambda_max, color="grey", ls="--", lw=1.5,
               label=f"λ_max = {lambda_max:.3f}")

    signal_eigs = eigenvalues[eigenvalues > lambda_max]
    ax.plot(signal_eigs, np.zeros_like(signal_eigs) + 0.02, "go",
            ms=8, zorder=5, label=f"Signal ({len(signal_eigs)} eigenvalues)")

    ax.set_xlabel("Eigenvalue λ", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Correlation Matrix Eigenspectrum — RMT Denoising", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    console.print(f"  [green]✓ Saved {save_path}[/green]")


def plot_results(
    pnl_rmt: pd.Series,
    pnl_base: pd.Series,
    spy_returns: Optional[pd.Series],
    save_path: str = "fig2_results.png",
) -> None:
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    eq_rmt = (1.0 + pnl_rmt).cumprod()
    eq_base = (1.0 + pnl_base).cumprod()

    ax1.plot(eq_rmt.index, eq_rmt.values, "b-", lw=1.3, label="RMT Stat-Arb")
    ax1.plot(eq_base.index, eq_base.values, color="orange", lw=1.1,
             label="Plain PCA Baseline")

    if spy_returns is not None:
        common = spy_returns.index.intersection(pnl_rmt.index)
        if len(common) > 10:
            eq_spy = (1.0 + spy_returns.loc[common]).cumprod()
            ax1.plot(eq_spy.index, eq_spy.values, "grey", ls="--", lw=1.0,
                     label="Buy-Hold SPY")

    ax1.set_ylabel("Cumulative Equity", fontsize=11)
    ax1.set_title("Walk-Forward Performance: RMT vs Baseline vs SPY", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Drawdown for RMT
    rm = eq_rmt.cummax()
    dd = (eq_rmt - rm) / rm
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    console.print(f"  [green]✓ Saved {save_path}[/green]")


def print_comparison_table(
    metrics_rmt: Dict[str, float],
    metrics_base: Dict[str, float],
) -> None:
    """Rich table: RMT vs Baseline across all metrics."""
    table = Table(title="Performance Comparison: RMT vs Plain PCA", show_lines=True)
    table.add_column("Metric", style="bold cyan", min_width=22)
    table.add_column("RMT Strategy", justify="right", style="green")
    table.add_column("Baseline (Plain PCA)", justify="right", style="yellow")

    fmt = {
        "total_return": "{:.4%}", "annualised_return": "{:.4%}",
        "annualised_vol": "{:.4%}", "sharpe_ratio": "{:.3f}",
        "sortino_ratio": "{:.3f}", "calmar_ratio": "{:.3f}",
        "max_drawdown": "{:.4%}", "win_rate": "{:.2%}",
        "profit_factor": "{:.3f}", "var_95": "{:.6f}", "cvar_95": "{:.6f}",
    }

    for key in metrics_rmt:
        f = fmt.get(key, "{:.4f}")
        table.add_row(
            key.replace("_", " ").title(),
            f.format(metrics_rmt[key]),
            f.format(metrics_base[key]),
        )
    console.print(table)


def print_summary_panel(metrics: Dict[str, float]) -> None:
    """Compact Rich panel with headline numbers."""
    txt = (
        f"[bold green]Sharpe[/bold green]  {metrics['sharpe_ratio']:.3f}   │   "
        f"[bold yellow]Calmar[/bold yellow]  {metrics['calmar_ratio']:.3f}   │   "
        f"[bold red]Max DD[/bold red]  {metrics['max_drawdown']:.2%}   │   "
        f"[bold cyan]Win Rate[/bold cyan]  {metrics['win_rate']:.1%}"
    )
    console.print(Panel(txt, title="RMT Strategy — Headline Metrics",
                        border_style="bold blue"))





def main() -> None:
    
    cfg = Config()

    console.rule("[bold magenta]Global Stat Arb · RMT-Cleaned Avellaneda-Lee[/bold magenta]")

    # ── [1/6] Fetch data ─────────────────────────────────────────────────
    prices = fetch_universe_data(ALL_TICKERS, cfg.START_DATE, cfg.END_DATE, cfg)

    # ── [2/6] Compute returns ────────────────────────────────────────────
    console.print("[bold cyan][2/6] Computing log returns …[/bold cyan]")
    log_rets = compute_log_returns(prices)
    console.print(f"[green]  ✓ {log_rets.shape[0]} days × {log_rets.shape[1]} assets[/green]")

    # Grab SPY for benchmark
    spy_ret: Optional[pd.Series] = None
    try:
        spy_px = yf.download("SPY", start=cfg.START_DATE, end=cfg.END_DATE,
                             auto_adjust=True, progress=False)
        if not spy_px.empty:
            spy_close = spy_px["Close"] if "Close" in spy_px.columns else spy_px.iloc[:, 0]
            spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    except Exception:
        console.print("[yellow]  SPY benchmark unavailable — skipping.[/yellow]")

    # ── [3/6] RMT walk-forward backtest ──────────────────────────────────
    console.print("[bold cyan][3/6] Running RMT backtest …[/bold cyan]")
    pnl_rmt, meta_rmt = run_walk_forward_backtest(log_rets, cfg, use_rmt=True)

    # ── [4/6] Baseline walk-forward backtest ─────────────────────────────
    console.print("[bold cyan][4/6] Running baseline (plain PCA) backtest …[/bold cyan]")
    pnl_base, meta_base = run_walk_forward_backtest(log_rets, cfg, use_rmt=False)

    # ── [5/6] Metrics + tables ───────────────────────────────────────────
    console.print("[bold cyan][5/6] Computing metrics …[/bold cyan]")
    m_rmt = compute_performance_metrics(pnl_rmt)
    m_base = compute_performance_metrics(pnl_base)

    print_comparison_table(m_rmt, m_base)
    print_summary_panel(m_rmt)

    # ── [6/6] Figures ────────────────────────────────────────────────────
    console.print("[bold cyan][6/6] Generating figures …[/bold cyan]")

    # Eigenspectrum on full sample
    C_full = np.corrcoef(log_rets.values, rowvar=False)
    T_full, N_full = log_rets.shape
    q_full = N_full / T_full
    evals_asc_full, _ = np.linalg.eigh(C_full)
    evals_full = evals_asc_full[::-1]
    sigma_sq_full, _, lam_max_full = fit_marchenko_pastur(evals_full, q_full)

    plot_eigenspectrum(evals_full, q_full, sigma_sq_full, lam_max_full)
    plot_results(pnl_rmt, pnl_base, spy_ret)

    console.rule("[bold green]Done ✓[/bold green]")


if __name__ == "__main__":
    main()
