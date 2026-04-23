

#  Global Statistical Arbitrage with RMT-Denoised PCA

## Overview

This project implements a **production-grade global statistical arbitrage strategy** based on the Avellaneda-Lee (2010) framework, with a key enhancement using **Random Matrix Theory (RMT)** to improve signal quality.

The core idea is simple in theory and messy in practice:
markets are noisy, correlations are unstable, and most “signals” are just statistical hallucinations. This strategy tries to fix that.

Instead of blindly applying PCA on noisy correlation matrices (like most implementations), this approach **filters out noise using the Marchenko-Pastur distribution** before extracting factors. The result is a more stable factor model and cleaner mean-reversion signals.

---

## What It Does

At a high level, the strategy:

1. Builds a **global cross-asset universe**:

   * Indian equities (Nifty 50 subset)
   * US equities (S&P 100 subset)
   * Commodity ETFs
   * FX / macro ETFs

2. Computes **log returns** and constructs a correlation matrix

3. Applies **RMT-based denoising**:

   * Uses the Marchenko-Pastur law to identify noise eigenvalues
   * Filters out statistically insignificant components
   * Reconstructs a cleaner correlation matrix

4. Performs **PCA on the cleaned matrix** to extract systematic factors

5. Removes these factors via regression to obtain **idiosyncratic residuals**

6. Models residuals as **Ornstein-Uhlenbeck (OU) processes**
   → capturing mean-reversion dynamics

7. Generates **trading signals (s-scores)**:

   * Long when undervalued
   * Short when overvalued

8. Sizes positions using a **half-Kelly framework**

9. Evaluates performance using a **walk-forward backtest**
   (out-of-sample, no cheating… mostly)

---

## Why I Built This

Most statistical arbitrage strategies break in the real world because they rely on **unstable correlation structures**.

A key insight from research (Laloux et al., 1999) is that:

> ~90% of eigenvalues in empirical correlation matrices are just noise.

Yet, standard PCA-based strategies still treat all of them as signal.
That’s like trusting every WhatsApp forward equally.

This project was built to:

* Reduce **noise-driven signals**
* Improve **out-of-sample robustness**
* Explore the intersection of:

  * statistical arbitrage
  * random matrix theory
  * cross-asset macro structure

The RMT step consistently led to **more stable factors and cleaner residuals**, which translated into better trading performance compared to a plain PCA baseline.

---

## Key Innovation

###  RMT + Stat Arb (the whole point)

This project combines:

* **Avellaneda-Lee statistical arbitrage**
* **Random Matrix Theory denoising**
* **Cross-asset global universe**

While each idea exists individually, combining them in a **practical, end-to-end trading system** is far less common.

The intuition:

* PCA on raw correlations → noisy, unstable factors
* PCA on RMT-cleaned correlations → **economically meaningful structure**

Less noise = fewer fake signals = less pain.

---

## Strategy Pipeline

```
Prices → Returns → Correlation Matrix
        ↓
   RMT Denoising
        ↓
       PCA
        ↓
   Factor Model
        ↓
   Residuals (ε)
        ↓
 Ornstein-Uhlenbeck Fit
        ↓
     S-Scores
        ↓
  Signal Generation
        ↓
 Position Sizing (Half-Kelly)
        ↓
 Walk-Forward Backtest
```

---

## Tech Stack

* Python
* numpy / pandas
* scipy
* scikit-learn
* yfinance (data)
* matplotlib (visualization)
* rich (CLI output)

No paid APIs. No magic black boxes.

---

## Features

*  RMT-based correlation denoising (from scratch)
*  PCA factor model (eigenportfolios)
*  OU process calibration for mean reversion
*  S-score signal framework
*  Half-Kelly position sizing
*  Transaction cost modeling
*  Walk-forward backtesting (no look-ahead bias)
*  Performance analytics (Sharpe, drawdown, etc.)
*  Rich terminal output + visualizations

---

## Results (Summary)

* Improved **Sharpe ratio** vs baseline PCA strategy
* Lower **drawdowns** due to cleaner signals
* More stable factor structure across time
* Better **out-of-sample consistency**

Because apparently removing noise helps. Who knew.

---

## How to Run

### 1. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib yfinance rich
```

### 2. Run the script

```bash
python strategy.py
```

---

## Outputs

*  Equity curve comparison (RMT vs baseline)
*  Eigenvalue spectrum (RMT validation)
*  S-score heatmap
*  Performance dashboard
*  Rich terminal tables with metrics

---

## File Structure

```
strategy.py        # Full end-to-end implementation
README.md          # You’re reading it
```

---

## Future Improvements

* Incorporate **intraday data**
* Add **transaction cost models with slippage**
* Extend to **crypto / alt data**
* Replace OU with **non-linear mean-reversion models**
* Explore **online / adaptive RMT filtering**

---

## Author

**Arya**
Quant enthusiast focused on systematic trading and market structure

---

## Final Thought

Most quant strategies don’t fail because the idea is wrong.
They fail because the data is noisy and the model believes it anyway.

This project is an attempt to fix that at the source.

Also, if this thing loses money in production, I will simply blame the market regime and pretend it’s temporary.
