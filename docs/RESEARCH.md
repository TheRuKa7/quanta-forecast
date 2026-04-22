# /ultraresearch — quanta-forecast

*State of time-series forecasting for a 2026 portfolio. Verify against latest ArXiv + Monash leaderboards before shipping.*

## 1. The four tiers

### Tier 1 — Classical statistics
- **ARIMA / SARIMA** — linear, interpretable, fast. Baseline everyone must beat.
- **ETS (Exponential Smoothing)** — additive/multiplicative, robust on short series
- **Prophet (Meta)** — seasonal-decomposition Bayesian, great for business series with holidays
- **Theta** — simple, ranked #2 in M3 competition
- Best lib: **statsforecast** (Nixtla) — Rust-accelerated, sklearn-style API

### Tier 2 — ML with engineered features
- **LightGBM / XGBoost** with lag, rolling-mean, calendar features
- **Linear with L1/L2** for interpretability baselines
- Best lib: **mlforecast** (Nixtla) — unified ML forecasting interface

### Tier 3 — Deep learning
- **N-BEATS / N-HiTS** — pure DL, no feature engineering, strong on M4
- **TFT (Temporal Fusion Transformer)** — interpretable, attention-based
- **DeepAR** — probabilistic, autoregressive
- **PatchTST** — patch-based transformer, SOTA on long-horizon
- **iTransformer** — inverted-dimension transformer (2023)
- **TSMixer** — MLP-based, fast
- Best lib: **darts** (Unit8) — one API across many DL models; **neuralforecast** (Nixtla)

### Tier 4 — Foundation models (the 2024 paradigm shift)
| Model | Architecture | Size | License | Zero-shot? |
|-------|-------------|------|---------|------------|
| **Chronos** (Amazon) | T5 encoder-decoder | 8M–710M | Apache-2 | ✅ |
| **TimesFM** (Google) | Decoder-only transformer | 200M | Apache-2 | ✅ |
| **Moirai** (Salesforce) | Masked encoder | 14M–311M | CC-BY-NC-4.0 | ✅ |
| **Lag-Llama** (HF) | Decoder-only | 200M | Apache-2 | ✅ |
| **TimeGPT-1** (Nixtla) | Closed API | — | Commercial | ✅ |
| **TimeMoE** | MoE transformer | 50M–2.4B | Apache-2 | ✅ |

**Pick for v1:** Chronos-tiny / Chronos-small — Apache-2, runs on CPU, strong zero-shot.

## 2. Evaluation metrics

| Metric | When |
|--------|------|
| sMAPE | Scale-free, standard in M-competitions |
| MASE | Scale-free, compares to naive |
| RMSE / MAE | Scale-dependent, intuitive |
| CRPS | Probabilistic, proper scoring rule |
| Pinball loss | Quantile-specific |
| Coverage (80/95%) | Calibration of intervals |

## 3. Benchmarks

- **M4** (Makridakis 4) — 100K series across frequencies; gold standard
- **M5** — Walmart demand; hierarchical
- **Monash Archive** — 30+ datasets; standard leaderboards
- **ETT** (Electricity Transformer Temperature) — long-horizon
- **Weather**, **Traffic**, **Exchange** — Informer-paper suite

## 4. Cross-validation strategies

- **Expanding window** (rolling origin) — standard for time series
- **Rolling window** — fixed history length
- **Blocked CV** — when autocorrelation long
- **Gap** — prevent leakage for financial data

## 5. Why foundation models change the game

Pre-2024: every new dataset required training a model from scratch.
Post-2024: foundation models give you a reasonable forecast *without training*.

**Implication for this repo:** the demo story is "upload CSV → get forecast in 3 seconds with zero config." That's a radically different value prop than "wait 20 minutes while I train an LSTM."

## 6. Stock-prediction reality check

Markets are close to weak-form efficient. Public OHLC features have near-zero predictive power for returns. **This repo must not imply otherwise.** The stock demo exists to:
- Showcase the library API on a dataset users recognize
- Demonstrate honest evaluation (directional accuracy ≈ 50%, as expected)
- Educate about survivorship bias, look-ahead, and overfitting

## 7. Open questions

- [ ] Chronos-tiny vs Chronos-small on M4 daily: which Pareto-dominates on CPU?
- [ ] How to handle hierarchical forecasting (retail SKU × store)?
- [ ] Probabilistic outputs: sample-based (Chronos) vs parametric (DeepAR) — unified schema?
- [ ] Cold-start: < 30 observations — which model family is least bad?
