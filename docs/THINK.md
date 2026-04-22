# /ultrathink — quanta-forecast

## 1. Why rebuild this?

The 2022 stock-predictor was a common portfolio trope — LSTM on AAPL close prices, claim high accuracy by measuring MSE on normalized prices (trivially good if you predict "tomorrow ≈ today"). That's a **negative signal** to a modern ML hiring manager.

**The rebuild flips it:**
1. **Broader scope** — general forecasting toolkit, not a stock oracle
2. **Foundation models** — demonstrates 2024+ awareness
3. **Honest evaluation** — directional accuracy on stocks ≈ random, shown transparently
4. **Library-quality API** — reusable, tested, documented, benchmarked

## 2. Why the four-tier architecture?

Most forecasting repos pick one tier and ignore the others. Hiring managers want to see:
- You know when classical beats deep (short series, interpretability needs)
- You know when ML beats deep (tabular with engineered features)
- You know when deep wins (long series, complex seasonality, multivariate)
- You know foundation models exist (and their trade-offs)

The unified `Forecaster` interface is the pedagogical artifact — it says *I think in abstractions, not one-off notebooks*.

## 3. Why Chronos as the hero?

- **Apache-2** — clean licensing
- **CPU-viable** — tiny model runs on laptops
- **Genuinely zero-shot** — no fine-tuning needed for reasonable accuracy
- **Amazon backed** — recruiter-credible
- **Text-tokenized forecasting** — conceptually novel (series → tokens → T5), makes for great interview chat

## 4. PM framing

- **Problem:** SMBs need forecasts (inventory, demand, staffing) but lack ML teams
- **Status quo:** Excel + gut, or expensive enterprise tools (RELEX, Blue Yonder)
- **Insight:** Foundation models just dropped the skills-cost to near-zero
- **MVP:** Upload CSV → get probabilistic forecast + backtest report → download
- **Monetization:** API pricing per series-month; premium tier = custom fine-tuning
- **Why now:** Chronos + TimesFM released 2024; GPU inference costs collapsing

## 5. Tradeoffs

| Decision | Alt | Why |
|----------|-----|-----|
| darts (unified DL) | train each from scratch | Speed + consistency |
| Nixtla statsforecast | statsmodels directly | 10× faster |
| Streamlit | Next.js | Data-viz is Streamlit's sweet spot |
| Polars | Pandas | 5-10× faster on large frames |
| Chronos-tiny default | Chronos-large | Runs on laptops |
| M4 daily subset | full M4 (100K series) | Fits in CI, repeatable |

## 6. Honest skepticism (the stock demo)

The stock demo lives in the repo for one reason: people recognize AAPL, they'll open the notebook. But it must **not** imply alpha:

- Directional accuracy reported honestly (≈ 50%)
- Evaluated on OHLC features only (no news/fundamentals)
- `docs/THINK.md` explains EMH, survivorship bias, look-ahead bias
- Notebook labeled "Case study in honest evaluation"

This *is itself* a hiring signal — candidates who acknowledge limitations are more trustworthy than those who oversell.

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Foundation model weights too big | Use Chronos-tiny (8M params, ~30MB) |
| Stock claim misinterpreted | Explicit "not financial advice" banner + honest results |
| CPU-only inference too slow for demo | Cache sample forecasts; lazy-load models |
| API key requirements | None — all models are open-weight |
| Deep model training flakiness | Pre-trained checkpoints committed via HF Hub |

## 8. Cross-repo glue

- Shared OpenTelemetry traces format with `idas-scene-ai` (`docs/OBSERVABILITY.md`)
- Streamlit demo references `doc-rag` for "ask a question about your forecast" (v2)
- `pm-copilot` can call `/forecast` as a tool for agentic demand planning

## 9. Interview talking points

- *"How do you pick a model?"* — tier selection by data volume, noise, interpretability needs, horizon
- *"When does deep lose?"* — short series (< 500 obs), strong seasonality already captured by ETS, need for explainability
- *"How do foundation models work for time series?"* — Chronos tokenizes numeric values, trains T5 on them; TimesFM uses patch embeddings
- *"How do you evaluate probabilistic forecasts?"* — CRPS, pinball loss, coverage — not just RMSE
- *"What's wrong with a stock LSTM?"* — EMH, look-ahead bias, normalization leaks, reporting MSE on prices instead of returns
