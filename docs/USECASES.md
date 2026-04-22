# USECASES — quanta-forecast

End-to-end flows for a multi-tier forecasting library spanning classical,
ML, deep-learning, and foundation-model forecasters behind one API.

## 1. Personas

| ID | Persona | Context | Primary JTBD |
|----|---------|---------|--------------|
| P1 | **Data-science lead (Anya)** | Retail analytics team; 10k SKUs × daily demand | "Back-test 4 forecaster families on last year's data, pick per-SKU winner" |
| P2 | **FinOps analyst (Luis)** | Tracks AWS spend; wants 14-day forecasts + anomaly bands | "Warn me when next week's spend is outside the 80% CI of the forecast" |
| P3 | **Quant researcher (Mei)** | Intraday equity prediction; skeptical of hype | "Give me honest baselines; don't tell me LSTM beat ARIMA unless the test is clean" |
| P4 | **Product-analytics PM (Jamal)** | Wants WAU projections with uncertainty | "Show me P10/P50/P90 for WAU over the next 6 weeks; update weekly" |
| P5 | **ML eng evaluating foundation models (Kara)** | Considering Chronos / TimesFM / Moirai for prod | "Are zero-shot foundation forecasters actually usable without fine-tuning?" |

## 2. Jobs-to-be-done

JTBD-1. **Unified API** across ARIMA, LightGBM, TFT/PatchTST, Chronos — one `fit/predict` contract.
JTBD-2. **Probabilistic outputs** — every forecaster returns quantiles, not just a point.
JTBD-3. **Honest back-testing** — time-series CV, no leakage, no lookahead.
JTBD-4. **Zero-shot foundation models** for cold-start series.
JTBD-5. **Shippable artefact** — Streamlit demo + CLI + Python package.
JTBD-6. **Stock-market honesty** — ship a disclaimer and refuse to pretend this beats the market.

## 3. End-to-end flows

### Flow A — Anya's per-SKU champion-challenger

1. Anya loads a long-format parquet (10k SKUs × 730 days).
2. Runs `quanta bench --family classical,ml,deep --horizon 14 --cv rolling --folds 5`.
3. Library spawns parallel jobs per SKU (joblib / Ray).
4. Output: a `leaderboard.parquet` with MASE/CRPS per SKU × family.
5. `quanta champion --select best-by-MASE` writes `champions.json` mapping SKU → model.
6. Daily cron runs `quanta predict --config champions.json` to produce next-day forecasts.

### Flow B — Luis's FinOps anomaly band

1. CSV of daily AWS spend (last 180 days) uploaded to Streamlit demo.
2. Luis picks `chronos-small` (zero-shot), horizon 14, confidence 80%.
3. Output: P10/P50/P90 forecast + shaded plot.
4. Streamlit "download Slack webhook snippet" button emits a monitoring script:
   if actual spend > P90, post to Slack.

### Flow C — Mei's honest equity baseline

1. Mei runs `quanta bench --data SPY.csv --family classical,deep --leak-check strict`.
2. Library enforces: test set strictly after train; feature engineering fit on train only.
3. Result: naive seasonal baseline is within 5% of deep models; Mei writes a blog post
   titled "No, your LSTM did not beat a seasonal naive baseline".
4. She cites `docs/STOCK_DISCLAIMER.md` as reference.

### Flow D — Jamal's WAU projection dashboard

1. Weekly cron pulls WAU from analytics warehouse → parquet.
2. `quanta predict --model tft --quantiles 0.1,0.5,0.9 --horizon 6` (weeks).
3. Output piped to the PM dashboard (Metabase) as a table + chart.
4. Jamal pastes P50 + P10/P90 band into his monthly review deck.

### Flow E — Kara evaluates foundation forecasters

1. Kara downloads 20 real business time series (spend, MAU, queue length, …).
2. Runs `quanta zero-shot --model chronos-small,timesfm-200m,moirai-base --horizon 14`.
3. Library returns per-series MASE; no training, cold start only.
4. Ships a notebook documenting which families help and where they lose to ARIMA.

### Flow F — Contributor adds a new forecaster

1. Implements `NixtlaWhatever(Forecaster)` conforming to protocol (`fit/predict` + `ForecastOutput`).
2. Registers via `@register_forecaster("nixtla-whatever")`.
3. `uv run pytest` asserts conformance: accepts a `(y, exog)` pair, returns quantiles of correct shape.
4. PR passes CI; leaderboard auto-includes it in the next `bench` run.

## 4. Acceptance scenarios

```gherkin
Scenario: Unified protocol across families
  Given forecasters "arima", "lightgbm", "patchtst", "chronos-small"
  When I fit each on the same series y with horizon=14
  Then each returns a ForecastOutput with fields {mean, quantiles, horizon, model_name}
  And quantiles.shape == (horizon, len(quantile_levels))

Scenario: Leak check catches lookahead features
  Given a feature column that uses future rows
  When I run quanta bench --leak-check strict
  Then the run fails with LeakageError and names the offending column

Scenario: Zero-shot Chronos on a series it has never seen
  Given a 90-day series never included in Chronos training
  When I call quanta zero-shot --model chronos-small
  Then it returns a forecast within 10 seconds on CPU
  And quantiles are monotonic (P10 <= P50 <= P90)

Scenario: Stock disclaimer is displayed
  When the Streamlit demo loads with data labelled "equity"
  Then a disclaimer banner is shown above the chart
  And the banner links to docs/STOCK_DISCLAIMER.md
```

## 5. Non-use-cases

- Alpha generation for trading (explicitly disclaimed)
- Tick-level / sub-second forecasting (out of scope; different stack)
- Causal inference / counterfactuals
- Forecasting with zero history (we return a configured fallback, not magic)
