# DFD — quanta-forecast

## Level 0 — Context

```mermaid
flowchart LR
  USER[Analyst / DS]
  CSV[CSV / Parquet source]
  WH[Data warehouse<br/>Snowflake / BQ]
  HF[HF Hub<br/>Chronos / TimesFM / Moirai weights]
  QF((quanta-forecast))
  SINK[Streamlit / dashboard / Slack webhook]

  USER -- CLI / Streamlit UI --> QF
  CSV  -- bytes --> QF
  WH   -- SQL rows --> QF
  HF   -- foundation weights --> QF
  QF   -- forecast CSV / plot / JSON --> SINK
  QF   -- leaderboard.parquet --> USER
```

## Level 1 — Functional decomposition

```mermaid
flowchart TD
  subgraph Ingest
    LD[1.0 Loader<br/>CSV / Parquet / SQL]
    LK[1.1 Leak detector]
    FT[1.2 Feature builder<br/>lag + rolling]
  end

  subgraph Modeling
    REG[2.0 Forecaster registry]
    FIT[2.1 Fit dispatcher]
    PRED[2.2 Predict dispatcher]
  end

  subgraph Evaluation
    BT[3.0 Backtester<br/>rolling-origin CV]
    MET[3.1 Metrics<br/>MASE / CRPS / sMAPE]
    LB[3.2 Leaderboard writer]
  end

  subgraph Present
    CLI[4.0 Typer CLI]
    UI[4.1 Streamlit demo]
    EXP[4.2 Exporter<br/>CSV / parquet / plotly]
  end

  subgraph Stores
    RUNS[[.quanta/runs<br/>JSON + artefacts]]
    CACHE[[weights cache<br/>~/.cache/quanta]]
  end

  LD --> LK --> FT
  FT --> FIT
  REG --> FIT
  FIT --> RUNS
  FIT --> PRED
  PRED --> MET
  BT --> MET --> LB
  PRED --> EXP
  LB --> EXP
  CLI -- commands --> FIT
  CLI -- commands --> BT
  UI -- forms --> FIT
  UI -- forms --> PRED
  PRED -- foundation weights --> CACHE
```

## Level 2 — `quanta bench` data flow

```mermaid
sequenceDiagram
  autonumber
  participant U as User CLI
  participant L as Loader
  participant B as Backtester
  participant R as Registry
  participant F as Forecaster
  participant M as Metrics
  participant O as Leaderboard
  U->>L: load data.csv
  L-->>U: pd.Series y
  U->>B: run(y, [arima, lightgbm, patchtst, chronos-small], cfg)
  loop for each forecaster in family list
    loop for each rolling origin
      B->>R: build(name)
      R-->>B: Forecaster instance
      B->>F: fit(y[:origin], exog[:origin])
      B->>F: predict(horizon, quantiles)
      F-->>B: ForecastOutput
      B->>M: score(y[origin:origin+h], output)
      M-->>B: {mase, crps, smape}
    end
  end
  B->>O: write leaderboard.parquet
  O-->>U: path
```

## Level 2 — Foundation-model zero-shot path

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant Q as quanta
  participant C as Chronos adapter
  participant H as HF Hub
  participant K as Local weights cache
  U->>Q: quanta zero-shot --model chronos-small --data x.csv
  Q->>C: build("chronos-small")
  C->>K: check cache
  alt cache miss
    K->>H: download
    H-->>K: weights
  end
  C->>C: load pipeline
  U->>C: predict(horizon, quantiles)
  C-->>U: ForecastOutput (mean, quantiles, samples)
  Q->>U: plot + CSV
```

## Data stores

| Store | Purpose | Retention |
|-------|---------|-----------|
| `.quanta/runs/` | Per-run JSON (seed, versions, data-hash) + artefacts | 30 days (configurable) |
| `~/.cache/quanta/weights/` | Foundation-model weight cache | Until manually purged |
| `leaderboards/*.parquet` | Cross-family evaluation results | User-controlled |

## Invariants & contracts

- `ForecastOutput.quantiles.shape == (horizon, len(quantile_levels))`
- `ForecastOutput.quantiles` is **monotonic** along the quantile axis (asserted in validator).
- No feature column may depend on rows with `t > origin` in strict mode.
- Weight downloads are **content-addressed**: the cache key includes SHA of config + weights.

## Trust / sensitivity

- Data loaded from user sources never leaves the local process.
- Foundation weights downloaded from HF over HTTPS; checksums verified.
- Opt-in telemetry (off by default) reports only anonymised run metadata, never user data.
