# RFC-001 — quanta-forecast library architecture

**Author:** Rushil Kaul · **Status:** Draft · **Target release:** P1–P2

## 1. Summary

Design the library around a single `Forecaster` protocol, a `ForecastOutput`
dataclass, a `Backtester` engine that enforces time-ordered CV, and a CLI/Streamlit
surface. Forecaster families are **extras** so a user installing only the
classical family has no torch dependency.

## 2. Context

See `RESEARCH.md` for family landscape and benchmark comparison, `PRD.md` for
requirements. This RFC pins module boundaries, public types, and extension
points.

## 3. Detailed design

### 3.1 Core types

```python
# src/quanta/types.py
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(slots=True)
class ForecastOutput:
    mean: np.ndarray                      # shape (H,)
    quantiles: np.ndarray                 # shape (H, Q) — monotonically increasing
    quantile_levels: tuple[float, ...]    # e.g. (0.1, 0.5, 0.9)
    horizon: int
    model_name: str
    sample_paths: np.ndarray | None = None  # optional (N, H) Monte Carlo draws

    def to_frame(self, index) -> pd.DataFrame: ...
    def interval(self, lo=0.1, hi=0.9) -> tuple[np.ndarray, np.ndarray]: ...
```

```python
# src/quanta/base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Forecaster(Protocol):
    name: str                             # unique registry key
    family: Literal["classical","ml","deep","foundation"]

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> None: ...
    def predict(self, horizon: int, exog_future: pd.DataFrame | None = None,
                quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)) -> ForecastOutput: ...
```

### 3.2 Registry

```python
_REGISTRY: dict[str, type[Forecaster]] = {}

def register_forecaster(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def build(name: str, **kwargs) -> Forecaster:
    return _REGISTRY[name](**kwargs)
```

Family modules import and register themselves; loading a family is a plugin-style
import (`quanta.families.classical` etc.). Foundation-model loaders are lazy:
the weight download happens on first `fit` call.

### 3.3 Backtester

```python
# src/quanta/backtest.py
@dataclass
class BacktestConfig:
    horizon: int
    folds: int                   # number of rolling origins
    step: int                    # stride between origins
    min_train: int
    metrics: list[str]           # ["mase","crps","smape"]
    leak_check: Literal["off","warn","strict"] = "strict"

class Backtester:
    def run(self, y: pd.Series, forecaster_name: str, cfg: BacktestConfig) -> pd.DataFrame: ...
```

- **Leak check** scans any engineered feature column for dependence on rows
  with `t > origin`; any match fails in `strict`.
- **Determinism** — sets `np.random.seed`, `torch.manual_seed`, logs lib versions.
- **Parallelism** — `joblib.Parallel` over folds; Ray in v2.

### 3.4 Feature engineering (ML family)

```python
# src/quanta/features.py
class LagFeatures(BaseFeature):
    lags: list[int]
    def build(self, y: pd.Series, up_to: pd.Timestamp) -> pd.DataFrame: ...

class RollingFeatures(BaseFeature):
    windows: list[int]
    ops: list[Literal["mean","std","min","max"]]

FEATURE_AUDIT = """
Every engineered feature is required to honour the contract:
  build(y, up_to) must NOT touch any y[t] with t > up_to.
The test harness runs 'build' on a truncated series and asserts
equality with 'build' on the full series.
"""
```

### 3.5 CLI surface

```
quanta bench --data data.csv --family classical,ml,deep --horizon 14 --cv rolling --folds 5
quanta predict --model champions.json --horizon 14 --out predictions.parquet
quanta zero-shot --model chronos-small --horizon 14
quanta info --model chronos-small         # license, weights URL, memory footprint
quanta demo --port 8501                   # launches Streamlit
```

### 3.6 Foundation-model path

Wrap each upstream library in a thin adapter:

```python
# src/quanta/families/foundation/chronos.py
@register_forecaster("chronos-small")
class ChronosSmall:
    name, family = "chronos-small", "foundation"

    def fit(self, y, exog=None):
        # chronos is zero-shot: fit() just caches the series
        self._y = y

    def predict(self, horizon, exog_future=None, quantile_levels=(0.1,0.5,0.9)):
        from chronos import ChronosPipeline  # lazy
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
        samples = pipe.predict(context=torch.tensor(self._y.values), prediction_length=horizon)
        q = np.quantile(samples, quantile_levels, axis=0).T
        mean = samples.mean(axis=0)
        return ForecastOutput(mean=mean, quantiles=q, quantile_levels=quantile_levels,
                              horizon=horizon, model_name=self.name, sample_paths=samples)
```

Same pattern for TimesFM (Google) and Moirai (Salesforce).

### 3.7 Streamlit demo contract

- Upload CSV with columns `[timestamp, value]`.
- Pick model from a dropdown (filtered by what's installed).
- Pick horizon + confidence band.
- Render interactive plotly chart; "Download forecast CSV" button.
- Disclaimer banner shown for any CSV with a column matching `/^(close|open|adj|price|ticker)$/i`.

### 3.8 Packaging / extras

```toml
[project.optional-dependencies]
classical   = ["statsforecast>=1.7"]
ml          = ["lightgbm>=4.5"]
deep        = ["neuralforecast>=1.7", "torch>=2.5"]
foundation  = ["chronos-forecasting>=1.2", "timesfm>=1.1", "uni2ts>=1.0"]
demo        = ["streamlit>=1.35", "plotly>=5.22"]
```

## 4. Alternatives considered

| Alt | Why not |
|-----|---------|
| Use `darts` directly | Dense, slow to extend, probabilistic outputs uneven |
| Use `sktime` directly | Great classical breadth; foundation-model story weak |
| Fork `statsforecast` | Locks into Nixtla's API shape; we want cross-vendor |
| Ray-only parallelism | Overkill for P1; joblib suffices |

## 5. Tradeoffs

- **Plugin-style family loading** adds complexity but lets classical-only users
  skip a 1 GB torch install.
- **Quantiles as the universal output** forces probabilistic thinking but
  means point-forecast-only models need a wrapper (use `q=[0.5]`).
- **Rolling-origin CV only** (no k-fold random) is slower but correct.

## 6. Rollout plan

1. Ship protocol + types + 2 classical forecasters + conformance tests (P1 wk1).
2. Add ML family + feature-audit harness + leak detector (P1 wk2).
3. Add deep family (N-BEATS, PatchTST) + GPU path (P2 wk3–4).
4. Add foundation family + `quanta info` + `zero-shot` CLI (P2 wk5).
5. Streamlit demo + HF Spaces deploy (P2 wk6).

## 7. Observability

- Run log — every `fit`/`predict` writes JSON to `.quanta/runs/{run_id}/` with
  data hash, seed, versions.
- `quanta bench` emits MLflow-compatible artefacts (optional).
- Telemetry is opt-in and off by default.

## 8. Open questions

- Do we offer a native anomaly-detection mode (threshold P90) or ship it as a recipe?
- Unified "panel" forecasting API (global models across many series) in v1 or v2?
- License-surfacing for foundation weights — CLI or Streamlit modal?
