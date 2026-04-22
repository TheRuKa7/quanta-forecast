"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { Activity, Zap } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { ForecastChart } from "@/components/forecast-chart";
import { MetricCard } from "@/components/metric-card";
import { FORECASTERS, ForecasterName, getForecast, listSeries } from "@/lib/api";
import { pct, fmt } from "@/lib/utils";

export default function Home() {
  const [seriesId, setSeriesId] = useState<string>("");
  const [forecaster, setForecaster] = useState<ForecasterName>("ets");
  const [horizon, setHorizon] = useState(24);

  const series = useQuery({ queryKey: ["series"], queryFn: listSeries });

  const run = useMutation({
    mutationFn: () => getForecast({ series_id: seriesId, forecaster, horizon }),
  });

  return (
    <main className="mx-auto max-w-6xl p-6 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold flex items-center gap-2">
            <Activity className="h-6 w-6 text-primary" />
            quanta-forecast
          </h1>
          <p className="text-sm text-muted">
            Probabilistic time-series forecasting over a pluggable backend.
          </p>
        </div>
        <Link href="/runs" className="text-sm text-primary hover:underline">
          Runs →
        </Link>
      </header>

      <section className="rounded-xl border border-border bg-card p-4 grid gap-3 md:grid-cols-4">
        <div>
          <label className="text-xs text-muted">Series</label>
          <select
            className="mt-1 w-full rounded-md bg-background border border-border px-2 py-2 text-sm"
            value={seriesId}
            onChange={(e) => setSeriesId(e.target.value)}
          >
            <option value="">
              {series.isPending ? "Loading…" : "Select a series"}
            </option>
            {series.data?.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} · {s.freq} · n={s.n_observations}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-muted">Forecaster</label>
          <select
            className="mt-1 w-full rounded-md bg-background border border-border px-2 py-2 text-sm"
            value={forecaster}
            onChange={(e) => setForecaster(e.target.value as ForecasterName)}
          >
            {FORECASTERS.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-muted">Horizon</label>
          <input
            type="number"
            min={1}
            max={500}
            className="mt-1 w-full rounded-md bg-background border border-border px-2 py-2 text-sm tabular-nums"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={() => run.mutate()}
            disabled={!seriesId || run.isPending}
            className="w-full rounded-md bg-primary text-white px-3 py-2 text-sm font-medium hover:opacity-90 disabled:opacity-50 inline-flex items-center justify-center gap-2"
          >
            <Zap className="h-4 w-4" />
            {run.isPending ? "Forecasting…" : "Run forecast"}
          </button>
        </div>
      </section>

      {run.isError ? (
        <div className="rounded-xl border border-danger/50 bg-danger/10 p-3 text-sm text-danger">
          {(run.error as Error).message}
        </div>
      ) : null}

      {run.data ? (
        <>
          <section className="grid gap-3 md:grid-cols-4">
            <MetricCard
              label="WAPE"
              value={pct(run.data.metrics?.wape ?? null)}
              hint="lower is better"
              tone={
                (run.data.metrics?.wape ?? 1) < 0.1
                  ? "success"
                  : (run.data.metrics?.wape ?? 1) < 0.25
                    ? "warning"
                    : "danger"
              }
            />
            <MetricCard
              label="Coverage 80"
              value={pct(run.data.metrics?.coverage_80 ?? null)}
              hint="target 80%"
            />
            <MetricCard
              label="Pinball q50"
              value={fmt(run.data.metrics?.pinball_0_5 ?? null, 3)}
            />
            <MetricCard
              label="Horizon"
              value={`${run.data.horizon}`}
              hint={`${run.data.forecaster} · ${run.data.series_id}`}
            />
          </section>
          <ForecastChart data={run.data} />
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-border p-12 text-center text-muted">
          Pick a series and forecaster, then hit <em>Run forecast</em>.
        </div>
      )}
    </main>
  );
}
