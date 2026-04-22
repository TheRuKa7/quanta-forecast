"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastOutput } from "@/lib/types";

/**
 * Quantile bands + mean line. Expects ForecastOutput with quantile_levels
 * like [0.1, 0.5, 0.9] or [0.05, 0.5, 0.95]. Shades the outer band.
 */
export function ForecastChart({ data }: { data: ForecastOutput }) {
  const levels = data.quantile_levels;
  const lowIdx = 0;
  const highIdx = levels.length - 1;

  const rows = data.timestamps.map((ts, i) => ({
    ts,
    mean: data.mean[i],
    lower: data.quantiles[i][lowIdx],
    upper: data.quantiles[i][highIdx],
    band: [data.quantiles[i][lowIdx], data.quantiles[i][highIdx]] as [number, number],
  }));

  const lo = Math.round(levels[lowIdx] * 100);
  const hi = Math.round(levels[highIdx] * 100);

  return (
    <div className="w-full h-[360px] rounded-xl border border-border bg-card p-4">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 28% 17%)" />
          <XAxis dataKey="ts" stroke="hsl(215 20% 65%)" fontSize={11} />
          <YAxis stroke="hsl(215 20% 65%)" fontSize={11} />
          <Tooltip
            contentStyle={{
              background: "hsl(224 71% 6%)",
              border: "1px solid hsl(215 28% 17%)",
              borderRadius: 8,
            }}
            labelStyle={{ color: "hsl(213 31% 91%)" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Area
            type="monotone"
            dataKey="band"
            name={`q${lo}–q${hi} band`}
            stroke="none"
            fill="hsl(234 89% 74%)"
            fillOpacity={0.18}
          />
          <Line
            type="monotone"
            dataKey="mean"
            name="mean"
            stroke="hsl(234 89% 74%)"
            strokeWidth={2}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
