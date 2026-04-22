import { z } from "zod";

export const ForecastOutputSchema = z.object({
  forecaster: z.string(),
  series_id: z.string(),
  horizon: z.number().int().positive(),
  timestamps: z.array(z.string()),
  mean: z.array(z.number()),
  quantile_levels: z.array(z.number()),
  quantiles: z.array(z.array(z.number())), // shape [horizon][n_quantiles]
  sample_paths: z.array(z.array(z.number())).nullable().optional(),
  metrics: z
    .object({
      wape: z.number().nullable().optional(),
      mape: z.number().nullable().optional(),
      rmse: z.number().nullable().optional(),
      pinball_0_1: z.number().nullable().optional(),
      pinball_0_5: z.number().nullable().optional(),
      pinball_0_9: z.number().nullable().optional(),
      coverage_80: z.number().nullable().optional(),
      coverage_95: z.number().nullable().optional(),
    })
    .nullable()
    .optional(),
});
export type ForecastOutput = z.infer<typeof ForecastOutputSchema>;

export const SeriesSchema = z.object({
  id: z.string(),
  name: z.string(),
  freq: z.string(), // "D", "H", "W", etc.
  n_observations: z.number(),
  last_value: z.number().nullable(),
  last_ts: z.string().nullable(),
});
export type Series = z.infer<typeof SeriesSchema>;

export const RunSchema = z.object({
  id: z.string(),
  series_id: z.string(),
  forecaster: z.string(),
  created_at: z.string(),
  horizon: z.number(),
  wape: z.number().nullable().optional(),
  coverage_80: z.number().nullable().optional(),
});
export type Run = z.infer<typeof RunSchema>;
