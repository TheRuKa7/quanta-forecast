import axios from "axios";
import { z } from "zod";
import {
  ForecastOutput,
  ForecastOutputSchema,
  Run,
  RunSchema,
  Series,
  SeriesSchema,
} from "./types";

// Always goes through Next rewrite → backend.
const client = axios.create({ baseURL: "/api", timeout: 30_000 });

export async function listSeries(): Promise<Series[]> {
  const r = await client.get("/series");
  return z.array(SeriesSchema).parse(r.data);
}

export async function getForecast(params: {
  series_id: string;
  forecaster: string;
  horizon: number;
}): Promise<ForecastOutput> {
  const r = await client.post("/forecast", params);
  return ForecastOutputSchema.parse(r.data);
}

export async function listRuns(seriesId?: string): Promise<Run[]> {
  const r = await client.get("/runs", {
    params: seriesId ? { series_id: seriesId } : undefined,
  });
  return z.array(RunSchema).parse(r.data);
}

export const FORECASTERS = ["naive", "ets", "arima", "prophet", "tft", "nbeats"] as const;
export type ForecasterName = (typeof FORECASTERS)[number];
