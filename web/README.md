# quanta-web

Next.js 15 dashboard for the `quanta-forecast` backend. Pick a series, pick a
forecaster, get a probabilistic forecast with quantile bands and scoring
metrics.

## Stack

- Next.js 15 (App Router, Turbopack dev) + React 19
- TanStack Query + TanStack Table
- Recharts (quantile band + mean line)
- Tailwind CSS v3 + Lucide icons
- Axios + Zod (validated API client)

## Routes

| Path       | Purpose                                       |
|------------|-----------------------------------------------|
| `/`        | Forecast runner (series × forecaster × H)     |
| `/runs`    | Sortable run history table                    |

## Run

```bash
cd web
pnpm install
cp .env.example .env.local   # point NEXT_PUBLIC_API_URL at your backend
pnpm dev                      # http://localhost:3001
```

All backend calls go through a Next rewrite at `/api/*` → `NEXT_PUBLIC_API_URL`,
so CORS is not a concern in dev.

## Backend contract

- `GET  /series`              — list registered series
- `POST /forecast`            — `{series_id, forecaster, horizon}` → `ForecastOutput`
- `GET  /runs?series_id=...`  — run history with headline metrics

See `src/lib/types.ts` for the exact Zod schemas.

## Build

```bash
pnpm build
pnpm start
```

Deployable on Vercel or any Node host; static export is not used because runs
are fetched client-side.
