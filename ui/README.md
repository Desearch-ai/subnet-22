# SN22 UI

A public, read-only dashboard over the task API's log endpoints. It shows what Subnet 22 is doing
right now and what it did recently: tasks waiting and claimed, uploads being checked, every
validator's vote, the final result of each task and the rows that were paid.

## Requirements

- Node.js 22 or newer and npm
- A task API to read from

## Run locally

Start a task API first. The sandbox serves one at `http://127.0.0.1:18080`
(see [Test Your Miner Locally](../docs/test-your-miner.md)). Then:

```bash
cd ui
npm install
npm run dev
```

Open <http://localhost:5173>. The dev server has to stay on port 5173: the task API allows
requests from `http://localhost:5173` and `http://127.0.0.1:5173` by default.

## Configuration

| Variable            | Default                  | Meaning                           |
| ------------------- | ------------------------ | --------------------------------- |
| `VITE_TASK_API_URL` | `http://127.0.0.1:18080` | Base URL of the task API to read. |

Copy `.env.example` to `.env.local` to change it. The value is read at build time.

## Scripts

| Command             | What it does                                   |
| ------------------- | ---------------------------------------------- |
| `npm run dev`       | Dev server on port 5173                        |
| `npm run build`     | Type-check and build the static site to `dist` |
| `npm run preview`   | Serve the built site locally                   |
| `npm run typecheck` | `tsc --noEmit`                                 |
| `npm run lint`      | ESLint, warnings fail                          |
| `npm run format`    | Prettier                                       |
| `npm test`          | Unit tests for the helpers in `src/lib`        |

## Pages

| Path                  | Shows                                                                  |
| --------------------- | ---------------------------------------------------------------------- |
| `/`                   | Tasks waiting and in progress, recent tasks, miners by share           |
| `/tasks`              | Finalized tasks, filtered by miner, validator or result                |
| `/tasks/:taskId`      | One task: result, crawl time, every validator's vote, the checked URLs |
| `/miners`             | Every miner: share, budget, tasks crawling and waiting, results, rows  |
| `/miners/:hotkey`     | One miner: stats, rows over time, budget history, its tasks            |
| `/validators`         | Every validator: activity, votes, agreement with the final result      |
| `/validators/:hotkey` | One validator: stats and its votes                                     |

Live views refresh every minute while the tab is visible. When the API answers `429` or `503`
the page waits for the `Retry-After` time before it asks again.

## Layout

```
src/
  api/            typed client, response types, request pause on 429/503
    queries/      one file of TanStack Query hooks per resource
  components/
    ui/           primitives: card, table, badge, stat, states
    layout/       app shell, notices
    charts/       the rows-over-time chart
    overview/     panels of the overview page
    tasks/        task table, votes comparison, per-URL table
    miners/       miner table, stats, budget history
    validators/   validator table, stats, votes
  hooks/          shared hooks
  lib/            formatters, labels, paths, sorting
  pages/          one component per route
  index.css       design tokens and base styles
```

Colours, fonts and radius are defined once in `src/index.css`. Components use those tokens only.

## Deploy

`npm run build` produces a static site in `dist`. Any static host works; every path has to fall
back to `index.html` (`vercel.json` does this on Vercel). Set `VITE_TASK_API_URL` to the public task
API at build time.

The task API only answers browsers from origins it knows. Its operator has to add the site's
origin to `TASK_API_CORS_ORIGINS`.
