/** Copy a trace from ../outputs into public/ so Vite can serve it.
 *
 * Usage:
 *   node scripts/sync-trace.mjs                 # the week-11 gridworld trace (default, unchanged)
 *   node scripts/sync-trace.mjs cube            # the cube trace (EXP-065)
 *   node scripts/sync-trace.mjs <file.jsonl>    # any trace in ../outputs
 *
 * The default is deliberately the gridworld file: `playwright.config.ts` runs `sync:trace`
 * before the e2e smoke, and that test is pinned to the gridworld contract.
 */
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

const NAMED = {
  gridworld: { src: "week11_trained_trace.jsonl", dst: "week11_dashboard_trace.jsonl" },
  cube: { src: "cube_dashboard_trace.jsonl", dst: "cube_dashboard_trace.jsonl" },
};

const arg = process.argv[2] ?? "gridworld";
const pick = NAMED[arg] ?? { src: arg, dst: arg };

const src = resolve(here, "../../outputs", pick.src);
const dst = resolve(here, "../public", pick.dst);

if (!existsSync(src)) {
  const hint =
    arg === "cube"
      ? "run: .venv/bin/python experiments/065_cube_dashboard_trace/run.py"
      : "run: python experiments/022_week11_dashboard_trace/run.py";
  console.error(`trace not found at ${src}\n${hint}`);
  process.exit(1);
}
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log(`synced ${arg} trace -> ${dst}`);
console.log(`  serve it with: VITE_TRACE_URL=/${pick.dst} npm run build && npm run preview`);
