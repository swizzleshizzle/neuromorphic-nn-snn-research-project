import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../outputs/week11_dashboard_trace.jsonl");
const dst = resolve(here, "../public/week11_dashboard_trace.jsonl");

if (!existsSync(src)) {
  console.error(`trace not found at ${src}\nrun: python experiments/022_week11_dashboard_trace/run.py`);
  process.exit(1);
}
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log(`synced trace -> ${dst}`);
