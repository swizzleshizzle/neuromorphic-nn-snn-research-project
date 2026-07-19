# Session Handoff — 2026-07-17 (Fri, laptop `SwizzlesDuo` → Desktop)

> Written on the **laptop**; pick up on the **Desktop**. This is the single-page pickup point.
> Everything below is committed/pushed — nothing is stranded on the laptop.

## 0. Start here (2 minutes)

```bash
git fetch --all --prune
git checkout week15-neuroscope-live      # this branch (PR #7)
.venv\Scripts\python.exe -m pip install -e ".[server]"   # REQUIRED — new optional deps
```

Then jump to §4 to see the live dashboard actually stream.

## 1. Git state

| Branch | Contains | Pushed | PR | Merged? |
|---|---|---|---|---|
| `main` | Phase 2 complete, tag `phase-2-complete` (2026-07-14) | — | — | untouched this session |
| `week15-arch-spec-v3` | architecture-spec **v3** (docs only, 1 commit) | ✅ | **none yet** | no |
| `week15-neuroscope-live` | NEURO·SCOPE Phase 2 Live (13 commits) | ✅ | **#7 OPEN** | no |

Both feature branches are **independent** (each forked from `main` at `9a1b439`) — they don't depend on
each other and can merge in either order.

PR #7: https://github.com/swizzleshizzle/neuromorphic-nn-snn-research-project/pull/7

## 2. What happened this session

**(a) Laptop was 131 commits behind.** Its `main` was stale and it sat on the merged-and-deleted
`week14-encoder-characterization` branch, which made *already-completed Desktop work look like missed
work*. Fixed (fast-forwarded, pruned, deleted the leftover `run_028.cmd`). **No Wednesday work was
actually missed** — the Desktop closed Phase 2 early on Tue Jul 14 (EXP-028 → ADR Amdt 6 → merge
`c292cda` → tag `phase-2-complete`).

**(b) Architecture spec v3** (`week15-arch-spec-v3`) — the last outstanding Phase-2 → Phase-3 rotation
item. `docs/architecture-spec-v3.md` folds the **as-trained** config into the region tables (frozen
extractor + linear REINFORCE head, 1-of-5 regions on the policy path, memory bypassed, R-STDP deferred)
and **retargets every region at the 2×2 cube**. v2 superseded; the item is ticked off in
`docs/phase2-to-phase3-transition.md`.

**(c) NEURO·SCOPE Phase 2 — Live monitoring** (`week15-neuroscope-live`, PR #7). Reviewed the dashboard
against its 5-phase roadmap: Phases 0/1 were built, but **Phase 2 (Live) never existed** — not even
stubs, just the abstract `TraceSink`/`TraceSource` seams. Built it via brainstorm → spec → plan →
subagent-driven execution (8 TDD tasks, per-task review, final whole-branch review).

## 3. What Live actually does (architecture)

**"The trace file is the queue."** The sim is never coupled to a viewer.

```
sim ──FileSink (flushes per line)──▶ active.jsonl ──tail poll──▶ FastAPI ──WS──▶ WebSocketTraceSource ──▶ store ──▶ existing UI
                                     active.jsonl.done ────────────────────────▶ {"type":"end"}
```

- `FileSink` now **flushes each line** (bytes identical; only timing changed) → the active trace is tailable.
- `src/neuromorphic/server/` — FastAPI app (`create_app`) + CLI (`python -m neuromorphic.server`). Sends
  typed envelopes `{"type":"header"|"frame"|"end"|"error","data"?}`. On connect it replays the whole file
  so far (catch-up), then tails.
- **End-of-run = a `<trace>.done` sidecar file**, never a control line — the JSONL stays pure.
- `dashboard/src/source/WebSocketTraceSource.ts` — implements the *existing* `TraceSource` interface, with
  auto-reconnect (exponential backoff). **The render path never learns which source it has** — that was the
  whole point of the Phase-0 seam.
- Store gained `connectionState` + follow-live `appendFrame`; `TopBar` shows a LIVE badge;
  `App` picks WS vs File by `VITE_WS_URL`. **Without `VITE_WS_URL` the old file-replay path is
  behavior-identical.**
- No `WebSocketSink` — deliberately. File-as-queue was chosen; a sim-side sink stays a Phase-4 fan-out concern.

## 4. Run / verify it

```powershell
# 1. simulate a running sim (or run a real experiment — FileSink now flushes)
.venv\Scripts\python.exe scripts\replay_into_file.py dashboard\public\week11_dashboard_trace.jsonl outputs\live.jsonl --delay 0.2

# 2. serve it  (needs: pip install -e ".[server]")
.venv\Scripts\python.exe -m neuromorphic.server --trace outputs\live.jsonl

# 3. dashboard pointed at the socket
cd dashboard
$env:VITE_WS_URL="ws://localhost:8000/stream"; npm run dev
```

Expect: frames stream in, playhead follows the tail, LIVE badge solid → flips to **ended** when `.done`
lands. Kill/restart the server to watch **reconnecting → live**.

Tests:
```powershell
.venv\Scripts\python.exe -m pytest tests\monitor tests\server tests\scripts --basetemp=.pytest_tmp -q   # 25 passed
cd dashboard; npx vitest run; npx tsc -b                                                                 # 51 passed, tsc clean
```

## 5. Environment gotchas (bit us this session)

- **`pip install -e ".[server]"` is required** for the server + its tests (new optional-dependency group).
- **matplotlib is NOT installed on the laptop**, so `tests/dashboard` + `tests/viz` error at *collection*
  there. It is not a declared dependency and this branch never touches it. If the Desktop has it, the full
  suite runs clean; if not, `pip install matplotlib`.
- **Windows pytest quirk:** a locked `pytest-current` symlink can make pytest exit non-zero in
  `pytest_sessionfinish` *even when every test PASSED*. Use `--basetemp=.pytest_tmp` for a clean exit.
  `.pytest_tmp/` is now gitignored (scratch files had been accidentally committed; cleaned up in `935867a`).
- **tsconfig targets `lib: ["ES2020"]`** — do **not** use ES2022 APIs. `Array.prototype.at()` broke
  `tsc -b` mid-session (caught by review, fixed in `6e5d485`). vitest won't catch this; only `tsc -b` does.

## 6. Deferred follow-ups (from the final whole-branch review)

> Recorded here because the SDD ledger (`.superpowers/sdd/`) is **gitignored** and does not transfer.
> All three were judged safe to defer for a single-user local dev tool.

1. **Idle-disconnect coroutine leak** (`server/app.py`, `stream_trace` poll loop) — a client that
   disconnects while idle (no new frames *and* no `.done`) isn't noticed, because disconnect only surfaces
   as an exception on the next `ws.send_json`. One tail coroutine + file handle leak per such disconnect.
   The (currently vestigial) `alive` flag is the natural fix hook — flip it from a concurrent
   `ws.receive()` task. **Keep `alive` for this reason.**
2. **Corrupt-line handling** — a bad mid-stream `json.loads` raises uncaught (only `WebSocketDisconnect`
   is caught), so the socket drops and the client reconnect-storms instead of receiving a clean `error`
   envelope. Low likelihood (FileSink writes valid JSON).
3. **Test coverage nits** — no test for multi-step backoff growth/cap or the consumer-`close()`
   no-reconnect path in `WebSocketTraceSource` (both correct by inspection). Cosmetic: `"reconnecting"`
   state is emitted twice on a drop (idempotent for the badge).

## 7. Open decisions for you

1. **Merge order / PRs.** PR #7 is open for Live. `week15-arch-spec-v3` is pushed but has **no PR** —
   open one, fold it into #7, or merge it separately?
2. **Phase 3 kickoff (~Jul 25).** v3 §7 parks three questions: engage-first vs run-the-v1-recipe-and-
   fail-first on 1-move scrambles; whether a cube analog to the grid displacement pre-training proxy
   exists (distance-to-solved?) or unfreeze end-to-end; how far to push plasticity. Also: stand up the
   **monolithic same-neuron-count baseline** early so "does regionalization help?" is answerable from day one.
3. **Remaining dashboard gaps** (from this session's audit, not built): the **task panel is grid-specific**
   — both `_grid_world` (Python) and React `TaskState` hardcode the 5×5 grid, so a cube trace won't render
   meaningfully without a cube state view (**this is the next Phase-3-relevant dashboard work**). Also:
   pathway "intensity" is a proxy (source-region rate, not measured edge traffic); `SpikeRaster` region is
   hardcoded to `prefrontal`; `detail.membrane` has no data source.

## 8. Pointers

- Spec: `docs/superpowers/specs/2026-07-17-neuroscope-phase2-live-design.md`
- Plan: `docs/superpowers/plans/2026-07-17-neuroscope-phase2-live.md`
- Platform roadmap (Phase 2 row now marked SHIPPED): `docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md`
- Run instructions: `dashboard/README.md`
- Architecture v3: `docs/architecture-spec-v3.md` (on `week15-arch-spec-v3`)
- Vault: `Weekly Notes/week-15-phase2-closeout.md` (Fri Jul 17 session appended)
