import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { useTraceStore } from "./store/traceStore";

// The R3F/WebGL hero can't mount in jsdom (no ResizeObserver/WebGL context).
// App.test covers trace-load + shell rendering; the hero is verified by the
// Playwright smoke (Task 10), so stub it out here to isolate this test.
vi.mock("./hero/Hero", () => ({ Hero: () => null }));

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 0, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [{ id: "sensory", label: "Sensory Cortex", n_neurons: 64, role: "input", render: "dots" }],
  pathways: [],
};
const frame = { episode: 0, step: 0, t: 0, task: { agent: [0, 0], goal: [4, 4], action: 0, action_label: "up", reward: -1, return: -1, terminated: false, truncated: false }, regions: { sensory: { rate: 0.2, spikes: 1, active_frac: 0.1, rate_t: [] } }, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: { sensory: { spikes: [[0]] } } };

afterEach(() => {
  useTraceStore.getState().reset();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("loads the trace through a TraceSource into the store and renders the shell", async () => {
    const body = JSON.stringify(header) + "\n" + JSON.stringify(frame) + "\n";
    // Plain response-like object — avoids depending on a global Response in jsdom.
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, text: async () => body })));

    render(<App />);

    await waitFor(() => expect(useTraceStore.getState().header).toBeDefined());
    expect(useTraceStore.getState().frames).toHaveLength(1);
    // shell renders the brain id somewhere
    expect(await screen.findByText(/five-region/i)).toBeInTheDocument();
  });
});
