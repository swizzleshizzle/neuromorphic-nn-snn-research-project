import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { RegionActivity } from "./RegionActivity";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory Cortex", n_neurons: 64, role: "input", render: "dots" },
    { id: "motor", label: "Motor Cortex", n_neurons: 4, role: "output", render: "dots" },
  ],
  pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: {} as Frame["task"],
  regions: { sensory: { rate: 0.21, spikes: 4, active_frac: 0.3, rate_t: [] }, motor: { rate: 0.62, spikes: 9, active_frac: 0.25, rate_t: [] } },
  pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("RegionActivity", () => {
  it("renders one row per header region with its live rate", () => {
    useTraceStore.getState().load(header, [frame]);
    render(<RegionActivity />);
    expect(screen.getByText("Sensory Cortex")).toBeInTheDocument();
    expect(screen.getByText("Motor Cortex")).toBeInTheDocument();
    expect(screen.getByText("0.21")).toBeInTheDocument();
    expect(screen.getByText("0.62")).toBeInTheDocument();
  });
});
