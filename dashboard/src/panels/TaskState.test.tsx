import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { TaskState } from "./TaskState";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [],
  pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: { agent: [2, 3], goal: [4, 4], action: 1, action_label: "right", reward: -1, return: -5, terminated: false, truncated: false },
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("TaskState", () => {
  it("renders a grid_n x grid_n grid and the action/coords readout", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<TaskState />);
    // 25 cells for grid_n=5
    expect(container.querySelectorAll("[data-cell]")).toHaveLength(25);
    expect(screen.getByText(/right/i)).toBeInTheDocument();
    expect(screen.getByText(/2,\s*3/)).toBeInTheDocument();
  });
});
