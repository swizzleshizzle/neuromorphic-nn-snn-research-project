import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { TaskState } from "./TaskState";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [], pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: { agent: [2, 3], goal: [4, 4], action: 1, action_label: "right", reward: -1, return: -5, terminated: false, truncated: false },
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("TaskState", () => {
  it("renders the grid, action arrow, coords, and sign-colored reward/return", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<TaskState />);
    expect(container.querySelectorAll("[data-cell]")).toHaveLength(25);
    expect(screen.getByText(/▶ right/i)).toBeInTheDocument();
    expect(screen.getByText(/2,\s*3/)).toBeInTheDocument();
    const reward = container.querySelector("[data-reward]") as HTMLElement;
    const ret = container.querySelector("[data-return]") as HTMLElement;
    expect(reward.textContent).toBe("-1");
    expect(reward.style.color).toBe("var(--reward-neg)");
    expect(ret.textContent).toBe("-5");
    expect(ret.style.color).toBe("var(--return-neg)");
  });
});

const cubeHeader = {
  schema_version: "1.1",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "cube", cube_n: 2, action_labels: ["U", "U'", "R", "R'", "F", "F'"] },
  regions: [], pathways: [],
} as unknown as TraceHeader;

const cubeFrame = (distance: number | null) => ({
  episode: 0, step: 0, t: 0,
  task: {
    facelets: Array.from({ length: 24 }, (_, i) => i % 6),
    solved: false, distance, scramble_depth: 2,
    move: 3, move_label: "R'",
    action: 3, action_label: "R'",
    reward: -1, return: -3, terminated: false, truncated: false,
  },
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
}) as unknown as Frame;

describe("TaskState cube", () => {
  it("renders 24 facelets in a net for a cube trace", () => {
    useTraceStore.getState().load(cubeHeader, [cubeFrame(2)]);
    const { container } = render(<TaskState />);
    expect(container.querySelectorAll("[data-facelet]")).toHaveLength(24);
    expect(container.querySelector("[data-cube-net]")).toBeTruthy();
    expect(container.querySelector("[data-cell]")).toBeNull();
    expect(container.textContent).toContain("R'");
    expect(container.textContent).toContain("distance 2");
  });

  it("shows a dash rather than the string null when distance is absent", () => {
    useTraceStore.getState().load(cubeHeader, [cubeFrame(null)]);
    const { container } = render(<TaskState />);
    expect(container.textContent).toContain("distance -");
    expect(container.textContent).not.toContain("null");
  });

  it("still renders the gridworld grid when the header says gridworld", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<TaskState />);
    expect(container.querySelectorAll("[data-cell]")).toHaveLength(25);
    expect(container.querySelector("[data-cube-net]")).toBeNull();
  });
});

const unknownHeader = {
  schema_version: "1.1",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "hanoi", action_labels: ["move"] },
  regions: [], pathways: [],
} as unknown as TraceHeader;

const unknownFrame = {
  episode: 0, step: 0, t: 0,
  task: { action: 0, action_label: "move", reward: 4, return: 9, terminated: false, truncated: false },
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("TaskState unknown task type", () => {
  it("degrades to a status-only readout instead of crashing on an unrecognized task.type", () => {
    useTraceStore.getState().load(unknownHeader, [unknownFrame]);
    expect(() => render(<TaskState />)).not.toThrow();
    const { container } = render(<TaskState />);
    expect(container.querySelector("[data-cell]")).toBeNull();
    expect(container.querySelector("[data-cube-net]")).toBeNull();
    const reward = container.querySelector("[data-reward]") as HTMLElement;
    expect(reward.textContent).toBe("4");
  });
});
