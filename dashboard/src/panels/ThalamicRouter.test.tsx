import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { ThalamicRouter } from "./ThalamicRouter";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [], pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: { agent: [0, 0], goal: [4, 4], action: 2, action_label: "down", reward: -1, return: -1, terminated: false, truncated: false },
  regions: {}, pathways: {},
  router: { gate_open: [0.0, 0.0, 0.7, 0.0], gate_open_t: [], utilities: [0.1, 0.2, 0.9, 0.3] },
  field: {},
} as unknown as Frame;

describe("ThalamicRouter", () => {
  it("renders one row per action with utility, gate pill, and selected highlight", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<ThalamicRouter />);
    expect(container.querySelectorAll("[data-action-row]")).toHaveLength(4);
    // gate pills: action 2 open (0.7 > 0.5), the other three closed
    expect(screen.getByText("OPEN")).toBeInTheDocument();
    expect(screen.getAllByText("CLOSED")).toHaveLength(3);
    // utility value for the selected action
    expect(screen.getByText("0.90")).toBeInTheDocument();
    // selected-action footer
    expect(screen.getByText(/selected action ▸ down/)).toBeInTheDocument();
  });
});
