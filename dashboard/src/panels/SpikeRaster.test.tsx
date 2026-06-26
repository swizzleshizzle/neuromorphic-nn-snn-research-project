import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { SpikeRaster } from "./SpikeRaster";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 4 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [], pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: {} as Frame["task"],
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] },
  field: { prefrontal: { spikes: [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]] } },
} as unknown as Frame;

describe("SpikeRaster", () => {
  it("renders one strip per prefrontal neuron, spike marks, a playhead, and the T label", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<SpikeRaster />);
    expect(container.querySelectorAll("[data-raster-row]")).toHaveLength(4);
    // two spikes total in the fixture
    expect(container.querySelectorAll("rect")).toHaveLength(2);
    // a playhead line per row
    expect(container.querySelectorAll("line")).toHaveLength(4);
    expect(screen.getByText("inference window · T=4")).toBeInTheDocument();
  });
});
