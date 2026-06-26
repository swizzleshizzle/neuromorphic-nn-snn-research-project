import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { CommunicationFlow } from "./CommunicationFlow";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory", n_neurons: 64, role: "input", render: "dots" },
    { id: "hippocampus", label: "Hippocampus", n_neurons: 150, role: "memory", render: "dots" },
    { id: "prefrontal", label: "Prefrontal", n_neurons: 4, role: "control", render: "dots" },
    { id: "motor", label: "Motor", n_neurons: 4, role: "output", render: "dots" },
  ],
  pathways: [
    { id: "sens_hippo", src: "sensory", dst: "hippocampus", gated: true, label: "store/recall" },
    { id: "sens_pfc", src: "sensory", dst: "prefrontal", gated: false, label: "perceive" },
    { id: "hippo_pfc", src: "hippocampus", dst: "prefrontal", gated: true, label: "recall" },
    { id: "pfc_motor", src: "prefrontal", dst: "motor", gated: true, label: "act" },
  ],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: {} as Frame["task"],
  regions: {},
  pathways: {
    sens_hippo: { intensity: 0.4, gate_open: 0 },
    sens_pfc: { intensity: 0.3 },
    hippo_pfc: { intensity: 0.2, gate_open: 0 },
    pfc_motor: { intensity: 0.1, gate_open: [0, 0, 0.7, 0] },
  },
  router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("CommunicationFlow", () => {
  it("renders a node per region, an edge per pathway, and gate tags", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<CommunicationFlow />);
    expect(container.querySelectorAll("[data-node]")).toHaveLength(4);
    expect(container.querySelectorAll("[data-edge]")).toHaveLength(4);
    // sens_hippo quiescent → STORE; sens_pfc ungated → OPEN; pfc_motor max 0.7 → OPEN; hippo_pfc 0 → CLOSED
    expect(screen.getByText("STORE")).toBeInTheDocument();
    expect(screen.getAllByText("OPEN")).toHaveLength(2);
    expect(screen.getByText("CLOSED")).toBeInTheDocument();
  });
});
