import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { TopBar } from "./TopBar";

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 7, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [],
  pathways: [],
} as TraceHeader;

describe("TopBar", () => {
  it("renders the run topology from the header", () => {
    useTraceStore.getState().load(header, []);
    render(<TopBar />);
    expect(screen.getByText(/five-region/i)).toBeInTheDocument();
    expect(screen.getByText(/T\s*32/i)).toBeInTheDocument();
    expect(screen.getByText(/seed\s*7/i)).toBeInTheDocument();
  });
});
