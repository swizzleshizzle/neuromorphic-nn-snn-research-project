import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { TopBar } from "./TopBar";

const header = { schema_version: "1.0", brain: { id: "b", config_hash: "h", seed: 0, T: 1 } } as unknown as TraceHeader;

describe("TopBar live badge", () => {
  beforeEach(() => {
    useTraceStore.getState().reset();
    useTraceStore.getState().load(header, []);
  });

  it("hides the badge when idle", () => {
    expect(screen.queryByTestId("live-badge")).toBeNull();
    render(<TopBar />);
    expect(screen.queryByTestId("live-badge")).toBeNull();
  });

  it("shows connection state when live", () => {
    useTraceStore.getState().setConnectionState("live");
    render(<TopBar />);
    expect(screen.getByTestId("live-badge").textContent?.toLowerCase()).toContain("live");
  });
});
