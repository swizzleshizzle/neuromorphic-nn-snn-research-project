import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Panel } from "./Panel";

describe("Panel", () => {
  it("renders kicker, title, and children", () => {
    render(
      <Panel kicker="PANEL 0X · TEST" title="My Panel">
        <div>body content</div>
      </Panel>,
    );
    expect(screen.getByText("PANEL 0X · TEST")).toBeInTheDocument();
    expect(screen.getByText("My Panel")).toBeInTheDocument();
    expect(screen.getByText("body content")).toBeInTheDocument();
  });
});
