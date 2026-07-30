import type React from "react";
import { useTraceStore } from "../../store/traceStore";
import { cubeNetPosition, NET_COLS, NET_ROWS } from "../../panels/cubeNet";
import { aggregateCubeFacelets, aggregateSensoryGrid } from "../sensory";

const FACELET_COLOR = [
  "#f2f2f2", "#e04a2f", "#2f7de0", "#f2c14a", "#3fae6a", "#c94ad6",
];

const shellStyle: React.CSSProperties = {
  position: "absolute",
  top: 44,
  left: 18,
  padding: "10px 11px",
  borderRadius: 10,
  background: "var(--panel)",
  border: "1px solid var(--edge)",
  backdropFilter: "var(--blur)",
  pointerEvents: "none",
  zIndex: 10,
};

const captionStyle: React.CSSProperties = {
  font: "600 8px/1 'IBM Plex Mono', monospace",
  color: "var(--text-faint)",
  letterSpacing: ".12em",
  marginBottom: 7,
};

function CubeSensoryView({ colors, cubeN }: { colors: number[]; cubeN: number }) {
  const cells = Array.from({ length: NET_ROWS * NET_COLS }, () => -1);
  colors.forEach((color, f) => {
    const { row, col } = cubeNetPosition(f);
    cells[row * NET_COLS + col] = color;
  });
  return (
    <div data-sensory-cube style={shellStyle}>
      <div style={captionStyle}>
        SENSORY INPUT · CUBE {cubeN}x{cubeN}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${NET_COLS}, 10px)`, gap: 2 }}>
        {cells.map((color, idx) =>
          color < 0 ? (
            <div key={idx} style={{ width: 10, height: 10 }} />
          ) : (
            <div
              key={idx}
              data-sensory-facelet
              style={{ width: 10, height: 10, borderRadius: 2, background: FACELET_COLOR[color] ?? "var(--edge)" }}
            />
          ),
        )}
      </div>
    </div>
  );
}

function GridSensoryView({ g, agg }: { g: number; agg: { agentCell: number; goalCell: number } }) {
  const cells = Array.from({ length: g * g }, (_, c) => c);

  return (
    <div
      data-sensory-grid
      style={{
        position: "absolute",
        top: 44,
        left: 18,
        padding: "10px 11px",
        borderRadius: 10,
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        backdropFilter: "var(--blur)",
        pointerEvents: "none",
        zIndex: 10,
      }}
    >
      <div style={{ font: "600 8px/1 'IBM Plex Mono', monospace", color: "var(--text-faint)", letterSpacing: ".12em", marginBottom: 7 }}>
        SENSORY INPUT · {g}×{g}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${g}, 10px)`, gap: 2 }}>
        {cells.map((c) => {
          const isA = c === agg.agentCell;
          const isG = c === agg.goalCell;
          return (
            <div
              key={c}
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: isA ? "var(--c-sensory)" : "var(--edge)",
                boxShadow: isA ? "0 0 8px var(--c-sensory)" : isG ? "inset 0 0 0 1.5px var(--c-motor)" : "none",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

export function SensoryGrid() {
  const header = useTraceStore((s) => s.header);
  const envStep = useTraceStore((s) => s.envStep);
  const frames = useTraceStore((s) => s.frames);
  if (!header) return null;
  const frame = frames[envStep];

  const cube = aggregateCubeFacelets(frame?.encoding);
  if (cube) {
    const si = frame!.encoding!.sensory_input;
    return <CubeSensoryView colors={cube} cubeN={"cube_n" in si ? si.cube_n : 2} />;
  }

  const agg = aggregateSensoryGrid(frame?.encoding);
  if (!agg) return null;
  const si = frame!.encoding!.sensory_input;
  const g = "grid_n" in si ? si.grid_n : 0;
  return <GridSensoryView g={g} agg={agg} />;
}
