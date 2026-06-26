import { useTraceStore } from "../../store/traceStore";
import { aggregateSensoryGrid } from "../sensory";

export function SensoryGrid() {
  const header = useTraceStore((s) => s.header);
  const envStep = useTraceStore((s) => s.envStep);
  const frames = useTraceStore((s) => s.frames);
  if (!header) return null;
  const frame = frames[envStep];
  const agg = aggregateSensoryGrid(frame?.encoding);
  if (!agg) return null;
  const g = frame!.encoding!.sensory_input.grid_n;
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
