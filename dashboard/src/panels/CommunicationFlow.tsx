import type { Pathway, PathwayState } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { regionHue } from "../theme/regionHue";
import { Panel } from "./Panel";

const W = 300;
const H = 120;

/** Aggregate a gate_open scalar/array into a single fraction. */
function aggGate(go: number | number[] | undefined): number {
  if (Array.isArray(go)) return go.length ? Math.max(...go) : 0;
  return typeof go === "number" ? go : 0;
}

function gateTag(p: Pathway, ps: PathwayState | undefined): { text: string; bg: string; fg: string } {
  const agg = aggGate(ps?.gate_open);
  if (p.id === "sens_hippo" && agg === 0) return { text: "STORE", bg: "var(--edge)", fg: "var(--text-faint)" };
  if (!p.gated) return { text: "OPEN", bg: "var(--gate-open)", fg: "#0a0a0c" };
  return agg > 0.5
    ? { text: "OPEN", bg: "var(--gate-open)", fg: "#0a0a0c" }
    : { text: "CLOSED", bg: "var(--gate-closed)", fg: "#0a0a0c" };
}

export function CommunicationFlow() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;

  const n = header.regions.length;
  const pos: Record<string, { x: number; y: number }> = {};
  header.regions.forEach((r, i) => {
    pos[r.id] = {
      x: n > 1 ? 24 + (i / (n - 1)) * (W - 48) : W / 2,
      y: H / 2 + (i % 2 === 0 ? -16 : 16),
    };
  });

  return (
    <Panel kicker="PANEL 02 · PATHWAY INTENSITY" title="Communication Flow">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
        {header.pathways.map((p) => {
          const a = pos[p.src];
          const b = pos[p.dst];
          if (!a || !b) return null;
          const ps = frame?.pathways[p.id];
          const intensity = ps?.intensity ?? 0;
          const closed = p.gated && aggGate(ps?.gate_open) <= 0.5;
          const mx = (a.x + b.x) / 2;
          return (
            <path
              key={p.id}
              data-edge
              d={`M ${a.x} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`}
              fill="none"
              stroke={regionHue(p.src)}
              strokeWidth={1 + intensity * 4}
              strokeOpacity={0.25 + intensity * 0.7}
              strokeDasharray={closed ? "3 4" : undefined}
            />
          );
        })}
        {header.regions.map((r) => (
          <circle key={r.id} data-node cx={pos[r.id].x} cy={pos[r.id].y} r={7} fill={regionHue(r.id)} fillOpacity={0.9} />
        ))}
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 8 }}>
        {header.pathways.map((p) => {
          const ps = frame?.pathways[p.id];
          const tag = gateTag(p, ps);
          return (
            <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 8, font: "10px monospace" }}>
              <span style={{ flex: 1, color: "var(--text-dim)" }}>{p.label ?? p.id}</span>
              <span
                data-gate-tag
                style={{ font: "600 7.5px/1 monospace", letterSpacing: ".08em", padding: "3px 6px", borderRadius: 4, color: tag.fg, background: tag.bg }}
              >
                {tag.text}
              </span>
              <span style={{ width: 34, textAlign: "right", color: regionHue(p.src) }}>{(ps?.intensity ?? 0).toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
