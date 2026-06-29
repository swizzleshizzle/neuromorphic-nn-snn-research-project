import { useTraceStore } from "../store/traceStore";
import { regionHue } from "../theme/regionHue";
import { Panel } from "./Panel";

/** Build SVG polyline points for a sparkline scaled to a w×h box. */
function sparkPoints(series: number[], w: number, h: number): string {
  if (series.length === 0) return "";
  const max = Math.max(...series, 1e-6);
  const n = series.length;
  return series
    .map((v, i) => {
      const x = n > 1 ? (i / (n - 1)) * w : 0;
      const y = h - (v / max) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function RegionActivity() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;
  const policy = header.policy_regions;

  return (
    <Panel kicker="PANEL 01 · ACTIVITY" title="Region Activity">
      {header.regions.map((r) => {
        const rs = frame?.regions[r.id];
        const rate = rs?.rate ?? 0;
        const hue = regionHue(r.id);
        const spectator = policy !== undefined && !policy.includes(r.id);
        return (
          <div key={r.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 0" }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: hue, boxShadow: `0 0 6px ${hue}`, flex: "none" }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", font: "12px sans-serif" }}>
                <span>
                  {r.label}
                  {spectator && (
                    <span
                      data-spectator
                      style={{ marginLeft: 6, font: "8px monospace", color: "var(--text-faint)", border: "1px solid var(--edge)", borderRadius: 3, padding: "1px 4px" }}
                    >
                      spectator
                    </span>
                  )}
                </span>
                <span style={{ font: "11px monospace", color: "var(--text-dim)" }}>{rate.toFixed(2)}</span>
              </div>
              <svg width="100%" height="16" viewBox="0 0 100 16" preserveAspectRatio="none" style={{ marginTop: 3, display: "block" }}>
                <polyline points={sparkPoints(rs?.rate_t ?? [], 100, 16)} fill="none" stroke={hue} strokeWidth="1.2" />
              </svg>
              <div style={{ font: "9px monospace", color: "var(--text-faint)", marginTop: 2 }}>
                active {((rs?.active_frac ?? 0) * 100).toFixed(0)}% · {rs?.spikes ?? 0} spikes
              </div>
            </div>
          </div>
        );
      })}
    </Panel>
  );
}
