import { useTraceStore } from "../store/traceStore";

export function RegionActivity() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;

  return (
    <section style={{ padding: 13, font: "12px sans-serif", color: "#e9edf6" }}>
      <h3 style={{ font: "600 8.5px monospace", letterSpacing: ".12em", color: "#5b6378", textTransform: "uppercase" }}>
        Panel 01 · Region Activity
      </h3>
      {header.regions.map((r) => {
        const rate = frame?.regions[r.id]?.rate ?? 0;
        return (
          <div key={r.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 0" }}>
            <span style={{ flex: 1 }}>{r.label}</span>
            <div style={{ width: 80, height: 6, background: "rgba(255,255,255,.06)", borderRadius: 3 }}>
              <div style={{ width: `${Math.min(100, rate * 100)}%`, height: "100%", background: "#3fd2ff", borderRadius: 3 }} />
            </div>
            <span style={{ font: "11px monospace", width: 34, textAlign: "right" }}>{rate.toFixed(2)}</span>
          </div>
        );
      })}
    </section>
  );
}
