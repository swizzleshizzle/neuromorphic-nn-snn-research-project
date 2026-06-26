import { useTraceStore } from "../store/traceStore";
import { Panel } from "./Panel";

export function ThalamicRouter() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;

  const labels = header.task.action_labels;
  const utilities = frame?.router?.utilities ?? [];
  const gates = frame?.router?.gate_open ?? [];
  const maxUtil = Math.max(...utilities, 1e-6);
  const selected = frame?.task?.action;

  return (
    <Panel kicker="PANEL 04 · GATING" title="Thalamic Router" accent="var(--c-router)">
      <div
        style={{ display: "flex", gap: 9, font: "500 8px/1 monospace", color: "var(--text-faint)", letterSpacing: ".08em", padding: "0 8px 6px" }}
      >
        <span style={{ width: 40 }}>ACTION</span>
        <span style={{ flex: 1 }}>UTILITY</span>
        <span style={{ width: 48, textAlign: "center" }}>GATE</span>
      </div>
      {labels.map((label, a) => {
        const util = utilities[a] ?? 0;
        const open = (gates[a] ?? 0) > 0.5;
        const isSel = selected === a;
        return (
          <div
            key={label}
            data-action-row
            style={{
              display: "flex",
              alignItems: "center",
              gap: 9,
              padding: "6px 8px",
              borderRadius: 7,
              background: isSel ? "var(--panel2)" : "transparent",
              border: `1px solid ${isSel ? "var(--edge)" : "transparent"}`,
            }}
          >
            <span style={{ width: 40, font: "11px monospace", textTransform: "uppercase", color: isSel ? "var(--c-motor)" : "var(--text)" }}>
              {label}
            </span>
            <div style={{ flex: 1, height: 6, background: "var(--edge)", borderRadius: 3 }}>
              <div style={{ width: `${(util / maxUtil) * 100}%`, height: "100%", background: "var(--c-prefrontal)", borderRadius: 3 }} />
            </div>
            <span style={{ width: 26, textAlign: "right", font: "9px monospace", color: "var(--text-dim)" }}>{util.toFixed(2)}</span>
            <span
              data-gate
              style={{
                width: 48,
                textAlign: "center",
                font: "600 7.5px/1 monospace",
                letterSpacing: ".08em",
                padding: "3px 6px",
                borderRadius: 4,
                color: "#0a0a0c",
                background: open ? "var(--gate-open)" : "var(--gate-closed)",
              }}
            >
              {open ? "OPEN" : "CLOSED"}
            </span>
          </div>
        );
      })}
      <div style={{ font: "11px monospace", color: "var(--c-motor)", marginTop: 6 }}>
        selected action ▸ {selected != null ? labels[selected] : "—"}
      </div>
    </Panel>
  );
}
