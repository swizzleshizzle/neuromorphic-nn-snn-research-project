import { useTraceStore } from "../store/traceStore";
import { Panel } from "./Panel";

const REGION = "prefrontal"; // hardcoded in 1a; region selection deferred (spec §5)
const STRIP_W = 232;
const STRIP_H = 14;

export function SpikeRaster() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  const winTi = useTraceStore((s) => s.winTi);
  if (!header) return null;

  const T = header.brain.T;
  const spikes = frame?.field?.[REGION]?.spikes ?? []; // [T][N]
  const nNeurons = spikes[0]?.length ?? 0;
  const playheadX = T > 1 ? (winTi / (T - 1)) * STRIP_W : 0;

  const rows = [];
  for (let neuron = 0; neuron < nNeurons; neuron++) {
    const marks = [];
    for (let ti = 0; ti < spikes.length; ti++) {
      if (spikes[ti][neuron] === 1) {
        const x = T > 1 ? (ti / (T - 1)) * STRIP_W : 0;
        marks.push(<rect key={ti} x={x} y={2} width={3.2} height={10} rx={1} fill="var(--c-prefrontal)" />);
      }
    }
    rows.push(
      <div key={neuron} data-raster-row style={{ display: "flex", alignItems: "center", gap: 8, padding: "3px 0" }}>
        <span style={{ width: 34, font: "8px monospace", color: "var(--text-faint)" }}>n{neuron}</span>
        <svg
          viewBox={`0 0 ${STRIP_W} ${STRIP_H}`}
          width="100%"
          height={STRIP_H}
          style={{ background: "var(--panel2)", borderRadius: 4, display: "block" }}
        >
          {marks}
          <line x1={playheadX} x2={playheadX} y1={0} y2={STRIP_H} stroke="#fff" strokeWidth={1} strokeOpacity={0.6} />
        </svg>
      </div>,
    );
  }

  return (
    <Panel kicker="PANEL 05 · field" title="Spike Raster">
      {rows}
      <div style={{ display: "flex", justifyContent: "space-between", font: "500 8px/1 monospace", color: "var(--text-faint)", marginTop: 4 }}>
        <span>t₀</span>
        <span>inference window · T={T}</span>
        <span>t{T - 1}</span>
      </div>
    </Panel>
  );
}
