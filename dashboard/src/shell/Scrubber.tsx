import { useTraceStore } from "../store/traceStore";

export function Scrubber() {
  const winTi = useTraceStore((s) => s.winTi);
  const T = useTraceStore((s) => s.T);
  const playing = useTraceStore((s) => s.playing);
  const play = useTraceStore((s) => s.play);
  const pause = useTraceStore((s) => s.pause);

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        height: 62,
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "0 16px",
        font: "11px monospace",
        color: "var(--text-dim)",
        background: "var(--bar)",
        backdropFilter: "var(--blur)",
      }}
    >
      <button onClick={playing ? pause : play} style={{ font: "11px monospace" }}>
        {playing ? "❚❚" : "▶"}
      </button>
      <span>
        t {winTi}/{T}
      </span>
    </div>
  );
}
