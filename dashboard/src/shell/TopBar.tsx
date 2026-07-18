import { useTraceStore } from "../store/traceStore";

const BADGE_COLOR: Record<string, string> = {
  connecting: "#9aa3b6",
  live: "#4ade80",
  reconnecting: "#ffd24a",
  ended: "#9aa3b6",
  error: "#ff6b6b",
};

export function TopBar() {
  const header = useTraceStore((s) => s.header);
  const conn = useTraceStore((s) => s.connectionState);
  if (!header) return null;
  const { brain } = header;
  return (
    <header
      className="topbar"
      style={{
        display: "flex",
        gap: 16,
        alignItems: "center",
        height: 56,
        padding: "0 16px",
        font: "12px monospace",
        color: "var(--text)",
        background: "var(--bg2)",
        borderBottom: "1px solid var(--edge)",
      }}
    >
      <strong style={{ font: "700 14px sans-serif" }}>NEURO·SCOPE</strong>
      <span>{brain.id}</span>
      <span>· {brain.config_hash}</span>
      <span>· seed {brain.seed}</span>
      <span>· T {brain.T}</span>
      {conn !== "idle" && (
        <span
          data-testid="live-badge"
          style={{ marginLeft: "auto", color: BADGE_COLOR[conn], font: "700 12px monospace", textTransform: "uppercase" }}
        >
          ● {conn}
        </span>
      )}
    </header>
  );
}
