import { useTraceStore } from "../store/traceStore";

export function TopBar() {
  const header = useTraceStore((s) => s.header);
  if (!header) return null;
  const { brain } = header;
  return (
    <header
      className="topbar"
      style={{ display: "flex", gap: 16, alignItems: "center", height: 56, padding: "0 16px", font: "12px monospace", color: "#e9edf6", background: "#080a12", borderBottom: "1px solid rgba(255,255,255,.075)" }}
    >
      <strong style={{ font: "700 14px sans-serif" }}>NEURO·SCOPE</strong>
      <span>{brain.id}</span>
      <span>· {brain.config_hash}</span>
      <span>· seed {brain.seed}</span>
      <span>· T {brain.T}</span>
    </header>
  );
}
