import { useTraceStore } from "../store/traceStore";

export function TopBar() {
  const header = useTraceStore((s) => s.header);
  if (!header) return null;
  return <header className="topbar">{header.brain.id}</header>;
}
