import { useEffect } from "react";
import { FileTraceSource } from "./source/FileTraceSource";
import { Shell } from "./shell/Shell";
import { useTraceStore } from "./store/traceStore";

const TRACE_URL = import.meta.env.VITE_TRACE_URL ?? "/week11_dashboard_trace.jsonl";

export function App() {
  const header = useTraceStore((s) => s.header);
  const load = useTraceStore((s) => s.load);

  useEffect(() => {
    const source = new FileTraceSource(TRACE_URL);
    let cancelled = false;
    source
      .open()
      .then((hdr) => {
        const frames: Parameters<typeof load>[1] = [];
        source.subscribe((f) => frames.push(f));
        if (!cancelled) load(hdr, frames);
      })
      .catch((err) => console.error("trace load failed:", err));
    return () => {
      cancelled = true;
      source.close();
    };
  }, [load]);

  if (!header) {
    return <div style={{ color: "#9aa3b6", font: "13px monospace", padding: 24 }}>Loading trace…</div>;
  }
  return <Shell />;
}
