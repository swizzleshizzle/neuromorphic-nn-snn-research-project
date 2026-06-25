import type { Frame, Trace, TraceHeader } from "../contract";

/** Parse JSONL trace text: line 0 = header, each remaining non-blank line = one Frame. */
export function parseTrace(text: string): Trace {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) {
    throw new Error("parseTrace: empty trace (no header line)");
  }
  const header = JSON.parse(lines[0]) as TraceHeader;
  const frames = lines.slice(1).map((l) => JSON.parse(l) as Frame);
  return { header, frames };
}
