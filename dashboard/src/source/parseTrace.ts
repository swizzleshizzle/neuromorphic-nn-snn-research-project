import type { Frame, Trace, TraceHeader } from "../contract";

/** Parse JSONL trace text: line 0 = header, each remaining non-blank line = one Frame.
 *
 * The wire format carries the task type once, in the header. Frames are stamped
 * with it here so the in-memory model is a real discriminated union and panels
 * cannot read the wrong variant. Traces written before the type existed still
 * load, because the stamp comes from their own header.
 */
export function parseTrace(text: string): Trace {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) {
    throw new Error("parseTrace: empty trace (no header line)");
  }
  const header = JSON.parse(lines[0]) as TraceHeader;
  const type = header.task?.type ?? "gridworld";

  const frames = lines.slice(1).map((l, i) => {
    const frame = JSON.parse(l) as Frame;
    const task = frame.task as unknown as Record<string, unknown>;
    if (task) {
      if (type === "cube" && "agent" in task) {
        throw new Error(
          `parseTrace: header declares task type "cube" but frame ${i} carries gridworld "agent"`,
        );
      }
      if (type === "gridworld" && "facelets" in task) {
        throw new Error(
          `parseTrace: header declares task type "gridworld" but frame ${i} carries cube "facelets"`,
        );
      }
      frame.task = { ...task, type } as Frame["task"];
    }
    return frame;
  });
  return { header, frames };
}
