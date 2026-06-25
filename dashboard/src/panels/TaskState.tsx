import { useTraceStore } from "../store/traceStore";

export function TaskState() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;
  const n = header.task.grid_n;
  const task = frame?.task;

  const cells = [];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const isAgent = task && task.agent[0] === x && task.agent[1] === y;
      const isGoal = task && task.goal[0] === x && task.goal[1] === y;
      cells.push(
        <div
          key={`${x},${y}`}
          data-cell
          style={{ aspectRatio: "1", border: "1px solid rgba(255,255,255,.06)", background: isAgent ? "#3fd2ff" : "transparent", boxShadow: isGoal ? "inset 0 0 0 2px #46f0a0" : "none" }}
        />,
      );
    }
  }

  return (
    <section style={{ padding: 13, font: "12px sans-serif", color: "#e9edf6" }}>
      <h3 style={{ font: "600 8.5px monospace", letterSpacing: ".12em", color: "#5b6378", textTransform: "uppercase" }}>
        Panel 03 · Task State
      </h3>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 2, maxWidth: 200 }}>{cells}</div>
      {task && (
        <div style={{ font: "11px monospace", color: "#9aa3b6", marginTop: 8 }}>
          <div>agent {task.agent[0]},{task.agent[1]} · goal {task.goal[0]},{task.goal[1]}</div>
          <div>action {task.action_label} · reward {task.reward} · return {task.return}</div>
        </div>
      )}
    </section>
  );
}
