import { useTraceStore } from "../store/traceStore";
import { Panel } from "./Panel";

const ARROW: Record<string, string> = { up: "▲", right: "▶", down: "▼", left: "◀" };

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
          style={{
            aspectRatio: "1",
            border: "1px solid var(--edge)",
            borderRadius: 2,
            background: isAgent ? "var(--c-sensory)" : "transparent",
            boxShadow: isGoal ? "inset 0 0 0 2px var(--c-motor)" : "none",
          }}
        />,
      );
    }
  }

  const rewardColor = (v: number) => (v >= 0 ? "var(--reward-pos)" : "var(--reward-neg)");
  const returnColor = (v: number) => (v >= 0 ? "var(--reward-pos)" : "var(--return-neg)");

  return (
    <Panel kicker="PANEL 03 · TASK" title="Task State">
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 3, maxWidth: 260, margin: "0 auto" }}>
        {cells}
      </div>
      {task && (
        <div style={{ font: "11px monospace", color: "var(--text-dim)", marginTop: 10, display: "flex", flexDirection: "column", gap: 3 }}>
          <div>
            agent {task.agent[0]},{task.agent[1]} · goal {task.goal[0]},{task.goal[1]}
          </div>
          <div>
            action {ARROW[task.action_label] ?? ""} {task.action_label}
          </div>
          <div>
            reward <span data-reward style={{ color: rewardColor(task.reward) }}>{task.reward}</span>
            {" · return "}
            <span data-return style={{ color: returnColor(task.return) }}>{task.return}</span>
          </div>
        </div>
      )}
    </Panel>
  );
}
