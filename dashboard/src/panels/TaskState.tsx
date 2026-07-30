import type { CubeTask, GridTask, TaskCore } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { cubeNetPosition, NET_COLS, NET_ROWS } from "./cubeNet";
import { Panel } from "./Panel";

const ARROW: Record<string, string> = { up: "▲", right: "▶", down: "▼", left: "◀" };

const FACELET_COLOR = [
  "#f2f2f2", "#e04a2f", "#2f7de0", "#f2c14a", "#3fae6a", "#c94ad6",
];

const rewardColor = (v: number) => (v >= 0 ? "var(--reward-pos)" : "var(--reward-neg)");
const returnColor = (v: number) => (v >= 0 ? "var(--reward-pos)" : "var(--return-neg)");

function RewardReturn({ task }: { task: TaskCore }) {
  return (
    <div>
      reward <span data-reward style={{ color: rewardColor(task.reward) }}>{task.reward}</span>
      {" · return "}
      <span data-return style={{ color: returnColor(task.return) }}>{task.return}</span>
    </div>
  );
}

function GridTaskView({ n, task }: { n: number; task: GridTask | undefined }) {
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

  return (
    <>
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
          <RewardReturn task={task} />
        </div>
      )}
    </>
  );
}

function CubeTaskView({ task }: { task: CubeTask }) {
  const cells = Array.from({ length: NET_ROWS * NET_COLS }, () => -1);
  task.facelets.forEach((color, f) => {
    const { row, col } = cubeNetPosition(f);
    cells[row * NET_COLS + col] = color;
  });
  const held = new Set([12, 16, 21].map((f) => {
    const { row, col } = cubeNetPosition(f);
    return row * NET_COLS + col;
  }));

  return (
    <>
      <div
        data-cube-net
        style={{ display: "grid", gridTemplateColumns: `repeat(${NET_COLS}, 1fr)`, gap: 2, maxWidth: 260, margin: "0 auto" }}
      >
        {cells.map((color, idx) =>
          color < 0 ? (
            <div key={idx} style={{ aspectRatio: "1" }} />
          ) : (
            <div
              key={idx}
              data-facelet
              style={{
                aspectRatio: "1",
                borderRadius: 2,
                background: FACELET_COLOR[color] ?? "var(--edge)",
                boxShadow: held.has(idx) ? "inset 0 0 0 2px var(--text-faint)" : "none",
              }}
            />
          ),
        )}
      </div>
      <div style={{ font: "11px monospace", color: "var(--text-dim)", marginTop: 10, display: "flex", flexDirection: "column", gap: 3 }}>
        <div>
          move {task.move_label ?? "-"} · distance {task.distance ?? "-"} · depth {task.scramble_depth}
        </div>
        <div>solved {task.solved ? "yes" : "no"}</div>
        <RewardReturn task={task} />
      </div>
    </>
  );
}

export function TaskState() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;
  const task = frame?.task;

  if (header.task.type === "cube") {
    return (
      <Panel kicker="PANEL 03 · TASK" title="Task State">
        {task && <CubeTaskView task={task as CubeTask} />}
      </Panel>
    );
  }

  return (
    <Panel kicker="PANEL 03 · TASK" title="Task State">
      <GridTaskView n={header.task.grid_n} task={task as GridTask | undefined} />
    </Panel>
  );
}
