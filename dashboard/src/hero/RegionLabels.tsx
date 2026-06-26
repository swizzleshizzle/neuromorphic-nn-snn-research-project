import { Html } from "@react-three/drei";
import { useMemo } from "react";
import type { Frame } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { clusterCentroids, type HeroNeuron } from "./layout";
import { hueFor } from "./palette";

export function RegionLabels({ neurons }: { neurons: HeroNeuron[] }) {
  const header = useTraceStore((s) => s.header);
  const envStep = useTraceStore((s) => s.envStep);
  const frames = useTraceStore((s) => s.frames);
  const centroids = useMemo(() => clusterCentroids(neurons, "cloud"), [neurons]);

  if (!header) return null;

  // Explicit annotation so optional-chaining below is type-correct at both
  // compile time (noUncheckedIndexedAccess is off, so frames[i] is Frame) and
  // at runtime (frames may be empty on first render).
  const frame: Frame | undefined = frames[envStep];

  return (
    <>
      {header.regions.map((r, ri) => {
        const c = centroids.get(r.id);
        if (!c) return null;
        const rate = frame?.regions?.[r.id]?.rate ?? 0;
        const hue = hueFor(r.id, ri);
        const isMotor = r.role === "output";

        return (
          <Html
            key={r.id}
            position={[c[0], c[1] - 0.45, c[2]]}
            center
            style={{ pointerEvents: "none" }}
          >
            <div style={{ textAlign: "center", whiteSpace: "nowrap" }}>
              <div
                style={{
                  font: "600 10px/1 'Space Grotesk', sans-serif",
                  color: hue,
                }}
              >
                {r.label.toUpperCase().split(" ")[0]}
              </div>
              <div
                style={{
                  font: "500 8px/1 'IBM Plex Mono', monospace",
                  color: "var(--text-faint)",
                  marginTop: 4,
                }}
              >
                {r.n_neurons}N · {rate.toFixed(2)}
              </div>
              {isMotor &&
                header.task.action_labels.map((lbl, a) => {
                  const selected = frame?.task?.action === a;
                  const gatedShut =
                    (frame?.router?.gate_open?.[a] ?? 1) <= 0.5;
                  const marker = selected ? "◂" : gatedShut ? "✕" : " ";
                  return (
                    <div
                      key={a}
                      style={{
                        font: "500 7px/1.4 'IBM Plex Mono', monospace",
                        color: selected
                          ? hue
                          : gatedShut
                            ? "#555"
                            : "var(--text-faint)",
                      }}
                    >
                      {marker} {lbl}
                    </div>
                  );
                })}
            </div>
          </Html>
        );
      })}
    </>
  );
}
