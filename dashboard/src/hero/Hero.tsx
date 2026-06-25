import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useTraceStore } from "../store/traceStore";
import { buildNeurons, isSpiking } from "./layout";

const REGION_HUE: Record<string, string> = {
  sensory: "#3fd2ff",
  hippocampus: "#ad8bff",
  prefrontal: "#ffd24a",
  router: "#ff5a8a",
  motor: "#46f0a0",
};

function NeuronCloud() {
  const header = useTraceStore((s) => s.header);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);

  const neurons = useMemo(() => (header ? buildNeurons(header) : []), [header]);
  const baseColors = useMemo(
    () => neurons.map((n) => new THREE.Color(REGION_HUE[n.region] ?? "#8aa")),
    [neurons],
  );

  // Imperative loop: read the store directly; never triggers a React render.
  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh || neurons.length === 0) return;
    const { frames, envStep, winTi } = useTraceStore.getState();
    const frame = frames[envStep];
    neurons.forEach((nrn, i) => {
      dummy.position.set(nrn.x, nrn.y, 0);
      const lit = frame ? isSpiking(frame, nrn.region, winTi, nrn.idx) : false;
      const s = lit ? 0.05 : 0.025;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      color.copy(baseColors[i]).multiplyScalar(lit ? 1 : 0.28);
      mesh.setColorAt(i, color);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  if (neurons.length === 0) return null;
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, neurons.length]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export function Hero() {
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas camera={{ position: [0, 0, 3], fov: 50 }} style={{ background: "#05060a" }}>
        <NeuronCloud />
      </Canvas>
    </div>
  );
}
