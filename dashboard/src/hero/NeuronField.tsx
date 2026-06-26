import { useFrame, useThree } from "@react-three/fiber";
import { type MutableRefObject, useMemo, useRef } from "react";
import * as THREE from "three";
import { useTraceStore } from "../store/traceStore";
import { makeGlowTexture } from "./glowTexture";
import { neuronGlow } from "./interp";
import type { HeroNeuron } from "./layout";
import { hueFor } from "./palette";

const UNIT = 0.012; // world units per r3 unit
const white = new THREE.Color("#ffffff");

export function NeuronField({ neurons, morphRef }: { neurons: HeroNeuron[]; morphRef: MutableRefObject<number> }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tex = useMemo(makeGlowTexture, []);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const col = useMemo(() => new THREE.Color(), []);
  const { camera } = useThree();

  const baseColors = useMemo(() => {
    const regionIndex = new Map<string, number>();
    neurons.forEach((n) => regionIndex.set(n.region, regionIndex.get(n.region) ?? regionIndex.size));
    return neurons.map((n) => new THREE.Color(hueFor(n.region, regionIndex.get(n.region) ?? 0)));
  }, [neurons]);

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh || neurons.length === 0) return;
    const { frames, envStep, winTi, T } = useTraceStore.getState();
    const frame = frames[envStep];
    const morph = morphRef.current;
    for (let i = 0; i < neurons.length; i++) {
      const n = neurons[i];
      dummy.position.set(
        n.cloudPos[0] + (n.flowPos[0] - n.cloudPos[0]) * morph,
        n.cloudPos[1] + (n.flowPos[1] - n.cloudPos[1]) * morph,
        n.cloudPos[2] + (n.flowPos[2] - n.cloudPos[2]) * morph,
      );
      dummy.quaternion.copy(camera.quaternion); // billboard
      const g = frame ? neuronGlow(frame, n.region, n.idx, winTi, T) : { sp: 0, act: 0.06 };
      const flash = g.sp ? 1 : 0;
      const size = n.r3 * UNIT * (1 + flash * 1.3);
      dummy.scale.setScalar(size);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      col.copy(baseColors[i]).lerp(white, flash * 0.6).multiplyScalar(Math.min(1, g.act + flash * 0.95));
      mesh.setColorAt(i, col);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  if (neurons.length === 0) return null;
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, neurons.length]}>
      <planeGeometry args={[1, 1]} />
      <meshBasicMaterial
        map={tex}
        transparent
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        toneMapped={false}
      />
    </instancedMesh>
  );
}
