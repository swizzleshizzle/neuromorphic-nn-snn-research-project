import { useFrame } from "@react-three/fiber";
import { type MutableRefObject, useMemo, useRef } from "react";
import * as THREE from "three";
import type { Frame } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { buildEdges, edgeState, quadPoint } from "./edges";
import { lerpVec3 } from "./interp";
import { clusterCentroids, type HeroNeuron } from "./layout";
import { hueFor } from "./palette";

const SEGMENTS = 24;
const PULSES = 3;

export function Pathways({
  neurons,
  morphRef,
}: {
  neurons: HeroNeuron[];
  morphRef: MutableRefObject<number>;
}) {
  const header = useTraceStore((s) => s.header);
  const edges = useMemo(() => (header ? buildEdges(header) : []), [header]);
  const cloudC = useMemo(() => clusterCentroids(neurons, "cloud"), [neurons]);
  const flowC = useMemo(() => clusterCentroids(neurons, "flow"), [neurons]);

  /**
   * THREE.Line objects built in memo and mounted via <primitive>.
   * This avoids the ref-typing conflict between the R3F <line> intrinsic
   * element and the SVG <line> element in TypeScript's DOM lib — no
   * @ts-expect-error suppression needed.
   */
  const lineObjects = useMemo(
    () =>
      edges.map(() => {
        const geo = new THREE.BufferGeometry();
        // LineDashedMaterial so quiescent (gated-closed) edges can render dashed;
        // gapSize is set to 0 each frame when the edge is open, which renders solid.
        const mat = new THREE.LineDashedMaterial({
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          dashSize: 0.05,
          gapSize: 0.05,
        });
        return new THREE.Line(geo, mat);
      }),
    [edges],
  );

  const pulseRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame(({ clock }) => {
    const { frames, envStep } = useTraceStore.getState();
    const frame: Frame | undefined = frames[envStep];
    const m = morphRef.current;
    const t = clock.elapsedTime;
    let pulseI = 0;
    const pmesh = pulseRef.current;

    edges.forEach((e, ei) => {
      const ca = cloudC.get(e.src);
      const fa = flowC.get(e.src);
      const cb = cloudC.get(e.dst);
      const fb = flowC.get(e.dst);
      if (!ca || !fa || !cb || !fb) return;

      // Use lerpVec3 from interp.ts instead of hand-rolling per-axis lerp.
      const a = lerpVec3(ca, fa, m);
      const b = lerpVec3(cb, fb, m);
      const bow = 0.5 * m; // straight in cloud, arced in flow

      const st = edgeState(frame?.pathways?.[e.id], e.gated);

      const lineObj = lineObjects[ei];
      if (lineObj) {
        const pts: THREE.Vector3[] = [];
        for (let s = 0; s <= SEGMENTS; s++) {
          const p = quadPoint(a, b, bow, s / SEGMENTS);
          pts.push(new THREE.Vector3(p[0], p[1], p[2]));
        }
        lineObj.geometry.setFromPoints(pts);
        lineObj.computeLineDistances(); // required for dashes after geometry changes
        const mat = lineObj.material as THREE.LineDashedMaterial;
        mat.color.set(hueFor(e.src, ei));
        mat.opacity = st.quiescent ? 0.16 : 0.06 + st.inten * 0.3;
        mat.gapSize = st.quiescent ? 0.05 : 0; // dashed when gated-closed, solid otherwise
      }

      if (pmesh && !st.quiescent) {
        for (let k = 0; k < PULSES; k++) {
          let pp = (t * 0.18 + k / PULSES + ei * 0.21) % 1;
          if (pp < 0) pp += 1;
          const p = quadPoint(a, b, bow, pp);
          dummy.position.set(p[0], p[1], p[2]);
          dummy.scale.setScalar(0.02 + st.inten * 0.05);
          dummy.updateMatrix();
          pmesh.setMatrixAt(pulseI++, dummy.matrix);
        }
      }
    });

    if (pmesh) {
      // Park unused instances at the origin with zero scale.
      for (let z = pulseI; z < edges.length * PULSES; z++) {
        dummy.position.set(0, 0, 0);
        dummy.scale.setScalar(0);
        dummy.updateMatrix();
        pmesh.setMatrixAt(z, dummy.matrix);
      }
      pmesh.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <>
      {lineObjects.map((obj, i) => (
        <primitive key={edges[i].id} object={obj} />
      ))}
      <instancedMesh
        ref={pulseRef}
        args={[undefined, undefined, Math.max(1, edges.length * PULSES)]}
      >
        <sphereGeometry args={[1, 6, 6]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </instancedMesh>
    </>
  );
}
