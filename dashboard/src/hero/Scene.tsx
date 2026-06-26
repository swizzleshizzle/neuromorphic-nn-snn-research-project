import { type MutableRefObject, useMemo } from "react";
import * as THREE from "three";
import { useTraceStore } from "../store/traceStore";
import { CameraRig } from "./CameraRig";
import { buildHeroNeurons } from "./layout";
import { NeuronField } from "./NeuronField";
import { Pathways } from "./Pathways";
import { RegionLabels } from "./RegionLabels";

export function Scene({ morphRef }: { morphRef: MutableRefObject<number> }) {
  const header = useTraceStore((s) => s.header);
  const neurons = useMemo(() => (header ? buildHeroNeurons(header) : []), [header]);
  return (
    <>
      <fog attach="fog" args={[new THREE.Color("#05060a"), 3.5, 7.5]} />
      <NeuronField neurons={neurons} morphRef={morphRef} />
      <Pathways neurons={neurons} morphRef={morphRef} />
      <RegionLabels neurons={neurons} />
      <CameraRig morphRef={morphRef} />
    </>
  );
}
