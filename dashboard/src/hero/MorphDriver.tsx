import { useFrame } from "@react-three/fiber";
import type { MutableRefObject } from "react";
import { useTraceStore } from "../store/traceStore";
import { damp } from "./interp";

export function MorphDriver({ morphRef }: { morphRef: MutableRefObject<number> }) {
  useFrame((_, dt) => {
    const target = useTraceStore.getState().heroLayout === "flow" ? 1 : 0;
    morphRef.current = damp(morphRef.current, target, 4, dt);
  });
  return null;
}
