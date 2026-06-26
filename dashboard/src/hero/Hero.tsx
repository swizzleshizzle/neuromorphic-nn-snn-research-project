import { Canvas } from "@react-three/fiber";
import { useRef } from "react";
import { CloudFlowToggle } from "./overlays/CloudFlowToggle";
import { HeroCaption } from "./overlays/HeroCaption";
import { SensoryGrid } from "./overlays/SensoryGrid";
import { Scene } from "./Scene";

export function Hero() {
  const morphRef = useRef(0);
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas camera={{ position: [0, 0, 3.2], fov: 50 }} style={{ background: "var(--bg)" }} dpr={[1, 2]}>
        <Scene morphRef={morphRef} />
      </Canvas>
      <CloudFlowToggle />
      <HeroCaption />
      <SensoryGrid />
    </div>
  );
}
