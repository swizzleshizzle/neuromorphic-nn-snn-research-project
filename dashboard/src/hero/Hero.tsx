import { Canvas } from "@react-three/fiber";
import { useRef } from "react";
import { CloudFlowToggle } from "./overlays/CloudFlowToggle";
import { ExportButton } from "./overlays/ExportButton";
import { HeroCaption } from "./overlays/HeroCaption";
import { SensoryGrid } from "./overlays/SensoryGrid";
import { Scene } from "./Scene";

export function Hero() {
  const morphRef = useRef(0);
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas
        camera={{ position: [0, 0, 3.2], fov: 50 }}
        style={{ background: "var(--bg)" }}
        dpr={[1, 2]}
        gl={{ preserveDrawingBuffer: true }}
      >
        <Scene morphRef={morphRef} />
      </Canvas>
      {/* One flex row owns the top-right corner. The two controls used to position
          themselves independently with `right: 18` and `right: 120`, which assumed the
          toggle was about 100px wide; it measures 144px, so they overlapped by 42.5px.
          Laying them out instead of guessing each other's width means a label change
          cannot reintroduce it. `e2e/smoke.spec.ts` asserts they do not intersect. */}
      <div
        data-hero-controls
        style={{
          position: "absolute",
          top: 14,
          right: 18,
          zIndex: 10,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <ExportButton />
        <CloudFlowToggle />
      </div>
      <SensoryGrid />
      <HeroCaption />
    </div>
  );
}
