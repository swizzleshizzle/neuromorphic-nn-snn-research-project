import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { type MutableRefObject, useRef } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

const FLOW_POS = [0, 0, 3.0] as const;

export function CameraRig({ morphRef }: { morphRef: MutableRefObject<number> }) {
  const ref = useRef<OrbitControlsImpl>(null);
  const { camera } = useThree();

  useFrame(() => {
    const m = morphRef.current;
    const ctrl = ref.current;
    if (!ctrl) return;
    // auto-rotate fades out as we approach flow; drag-orbit disabled in flow
    ctrl.autoRotate = m < 0.05;
    ctrl.enabled = m < 0.5;
    if (m > 0.001) {
      // ease the camera toward a locked front view while in/approaching flow
      camera.position.x += (FLOW_POS[0] - camera.position.x) * m * 0.2;
      camera.position.y += (FLOW_POS[1] - camera.position.y) * m * 0.2;
      camera.position.z += (FLOW_POS[2] - camera.position.z) * m * 0.2;
      ctrl.target.x += (0 - ctrl.target.x) * m * 0.2;
      ctrl.target.y += (0 - ctrl.target.y) * m * 0.2;
    }
    ctrl.update();
  });

  return (
    <OrbitControls
      ref={ref}
      enablePan={false}
      autoRotate
      autoRotateSpeed={0.6}
      enableDamping
      minDistance={2}
      maxDistance={6}
      makeDefault
    />
  );
}
