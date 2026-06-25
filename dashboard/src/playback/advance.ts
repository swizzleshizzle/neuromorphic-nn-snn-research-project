export interface PlaybackState {
  winTi: number;
  envStep: number;
  T: number;
  frameCount: number;
}

/** Advance the T-window playhead by one; wrap to the next episode frame at the boundary. */
export function advancePlayback(s: PlaybackState): { winTi: number; envStep: number } {
  const winTi = s.winTi + 1;
  if (winTi >= s.T) {
    const envStep = s.frameCount > 0 ? (s.envStep + 1) % s.frameCount : 0;
    return { winTi: 0, envStep };
  }
  return { winTi, envStep: s.envStep };
}
