export type RenderHint = "dots" | "cloud" | "density";

export interface Region {
  id: string;
  label: string;
  n_neurons: number;
  role: string;
  render: RenderHint;
}

export interface Pathway {
  id: string;
  src: string;
  dst: string;
  gated: boolean;
  label?: string;
}

export interface TraceHeader {
  schema_version: string;
  brain: { id: string; config_hash: string; seed: number; T: number };
  task: { type: string; grid_n: number; action_labels: string[] };
  regions: Region[];
  pathways: Pathway[];
  policy_regions?: string[];
}

export interface RegionState {
  rate: number;
  spikes: number;
  active_frac: number;
  rate_t: number[];
}

export interface PathwayState {
  intensity: number;
  gate_open?: number | number[];
}

export interface RouterState {
  gate_open: number[];
  gate_open_t: number[][];
  utilities: number[];
}

export interface FieldState {
  spikes: number[][]; // [T][N]
}

export interface TaskState {
  agent: [number, number];
  goal: [number, number];
  action: number;
  action_label: string;
  reward: number;
  return: number;
  terminated: boolean;
  truncated: boolean;
}

export interface SensoryInput {
  spikes: number[][]; // [T][2*grid_n^2]
  grid_n: number;
  planes: string[];
  index: string;
}

export interface Frame {
  episode: number;
  step: number;
  t: number;
  task: TaskState;
  regions: Record<string, RegionState>;
  pathways: Record<string, PathwayState>;
  router: RouterState;
  field: Record<string, FieldState>;
  encoding?: { sensory_input: SensoryInput };
}

export interface Trace {
  header: TraceHeader;
  frames: Frame[];
}
