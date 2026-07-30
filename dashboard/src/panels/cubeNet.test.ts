import { describe, expect, it } from "vitest";
import { cubeNetPosition, NET_COLS, NET_ROWS } from "./cubeNet";

describe("cubeNetPosition", () => {
  it("is a bijection over all 24 facelets", () => {
    const seen = new Set<string>();
    for (let f = 0; f < 24; f++) {
      const { row, col } = cubeNetPosition(f);
      expect(row).toBeGreaterThanOrEqual(0);
      expect(row).toBeLessThan(NET_ROWS);
      expect(col).toBeGreaterThanOrEqual(0);
      expect(col).toBeLessThan(NET_COLS);
      seen.add(`${row},${col}`);
    }
    expect(seen.size).toBe(24);
  });

  it("places U in the top band above F", () => {
    // U = facelets 0-3, F = facelets 8-11, F sits directly below U.
    for (let i = 0; i < 4; i++) {
      const u = cubeNetPosition(i);
      const f = cubeNetPosition(8 + i);
      expect(u.col).toBe(f.col);
      expect(f.row - u.row).toBe(2);
    }
  });

  it("places D in the bottom band below F", () => {
    for (let i = 0; i < 4; i++) {
      const f = cubeNetPosition(8 + i);
      const d = cubeNetPosition(12 + i);
      expect(d.col).toBe(f.col);
      expect(d.row - f.row).toBe(2);
    }
  });

  it("orders the middle band L F R B left to right", () => {
    const col = (f: number) => cubeNetPosition(f).col;
    expect(col(16)).toBeLessThan(col(8));  // L before F
    expect(col(8)).toBeLessThan(col(4));   // F before R
    expect(col(4)).toBeLessThan(col(20));  // R before B
  });

  it("keeps every face a contiguous 2x2 block", () => {
    for (let face = 0; face < 6; face++) {
      const pos = [0, 1, 2, 3].map((i) => cubeNetPosition(face * 4 + i));
      const rows = new Set(pos.map((p) => p.row));
      const cols = new Set(pos.map((p) => p.col));
      expect(rows.size).toBe(2);
      expect(cols.size).toBe(2);
      expect(Math.max(...rows) - Math.min(...rows)).toBe(1);
      expect(Math.max(...cols) - Math.min(...cols)).toBe(1);
    }
  });

  it("rejects an out-of-range facelet", () => {
    expect(() => cubeNetPosition(-1)).toThrow();
    expect(() => cubeNetPosition(24)).toThrow();
    expect(() => cubeNetPosition(1.5)).toThrow();
    expect(() => cubeNetPosition(NaN)).toThrow();
  });
});
