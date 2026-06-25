import { usePlayback } from "../playback/usePlayback";
import { Hero } from "../hero/Hero";
import { RegionActivity } from "../panels/RegionActivity";
import { TaskState } from "../panels/TaskState";
import { Scrubber } from "./Scrubber";
import { TopBar } from "./TopBar";

export function Shell() {
  usePlayback();
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#05060a" }}>
      <TopBar />
      <div style={{ display: "grid", gridTemplateColumns: "316px minmax(0,1fr) 336px", flex: 1, minHeight: 0 }}>
        <aside style={{ overflow: "auto", borderRight: "1px solid rgba(255,255,255,.06)" }}>
          <RegionActivity />
        </aside>
        <main style={{ position: "relative", minWidth: 0 }}>
          <Hero />
          <Scrubber />
        </main>
        <aside style={{ overflow: "auto", borderLeft: "1px solid rgba(255,255,255,.06)" }}>
          <TaskState />
        </aside>
      </div>
    </div>
  );
}
