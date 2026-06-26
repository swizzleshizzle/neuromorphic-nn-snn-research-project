import { usePlayback } from "../playback/usePlayback";
import { Hero } from "../hero/Hero";
import { CommunicationFlow } from "../panels/CommunicationFlow";
import { RegionActivity } from "../panels/RegionActivity";
import { SpikeRaster } from "../panels/SpikeRaster";
import { TaskState } from "../panels/TaskState";
import { ThalamicRouter } from "../panels/ThalamicRouter";
import { Scrubber } from "./Scrubber";
import { TopBar } from "./TopBar";

export function Shell() {
  usePlayback();
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg)" }}>
      <TopBar />
      <div style={{ display: "grid", gridTemplateColumns: "316px minmax(0,1fr) 336px", flex: 1, minHeight: 0 }}>
        <aside style={{ overflow: "auto", borderRight: "1px solid var(--edge)" }}>
          <RegionActivity />
          <ThalamicRouter />
        </aside>
        <main style={{ position: "relative", minWidth: 0 }}>
          <Hero />
          <Scrubber />
        </main>
        <aside style={{ overflow: "auto", borderLeft: "1px solid var(--edge)" }}>
          <CommunicationFlow />
          <TaskState />
          <SpikeRaster />
        </aside>
      </div>
    </div>
  );
}
