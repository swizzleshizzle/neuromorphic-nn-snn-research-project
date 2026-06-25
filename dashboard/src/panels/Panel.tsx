import type { ReactNode } from "react";

interface PanelProps {
  kicker: string;
  title: string;
  accent?: string; // CSS color for the accent dot
  children: ReactNode;
}

export function Panel({ kicker, title, accent, children }: PanelProps) {
  return (
    <section
      style={{
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        borderRadius: 11,
        backdropFilter: "var(--blur)",
        margin: 13,
        color: "var(--text)",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "12px 13px 10px",
          borderBottom: "1px solid var(--edge2)",
        }}
      >
        <div style={{ flex: 1 }}>
          <div
            style={{
              font: "600 8px/1 monospace",
              letterSpacing: ".12em",
              color: "var(--text-faint)",
              textTransform: "uppercase",
            }}
          >
            {kicker}
          </div>
          <div style={{ font: "600 13px sans-serif", marginTop: 4 }}>{title}</div>
        </div>
        {accent && (
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: accent,
              boxShadow: `0 0 7px ${accent}`,
              flex: "none",
            }}
          />
        )}
      </header>
      <div style={{ padding: "11px 13px 13px" }}>{children}</div>
    </section>
  );
}
