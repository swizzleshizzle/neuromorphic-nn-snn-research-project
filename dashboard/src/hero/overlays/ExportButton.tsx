export function ExportButton() {
  const onExport = () => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = "neuroscope-hero.png";
    a.click();
  };
  return (
    <button
      data-export-png
      onClick={onExport}
      style={{
        position: "absolute",
        top: 14,
        right: 120,
        zIndex: 10,
        font: "600 10px/1 'Space Grotesk', sans-serif",
        color: "var(--text-dim)",
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        borderRadius: 8,
        backdropFilter: "var(--blur)",
        padding: "6px 11px",
        cursor: "pointer",
      }}
    >
      Export PNG
    </button>
  );
}
