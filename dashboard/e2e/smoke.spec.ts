import { expect, test } from "@playwright/test";

test("boots on the real trace and renders the data-driven shell", async ({ page }) => {
  await page.goto("/");

  // TopBar shows the brain id from the real header
  await expect(page.getByText(/five-region/i)).toBeVisible();

  // Region Activity renders rows from the header topology
  await expect(page.getByText("Sensory Cortex")).toBeVisible();
  await expect(page.getByText("Hippocampus")).toBeVisible();

  // Task State renders the 5x5 gridworld (25 cells)
  await expect(page.locator("[data-cell]")).toHaveCount(25);

  // The hero canvas mounted
  await expect(page.locator("canvas")).toBeVisible();

  // Hero Cloud/Flow toggle is present and switches
  const toggle = page.locator("[data-hero-toggle]");
  await expect(toggle).toBeVisible();
  await expect(toggle.getByRole("button", { name: "Flow Map" })).toBeVisible();
  await toggle.getByRole("button", { name: "Flow Map" }).click();
  // sensory-input overlay renders from encoding.sensory_input
  await expect(page.locator("[data-sensory-grid]")).toBeVisible();

  // Phase 1a panels render from the real trace
  await expect(page.getByText(/PANEL 04/)).toBeVisible();
  await expect(page.getByText("Communication Flow")).toBeVisible();
  await expect(page.getByText("Spike Raster")).toBeVisible();
  // at least one router gate pill is present
  await expect(page.locator("[data-gate]").first()).toBeVisible();

  // spectator badges render for frozen regions (trace carries policy_regions)
  await expect(page.locator("[data-spectator]").first()).toBeVisible();

  // export control is present and clickable
  const exportBtn = page.locator("[data-export-png]");
  await expect(exportBtn).toBeVisible();
  await exportBtn.click();
});
