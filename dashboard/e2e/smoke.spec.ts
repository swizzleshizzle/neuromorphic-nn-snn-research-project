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

test("the hero controls do not overlap each other", async ({ page }) => {
  // A measured geometric assertion, not a qualitative one. Before the fix these two
  // positioned themselves independently at right:18 and right:120, and overlapped by
  // 42.5px horizontally and 24px vertically at a 1600x1000 viewport - the Export PNG
  // button sat on top of the "3D Cloud" segment. This fails against that code.
  await page.goto("/");
  const toggle = await page.locator("[data-hero-toggle]").boundingBox();
  const exportBtn = await page.locator("[data-export-png]").boundingBox();
  expect(toggle).not.toBeNull();
  expect(exportBtn).not.toBeNull();
  const overlapX =
    Math.min(toggle!.x + toggle!.width, exportBtn!.x + exportBtn!.width) -
    Math.max(toggle!.x, exportBtn!.x);
  const overlapY =
    Math.min(toggle!.y + toggle!.height, exportBtn!.y + exportBtn!.height) -
    Math.max(toggle!.y, exportBtn!.y);
  expect(overlapX > 0 && overlapY > 0, `controls intersect by ${overlapX}x${overlapY}px`).toBe(
    false,
  );
});
