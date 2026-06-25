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
});
