const { defineConfig, devices } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests/e2e",
  testMatch: "*.spec.js",
  fullyParallel: false,
  timeout: 60000,
  expect: { timeout: 15000 },
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  use: {
    ...devices["Desktop Chrome"],
    channel: process.env.PLAYWRIGHT_BROWSER_CHANNEL || undefined,
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:5000",
    headless: true,
    trace: "retain-on-failure",
  },
});
