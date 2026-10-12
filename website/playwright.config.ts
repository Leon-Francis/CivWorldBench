import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  outputDir: "./test-results",
  workers: 1,
  use: { baseURL: process.env.CIVWORLD_BASE_URL || "http://127.0.0.1:4173", headless: true, channel: "chromium" },
  reporter: "list",
});
