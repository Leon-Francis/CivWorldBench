import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: { sourcemap: false, target: "es2022" },
  server: { host: "127.0.0.1", strictPort: true },
});
