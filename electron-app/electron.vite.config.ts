import { defineConfig } from "electron-vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  main: {
    build: {
      outDir: "dist-electron/main",
      lib: {
        entry: "electron/main.ts",
        formats: ["cjs"],
        fileName: () => "main.js",
      },
      rollupOptions: {
        external: ["electron", "child_process", "path", "fs", "os"],
      },
    },
  },
  preload: {
    build: {
      outDir: "dist-electron/preload",
      lib: {
        entry: "electron/preload.ts",
        formats: ["cjs"],
        fileName: () => "preload.js",
      },
      rollupOptions: {
        external: ["electron"],
      },
    },
  },
  renderer: {
    root: ".",
    build: {
      outDir: "dist-electron/renderer",
      emptyOutDir: false,
      rollupOptions: {
        input: resolve(__dirname, "index.html"),
      },
    },
    plugins: [react()],
  },
});
