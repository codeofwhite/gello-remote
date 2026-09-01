import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vite";

export default defineConfig({
  build: {
    emptyOutDir: true,
    outDir: fileURLToPath(new URL("../src/gello_remote/web/console", import.meta.url)),
  },
  server: {
    fs: { allow: [fileURLToPath(new URL("..", import.meta.url))] },
    proxy: {
      "/control": {
        target: "ws://127.0.0.1:18081",
        ws: true,
      },
      "/teleop-config.json": "http://127.0.0.1:18081",
    },
  },
});
