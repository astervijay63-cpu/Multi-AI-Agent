import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build lands in frontend/dist, which api/main.py mounts
// and serves. Nothing else needs to know where the assets live.
//
// During `npm run dev` the Vite server proxies /api to uvicorn so the
// frontend can be developed with hot reload against the real backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
