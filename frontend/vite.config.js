import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // The .env file lives at the repo root (shared with the backend), not in
  // frontend/, so point loadEnv there instead of Vite's default cwd.
  const env = loadEnv(mode, "..", "");

  return {
    // Relative asset paths, so the built bundle works wherever it is served
    // from — a subpath, a static folder, or straight off disk.
    base: "./",
    plugins: [react()],
    server: {
      port: Number(env.FRONTEND_PORT) || 5173,
      // Proxying /api to the FastAPI server means the browser sees one
      // origin in development, so a misconfigured CORS rule can't break
      // local work.
      proxy: {
        "/api": {
          target: env.BACKEND_URL || "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
