import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    fs: { allow: [".."] },
    proxy: {
      // LLM next-word → python scripts/serve.py (OPENROUTER_API_KEY stays server-side)
      "/next": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
