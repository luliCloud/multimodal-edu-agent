import type { NextConfig } from "next";

// The browser talks to "/api/*" on the Next origin; Next proxies it to FastAPI. This keeps
// uploads, polling and video range requests same-origin. Set NEXT_PUBLIC_API_URL instead to
// call the API directly (it must then list this origin in CORS_ORIGINS).
const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
