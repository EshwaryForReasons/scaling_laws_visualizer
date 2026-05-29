/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  assetPrefix: ".",
  // Important: do NOT use trailingSlash with assetPrefix: ".".
  // With trailingSlash: true, /designer/index.html resolves "./_next/..."
  // as /designer/_next/..., but the exported _next folder is at /_next.
  trailingSlash: false,

  // Static export has no server redirects anyway, but this avoids relying on
  // slash redirects in builds/deployments that do have a server layer.
  skipTrailingSlashRedirect: true,
};

export default nextConfig;