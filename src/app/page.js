const APP_PAGES = [
  {
    route: "interactive",
    title: "Scaling Law Dashboard",
    section: "Interactive controls",
    description:
      "Use sliders for architecture, dataset size, batch size, optimizer settings, and compute estimates. This is the quick interactive sandbox for checking whether a proposed run is reasonable.",
  },
  {
    route: "designer",
    title: "Sweep Designer",
    section: "Experiment planning",
    description:
      "Design 1D and 2D sweeps, inspect tradeoffs, and export sweep configuration JSON that can be reused by scripts or pasted back into the site later.",
  },
  {
    route: "compare",
    title: "Curve Compare",
    section: "Run comparison",
    description:
      "Paste the CSV export from your pandas dataframe and compare saved training results across model sizes, dataset sizes, compute budgets, and validation loss curves.",
  },
  {
    route: "fix",
    title: "Kaplan Diagnostics",
    section: "Scaling diagnosis",
    description:
      "Compare your experimental curves to Kaplan-style expectations and get concrete guidance on whether the sweep looks data-limited, model-limited, undertrained, or noisy.",
  },
  {
    route: "refit",
    title: "Scaling Law Refit",
    section: "Fit your constants",
    description:
      "Refit Kaplan-like scaling laws directly from your CSV results, report fitted constants, show extrapolated curves, and compare your constants against Kaplan reference values.",
  },
];

function routeToStaticHref(route) {
  const clean = String(route ?? "")
    .trim()
    .replace(/^\/+/, "")
    .replace(/\/+$/, "");

  return clean ? `./${clean}.html` : "./index.html";
}

function routeToDisplayPath(route) {
  const clean = String(route ?? "")
    .trim()
    .replace(/^\/+/, "")
    .replace(/\/+$/, "");

  return clean ? `/${clean}` : "/";
}

function PageCard({ page }) {
  const href = routeToStaticHref(page.route);
  const displayPath = routeToDisplayPath(page.route);

  return (
    <a
      href={href}
      className="group block rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4 transition hover:border-blue-500/50 hover:bg-zinc-900/80"
    >
      <div className="flex min-w-0 flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            {page.section}
          </p>
          <span className="rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-0.5 text-xs text-zinc-500">
            {displayPath}
          </span>
        </div>

        <div className="flex items-center justify-between gap-4">
          <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
            {page.title}
          </h2>
          <span className="hidden shrink-0 text-sm font-medium text-zinc-500 transition group-hover:text-blue-300 sm:inline">
            Open →
          </span>
        </div>

        <p className="max-w-3xl text-sm leading-6 text-zinc-400">
          {page.description}
        </p>
      </div>
    </a>
  );
}

export default function HomePage() {
  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="min-h-screen lg:flex lg:items-stretch">
        <aside className="border-b border-zinc-800 bg-zinc-950/95 p-5 lg:sticky lg:top-0 lg:h-screen lg:w-[430px] lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r">
          <div className="mb-6">
            <p className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
              particleGPT
            </p>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
              Scaling Laws Visualizer
            </h1>
            <p className="mt-2 text-sm leading-6 text-zinc-500">
              A launch page for the scaling-law tools in this app. Pick a page
              on the right to continue the workflow.
            </p>
          </div>

          <section className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
              Workflow
            </h2>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4">
              <ol className="space-y-3 text-sm text-zinc-400">
                <li>
                  <span className="font-medium text-zinc-100">1.</span> Check a
                  single setup in the dashboard.
                </li>
                <li>
                  <span className="font-medium text-zinc-100">2.</span> Design a
                  sweep and export the JSON config.
                </li>
                <li>
                  <span className="font-medium text-zinc-100">3.</span> Train
                  the runs and export the dataframe as CSV.
                </li>
                <li>
                  <span className="font-medium text-zinc-100">4.</span> Compare,
                  diagnose, and refit the scaling laws.
                </li>
              </ol>
            </div>
          </section>
        </aside>

        <main className="min-w-0 flex-1 space-y-4 p-4 lg:p-5">
          <header className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-zinc-100">
                  Select a page
                </h1>
                <p className="mt-2 text-sm leading-6 text-zinc-500">
                  Each entry opens one part of the scaling-law workflow. These
                  links target the exported HTML files directly, which is the
                  safest option when using a relative asset prefix.
                </p>
              </div>
            </div>
          </header>

          <section className="grid grid-cols-1 gap-3">
            {APP_PAGES.map((page) => (
              <PageCard key={page.route} page={page} />
            ))}
          </section>
        </main>
      </div>
    </main>
  );
}
