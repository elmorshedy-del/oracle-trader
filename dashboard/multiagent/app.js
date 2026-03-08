const SECTION_ORDER = [
  "runtime",
  "overview",
  "flow",
  "modules",
  "contracts",
  "allocation",
  "observability",
  "weather",
  "crypto",
  "roadmap",
];

const DATA = {
  metrics: [
    { label: "Runtime", value: "isolated_runtime", tone: "good" },
    { label: "Scan Count", value: "0", tone: "good" },
    { label: "Active Markets", value: "0", tone: "good" },
    { label: "Top Blocker", value: "Waiting for runtime data", tone: "warn" },
  ],
  sections: {
    runtime: {
      label: "Runtime",
      description: "Live Oracle runtime translated into the new multi-agent vocabulary.",
      render() {
        return renderRuntimeSection();
      },
    },
    overview: {
      label: "Overview",
      description: "Why this section exists separately and what it actually owns.",
      render() {
        return `
          <div class="content-grid two">
            <article class="content-card">
              <div class="metric-pill">Separate Surface</div>
              <h3 style="margin-top:16px">Oracle Multi-Agent Lab</h3>
              <p>
                This section is the clean room for the Opus plan inside Oracle.
                It does not replace the current trading dashboard. It owns its own
                route, layout, and architecture surface so the next system can be
                designed without destabilizing the current one.
              </p>
            </article>
            <article class="content-card">
              <h3>Plain-language model</h3>
              <div class="soft-stack">
                <div class="soft-row">One orchestrator runs the scan in a strict order.</div>
                <div class="soft-row">Strategies only propose trades and never size them.</div>
                <div class="soft-row">Allocation owns sleeves, caps, reserve cash, and cooldowns.</div>
                <div class="soft-row">Audit writes the explanation artifact after every cycle.</div>
              </div>
            </article>
          </div>
          <div class="content-grid three">
            ${[
              {
                title: "Deterministic flow",
                body: "No autonomous swarm, no side-channel agent chatter, no prompt driving execution.",
              },
              {
                title: "Bounded modules",
                body: "Scanner, enricher, strategies, validator, allocator, executor, state, and audit each keep a narrow contract.",
              },
              {
                title: "Separate build surface",
                body: "This route can grow into the full multi-agent product area without forcing changes into the current Oracle dashboard.",
              },
            ]
              .map(
                (card) => `
                <article class="content-card">
                  <h4>${card.title}</h4>
                  <p>${card.body}</p>
                </article>
              `
              )
              .join("")}
          </div>
        `;
      },
    },
    flow: {
      label: "Scan Flow",
      description: "The deterministic cycle from discovery to audit.",
      stages: [
        {
          name: "Scanner",
          output: "NormalizedMarket[]",
          why: "Discover active markets, normalize API fields, and drop obvious junk early.",
        },
        {
          name: "Enricher",
          output: "MarketContext[]",
          why: "Attach weather, crypto, metadata, relationship, and news context without letting strategies call feeds directly.",
        },
        {
          name: "Strategies",
          output: "SignalCandidate[]",
          why: "Generate trade ideas only. No sizes, no orders, no mutable state writes.",
        },
        {
          name: "Validator",
          output: "ValidatedSignal[]",
          why: "Apply universal staleness, duplicate-position, and ambiguity checks with explicit rejection codes.",
        },
        {
          name: "Allocator",
          output: "ExecutionIntent[]",
          why: "Own sleeves, per-strategy caps, reserve cash, cooldowns, and concentration rules.",
        },
        {
          name: "Executor",
          output: "ExecutionResult[]",
          why: "Simulate fills, slippage, and failures without changing ranking or risk policy.",
        },
        {
          name: "State Manager",
          output: "PortfolioSnapshot",
          why: "Remain the single writer for capital, positions, resolutions, and mark-to-market state.",
        },
        {
          name: "Audit",
          output: "ScanCycleReport",
          why: "Persist exactly why trades were or were not taken so zero-trade cycles are explainable immediately.",
        },
      ],
      render() {
        return `
          ${renderWorkflowIllustration()}
          <article class="content-card">
            <h3>One scan cycle</h3>
            <p>
              This is the central Opus idea: not a real autonomous multi-agent swarm,
              but one strict cycle with narrow handoffs.
            </p>
          </article>
          <div class="flow-rail">
            ${this.stages
              .map(
                (stage, index) => `
                <article class="content-card flow-card">
                  <div class="flow-index">Step ${index + 1}</div>
                  <h4 style="margin-top:12px">${stage.name}</h4>
                  <div class="flow-output">emits ${stage.output}</div>
                  <p>${stage.why}</p>
                </article>
              `
              )
              .join("")}
          </div>
        `;
      },
    },
    modules: {
      label: "Modules",
      description: "Bounded module ownership and the separate Oracle folder tree.",
      cards: [
        {
          name: "scanner",
          owner: "Market discovery + normalization",
          inputs: ["Polymarket discovery APIs", "scanner config"],
          outputs: ["NormalizedMarket[]", "FilteredOut[]"],
          forbidden: ["portfolio state", "position sizing", "execution"],
          notes: "Stateless and deterministic. The only module allowed to touch raw market discovery endpoints.",
        },
        {
          name: "enrichment",
          owner: "Context attachment",
          inputs: ["NormalizedMarket[]", "provider cache"],
          outputs: ["MarketContext[]"],
          forbidden: ["trade decisions", "risk rules", "capital"],
          notes: "Providers may fail independently and degrade to empty enrichment instead of crashing the cycle.",
        },
        {
          name: "strategies",
          owner: "Signal generation only",
          inputs: ["MarketContext[]", "PortfolioSnapshot"],
          outputs: ["SignalCandidate[]"],
          forbidden: ["dollar sizing", "order placement", "mutable portfolio state"],
          notes: "Strategies propose opportunities. They do not know budget and they do not know how execution works.",
        },
        {
          name: "validation",
          owner: "Universal rule enforcement",
          inputs: ["SignalCandidate[]", "PortfolioSnapshot"],
          outputs: ["ValidatedSignal[]", "RejectedSignal[]"],
          forbidden: ["strategy math", "allocation", "execution"],
          notes: "Every rejection gets an explicit code, threshold, and actual value so the operator can explain a quiet cycle.",
        },
        {
          name: "allocation",
          owner: "Sizing + risk constraints",
          inputs: ["ValidatedSignal[]", "PortfolioSnapshot", "risk config"],
          outputs: ["ExecutionIntent[]", "AllocationRejection[]"],
          forbidden: ["signal generation", "external API calls", "fill simulation"],
          notes: "Risk lives here as deterministic constraints, not as a separate talking-head module.",
        },
        {
          name: "audit",
          owner: "Diagnostics + snapshots",
          inputs: ["all cycle emissions"],
          outputs: ["ScanCycleReport", "ModuleHealth[]"],
          forbidden: ["trade decisions"],
          notes: "This is the operator truth source. If a cycle produced zero trades, this module must say why clearly.",
        },
      ],
      repoTree: [
        "dashboard/multiagent/index.html",
        "dashboard/multiagent/styles.css",
        "dashboard/multiagent/app.js",
        "engine/contracts.py",
        "engine/context_builder.py",
        "engine/validator.py",
        "engine/allocator.py",
        "engine/audit.py",
      ],
      render() {
        return `
          <div class="content-grid two">
            ${this.cards
              .map(
                (card) => `
                <article class="content-card">
                  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
                    <div>
                      <div class="panel-eyebrow">Module</div>
                      <h3 style="font-size:32px;margin-top:10px">${card.name}</h3>
                    </div>
                    <span class="module-badge">${card.owner}</span>
                  </div>
                  <div class="soft-stack">
                    <div class="soft-row"><strong>Inputs:</strong> ${card.inputs.join(", ")}</div>
                    <div class="soft-row"><strong>Outputs:</strong> ${card.outputs.join(", ")}</div>
                    <div class="soft-row"><strong>Must not own:</strong> ${card.forbidden.join(", ")}</div>
                    <div class="soft-row">${card.notes}</div>
                  </div>
                </article>
              `
              )
              .join("")}
          </div>
          <article class="content-card">
            <div class="panel-eyebrow">Folder layout</div>
            <h3 style="margin-top:10px">Separate Oracle section</h3>
            <div class="code-block">${this.repoTree.join("\n")}</div>
          </article>
        `;
      },
    },
    contracts: {
      label: "Contracts",
      description: "Typed handoffs so modules cannot leak into each other.",
      cards: [
        {
          name: "NormalizedMarket",
          owner: "scanner",
          producedBy: "scanner.normalize()",
          consumedBy: "enrichment, audit",
          purpose: "Typed market record after raw API normalization and pre-filters. Downstream modules should never read raw dicts.",
          fields: ["market_id", "slug", "question", "category", "outcomes", "close_time"],
        },
        {
          name: "MarketContext",
          owner: "enrichment",
          producedBy: "context builder",
          consumedBy: "strategies, audit",
          purpose: "Single read model for strategies. Merges normalized market data with provider outputs and freshness labels.",
          fields: ["market", "books", "weather", "crypto", "news", "relationships"],
        },
        {
          name: "SignalCandidate",
          owner: "strategy",
          producedBy: "strategy.generate()",
          consumedBy: "validation, audit",
          purpose: "Trade proposal with thesis and edge basis but no size. This boundary keeps strategies away from capital.",
          fields: ["strategy", "market_id", "direction", "edge_basis", "thesis", "priority_hint"],
        },
        {
          name: "ExecutionIntent",
          owner: "allocation",
          producedBy: "allocator.allocate()",
          consumedBy: "execution, audit",
          purpose: "Sized instruction approved for execution. This is where capital decisions become explicit and inspectable.",
          fields: ["strategy", "market_id", "size_usd", "max_slippage_bps", "reason"],
        },
      ],
      render() {
        return `
          <div class="content-grid two">
            ${this.cards
              .map(
                (card) => `
                <article class="content-card">
                  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
                    <div>
                      <div class="panel-eyebrow">Contract</div>
                      <h3 style="font-size:32px;margin-top:10px">${card.name}</h3>
                    </div>
                    <span class="module-badge">${card.owner}</span>
                  </div>
                  <p>${card.purpose}</p>
                  <div class="contract-meta">
                    <div class="meta-box">
                      <div class="meta-label">Produced by</div>
                      <div class="meta-value">${card.producedBy}</div>
                    </div>
                    <div class="meta-box">
                      <div class="meta-label">Consumed by</div>
                      <div class="meta-value">${card.consumedBy}</div>
                    </div>
                  </div>
                  <div class="field-wrap">
                    ${card.fields.map((field) => `<span class="chip">${field}</span>`).join("")}
                  </div>
                </article>
              `
              )
              .join("")}
          </div>
        `;
      },
    },
    allocation: {
      label: "Allocation",
      description: "Sleeves, caps, reserve cash, and deterministic risk policy.",
      groups: [
        {
          title: "Capital sleeves",
          summary: "Reserve capital is explicit. Sparse strategies do not get crowded out by high-frequency ones.",
          rules: [
            "Global reserve cash stays untouched unless a strategy override releases it.",
            "Each strategy receives its own sleeve with current usage shown every cycle.",
            "Weather sub-strategies keep separate sleeves so sniper, latency, and swing remain comparable.",
          ],
        },
        {
          title: "Per-trade approval",
          summary: "A validated signal still must survive deterministic budget and concentration checks before it reaches execution.",
          rules: [
            "No strategy may size itself; only the allocator emits ExecutionIntent.",
            "Same-market re-entry checks happen before sizing, not after a fill attempt.",
            "Thin-book intents must shrink or reject with a surfaced reason code.",
          ],
        },
        {
          title: "Concentration control",
          summary: "The allocator owns correlation, duplicate-event exposure, cooldowns, and stale-position rotation.",
          rules: [
            "Per-market and per-event exposure caps prevent one story from dominating the book.",
            "Cooldowns apply after exits so churn does not become fake activity.",
            "Stale-position rotation trims saturated sleeves before new trades are starved out.",
          ],
        },
      ],
      render() {
        return `
          <div class="content-grid two">
            ${this.groups
              .map(
                (group) => `
                <article class="content-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
                    <h3 style="font-size:32px">${group.title}</h3>
                    <span class="module-badge">deterministic</span>
                  </div>
                  <p>${group.summary}</p>
                  <div class="soft-stack">
                    ${group.rules.map((rule) => `<div class="soft-row">${rule}</div>`).join("")}
                  </div>
                </article>
              `
              )
              .join("")}
          </div>
        `;
      },
    },
    observability: {
      label: "Observability",
      description: "Operator views, blocker reporting, and zero-trade diagnostics.",
      views: [
        {
          title: "System health",
          summary: "Show module status first: scanner, enrichment providers, strategies, validation, allocation, execution, and audit.",
          highlights: [
            "Freshness labels per provider",
            "Degraded mode flags instead of silent failures",
            "Red state whenever a module stops emitting expected artifacts",
          ],
        },
        {
          title: "Signal funnel",
          summary: "One place to see candidates, validations, allocations, executions, and where the funnel collapsed.",
          highlights: [
            "Signals by strategy",
            "Validation rejections by code",
            "Allocation rejections by cap or reserve conflict",
          ],
        },
        {
          title: "Blocker summary",
          summary: "If no trades happened, the app should say why immediately instead of forcing a log dive.",
          highlights: [
            "Top quiet-cycle reason",
            "Held-market saturation",
            "Stale data or missing enrichment",
          ],
        },
        {
          title: "Operator drill-down",
          summary: "Every execution or skip needs a traceable rationale: context, checks, sleeve usage, and final action.",
          highlights: [
            "Trade rationale",
            "Skip rationale",
            "Per-cycle portfolio deltas",
          ],
        },
      ],
      render() {
        return `
          <div class="content-grid two">
            ${this.views
              .map(
                (view) => `
                <article class="content-card">
                  <h3 style="font-size:32px">${view.title}</h3>
                  <p>${view.summary}</p>
                  <div class="soft-stack">
                    ${view.highlights.map((item) => `<div class="soft-row">${item}</div>`).join("")}
                  </div>
                </article>
              `
              )
              .join("")}
          </div>
          <article class="content-card">
            <div class="panel-eyebrow">Quiet-cycle checklist</div>
            <h3 style="margin-top:10px">Why no trades happened</h3>
            <div class="soft-stack">
              ${[
                "Did any strategy emit candidates at all?",
                "Which validation rule blocked the most signals?",
                "Did allocation reject everything on caps or reserve cash?",
                "Did execution fail, or was there nothing approved to execute?",
              ]
                .map((item) => `<div class="soft-row">${item}</div>`)
                .join("")}
            </div>
          </article>
        `;
      },
    },
    weather: {
      label: "Weather",
      description: "Three weather strategies sharing one enrichment layer.",
      strategies: [
        {
          title: "Weather Sniper",
          edge: "High-conviction forecast-vs-market divergence",
          inputs: ["parsed weather question", "forecast values", "historical forecast error", "yes/no market prices"],
          entry: [
            "weather enrichment present and fresh",
            "lead time inside configured band",
            "positive edge above min_edge",
            "no existing position in that market",
          ],
          exits: ["hold to resolution in the first slice", "later add optional trim-on-convergence"],
          failures: ["missing forecast parse", "stale weather data", "missing historical error stats"],
        },
        {
          title: "Weather Latency",
          edge: "Forecast update moved, market has not absorbed it yet",
          inputs: ["current forecast snapshot", "previous forecast snapshot", "current yes price", "previous yes price"],
          entry: [
            "forecast shift exceeds threshold",
            "market price absorption stays below configured ratio",
            "current edge remains positive after repricing math",
          ],
          exits: ["early exit when the market catches up", "resolution fallback if the gap never closes"],
          failures: ["no prior forecast stored yet", "forecast history missing or evicted", "weather provider stale"],
        },
        {
          title: "Weather Swing",
          edge: "Forecast trend over multiple observations",
          inputs: ["rolling forecast history", "linear slope over observation points", "current fair-value estimate"],
          entry: [
            "minimum trend points reached",
            "trend slope exceeds threshold",
            "current edge remains positive after fees",
          ],
          exits: ["fade the position when trend slope breaks or fair value converges"],
          failures: ["not enough history yet", "flat trend", "forecast feed missing for too long"],
        },
      ],
      render() {
        return renderStrategyCards(this.strategies, "Three separate weather strategies sharing one enrichment provider and shared probability math. Sniper, latency, and swing stay independently measurable and independently budgeted.");
      },
    },
    crypto: {
      label: "Crypto",
      description: "Structure and latency without fuzzy directional guessing.",
      strategies: [
        {
          title: "Crypto Structure",
          edge: "Implication breaks and digital-option mispricing",
          inputs: ["asset groupings", "threshold ladder prices", "spot price", "volatility", "hours to resolution"],
          entry: [
            "threshold ladder violates monotonicity",
            "model-vs-market edge exceeds min_edge",
            "market is not already held",
          ],
          exits: ["resolution fallback", "future slice can add convergence exits once structure history exists"],
          failures: ["unparseable threshold market", "missing vol or spot data", "not enough ladder levels"],
        },
        {
          title: "Crypto Latency",
          edge: "Spot move already happened but market has not repriced enough",
          inputs: ["previous spot snapshot", "current spot snapshot", "previous market prices", "current market prices"],
          entry: [
            "spot move exceeds minimum percentage",
            "market absorption ratio stays below cap",
            "new model edge exceeds min_edge",
          ],
          exits: ["exit when absorption ratio normalizes or when the spot move reverses"],
          failures: ["no previous spot snapshot yet", "missing threshold parse", "market already absorbed the move"],
        },
      ],
      render() {
        return renderStrategyCards(this.strategies, "Crypto is split into structure and latency. The guardrail is explicit: no fuzzy directional guessing. Every trade must come from a structural edge basis or a spot-move absorption lag.");
      },
    },
    roadmap: {
      label: "Roadmap",
      description: "Step-by-step build slices in the Oracle engine repo.",
      steps: [
        {
          title: "Slice 1: isolated section",
          detail: "Ship the route, dedicated multiagent folder, module map, deterministic flow, and operator-facing blueprint without touching the current dashboard behavior.",
        },
        {
          title: "Slice 2: typed contracts",
          detail: "Implement NormalizedMarket, MarketContext, SignalCandidate, ValidatedSignal, ExecutionIntent, PortfolioSnapshot, and ScanCycleReport in the Oracle engine.",
        },
        {
          title: "Slice 3: cycle diagnostics",
          detail: "Add tracer, persisted cycle snapshots, zero-trade explanations, and module health so the operator can see blockers immediately.",
        },
        {
          title: "Slice 4: validation + allocation extraction",
          detail: "Move duplicate-position, stale-data, caps, cooldowns, and sleeve math out of strategy code into dedicated validation and allocation layers.",
        },
        {
          title: "Slice 5: weather + crypto split",
          detail: "Promote sniper / latency / swing and crypto_structure / crypto_latency into distinct strategies sharing clean enrichment providers.",
        },
        {
          title: "Slice 6: deferred LLM enrichment",
          detail: "Ask later for bounded LLM help only in news relevance, borderline relationship linking, and rule extraction. Keep probabilities, edge math, allocation, risk, and execution fully deterministic.",
        },
      ],
      render() {
        return `
          <div class="content-grid">
            ${this.steps
              .map(
                (step, index) => `
                <article class="content-card">
                  <div style="display:grid;grid-template-columns:72px 1fr;gap:18px;align-items:start">
                    <div style="width:60px;height:60px;border-radius:999px;background:var(--accent);display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:22px">${index + 1}</div>
                    <div>
                      <h3 style="font-size:32px">${step.title}</h3>
                      <p>${step.detail}</p>
                    </div>
                  </div>
                </article>
              `
              )
              .join("")}
          </div>
        `;
      },
    },
  },
  nonNegotiables: [
    "One orchestrator, not an autonomous agent swarm.",
    "Risk stays deterministic and sits with allocation.",
    "Strategies never size positions or mutate capital directly.",
    "Audit must explain every zero-trade cycle in plain language.",
  ],
};

let activeSection = "runtime";
let LIVE_STATUS = null;
let LIVE_STATUS_ERROR = null;
let LIVE_STATUS_LOADED_AT = null;
let CONSULT_LOADING = false;
let CONSULT_RESPONSE = null;
let CONSULT_QUESTION = "";
let selectedRuntimeView = "all";

function renderMetrics() {
  const metrics = buildMetrics();
  const container = document.getElementById("metrics-grid");
  container.innerHTML = metrics
    .map(
      (metric) => `
        <article class="metric-card ${metric.tone || ""}">
          <div class="metric-label">${metric.label}</div>
          <div class="metric-value">${metric.value}</div>
        </article>
      `
    )
    .join("");
}

function renderNonNegotiables() {
  const list = document.getElementById("non-negotiables");
  if (!list) {
    return;
  }
  list.innerHTML = DATA.nonNegotiables.map((item) => `<li>${item}</li>`).join("");
}

function renderSectionControls() {
  const nav = document.getElementById("section-nav");
  const tabs = document.getElementById("section-tabs");

  if (nav) {
    nav.innerHTML = "";
  }

  if (!tabs) {
    return;
  }

  const tabHtml = SECTION_ORDER.map((key) => {
    const section = DATA.sections[key];
    const active = key === activeSection ? "active" : "";
    return `
      <button class="view-tab ${active}" data-section="${key}">
        ${section.label}
      </button>
    `;
  }).join("");

  tabs.innerHTML = tabHtml;

  document.querySelectorAll("[data-section]").forEach((button) => {
    button.addEventListener("click", () => {
      activeSection = button.getAttribute("data-section");
      renderMetrics();
      renderActiveSection();
      renderSectionControls();
    });
  });
}

function renderStrategyCards(strategies, subtitle) {
  return `
    <article class="content-card">
      <h3>Strategy stack</h3>
      <p>${subtitle}</p>
    </article>
    <div class="content-grid two">
      ${strategies
        .map(
          (strategy) => `
          <article class="content-card">
            <h3 style="font-size:32px">${strategy.title}</h3>
            <div class="module-badge" style="margin-top:14px">${strategy.edge}</div>
            <div class="soft-stack">
              <div class="soft-row"><strong>Inputs:</strong> ${strategy.inputs.join(", ")}</div>
              <div class="soft-row"><strong>Entry:</strong> ${strategy.entry.join("; ")}</div>
              <div class="soft-row"><strong>Exits:</strong> ${strategy.exits.join("; ")}</div>
              <div class="soft-row"><strong>Failure modes:</strong> ${strategy.failures.join("; ")}</div>
            </div>
          </article>
        `
        )
        .join("")}
    </div>
  `;
}

function pnlColor(value) {
  if (Number(value || 0) > 0) {
    return "positive";
  }
  if (Number(value || 0) < 0) {
    return "negative";
  }
  return "neutral";
}

function timeAgo(iso) {
  if (!iso) {
    return "--";
  }
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) {
    return `${Math.floor(diff)}s ago`;
  }
  if (diff < 3600) {
    return `${Math.floor(diff / 60)}m ago`;
  }
  return `${Math.floor(diff / 3600)}h ago`;
}

function truncate(value, length) {
  if (!value) {
    return "";
  }
  return value.length > length ? `${value.slice(0, length)}…` : value;
}

function getRuntimeViews() {
  const views = LIVE_STATUS?.comparison_views || [];
  if (!views.length) {
    return [];
  }
  if (!views.some((view) => view.key === selectedRuntimeView)) {
    selectedRuntimeView = views[0].key;
  }
  return views;
}

function getActiveRuntimeView() {
  const views = getRuntimeViews();
  return (
    views.find((view) => view.key === selectedRuntimeView) ||
    views[0] || {
      key: "all",
      label: "All Opus",
      source: "all",
      portfolio: {},
      performance: {},
      signals: [],
      trades: [],
    }
  );
}

function buildMetrics() {
  if (!LIVE_STATUS) {
    return DATA.metrics;
  }

  if (activeSection === "runtime") {
    const view = getActiveRuntimeView();
    const portfolio = view.portfolio || {};
    const performance = view.performance || {};
    const scope = portfolio.scope || "aggregate";
    if (scope === "aggregate") {
      return [
        {
          label: "Total Value",
          value: formatUsd(portfolio.total_value || 0),
          tone: pnlColor(portfolio.total_pnl || 0) === "negative" ? "bad" : "good",
        },
        {
          label: "Total PnL",
          value: `${Number(portfolio.total_pnl || 0) >= 0 ? "+" : ""}${formatUsd(portfolio.total_pnl || 0)}`,
          tone: pnlColor(portfolio.total_pnl || 0) === "negative" ? "bad" : "good",
        },
        {
          label: "Trades",
          value: String(portfolio.total_trades || 0),
          tone: "good",
        },
        {
          label: "Open Positions",
          value: String(portfolio.open_positions || 0),
          tone: "good",
        },
      ];
    }
    return [
      {
        label: "Strategy PnL",
        value: `${Number(portfolio.total_pnl || 0) >= 0 ? "+" : ""}${formatUsd(portfolio.total_pnl || 0)}`,
        tone: pnlColor(portfolio.total_pnl || 0) === "negative" ? "bad" : "good",
      },
      {
        label: "Exposure",
        value: formatUsd(performance.exposure || portfolio.positions_value || 0),
        tone: "good",
      },
      {
        label: "Trades",
        value: String(portfolio.total_trades || 0),
        tone: "good",
      },
      {
        label: "Signals",
        value: String(performance.signals || (view.signals || []).length || 0),
        tone: "good",
      },
    ];
  }

  return [
    {
      label: "Runtime",
      value: LIVE_STATUS.bridge?.mode || "isolated_runtime",
      tone: "good",
    },
    {
      label: "Scan Count",
      value: String(LIVE_STATUS.summary?.scan_count ?? 0),
      tone: "good",
    },
    {
      label: "Active Markets",
      value: String(LIVE_STATUS.summary?.active_markets ?? 0),
      tone: "good",
    },
    {
      label: "Top Blocker",
      value: LIVE_STATUS.summary?.top_blocker || "No dominant blocker detected",
      tone: LIVE_STATUS.blockers?.length ? "warn" : "good",
    },
  ];
}

function renderWorkflowIllustration() {
  const nodes = buildWorkflowNodes();

  return `
    <article class="content-card workflow-shell">
      <div class="workflow-head">
        <div>
          <div class="panel-eyebrow">System flow</div>
          <h3 style="margin-top:10px">Opus signal river</h3>
          <p>
            A live view of how the isolated engine moves from market discovery to final audit.
            The pulses show the intended handoff path, and the node color reflects current health.
          </p>
        </div>
        <div class="workflow-legend">
          <span class="status-badge healthy">healthy</span>
          <span class="status-badge degraded">degraded</span>
          <span class="status-badge failed">failed</span>
        </div>
      </div>
      <div class="workflow-visual">
        ${nodes
          .map(
            (node, index) => `
              <div class="workflow-node ${escapeHtml(node.status)}" style="--node-index:${index}">
                <div class="workflow-node-glow"></div>
                <div class="workflow-node-body">
                  <div class="workflow-node-title-row">
                    <span class="workflow-node-step">0${index + 1}</span>
                    <span class="status-badge ${escapeHtml(node.status)}">${escapeHtml(node.status)}</span>
                  </div>
                  <h4>${escapeHtml(node.label)}</h4>
                  <div class="workflow-node-meta">${escapeHtml(node.meta)}</div>
                </div>
              </div>
              ${
                index < nodes.length - 1
                  ? `
                    <div class="workflow-lane ${escapeHtml(node.status)}">
                      <div class="workflow-pulse pulse-a"></div>
                      <div class="workflow-pulse pulse-b"></div>
                    </div>
                  `
                  : ""
              }
            `
          )
          .join("")}
      </div>
    </article>
  `;
}

function buildWorkflowNodes() {
  const diagnostics = LIVE_STATUS?.diagnostics || {};
  const signalCount = sumRecordValues(diagnostics.signals_by_strategy || {});

  return [
    {
      label: "Scanner",
      status: findModuleStatus(["scanner", "scanner.discover"]),
      meta: `${LIVE_STATUS?.summary?.active_markets ?? 0} markets discovered`,
    },
    {
      label: "Enricher",
      status: findModuleStatus(["enricher"]),
      meta: `${diagnostics.markets_tradeable ?? LIVE_STATUS?.summary?.active_markets ?? 0} contexts prepared`,
    },
    {
      label: "Strategies",
      status: deriveStrategyStageStatus(signalCount),
      meta: `${signalCount} candidates proposed`,
    },
    {
      label: "Validator",
      status: findModuleStatus(["validator"]),
      meta: `${sumRecordValues(diagnostics.filtered_signals || {})} filtered or rejected`,
    },
    {
      label: "Allocator",
      status: findModuleStatus(["allocator"]),
      meta: `${LIVE_STATUS?.portfolio?.position_count ?? 0} open positions tracked`,
    },
    {
      label: "Executor",
      status: findModuleStatus(["executor"]),
      meta: `${diagnostics.executed ?? 0} executions this cycle`,
    },
    {
      label: "Audit",
      status: normalizeStatus(LIVE_STATUS?.health?.overall_status),
      meta: `${LIVE_STATUS?.summary?.scan_count ?? 0} cycle reports recorded`,
    },
  ];
}

function findModuleStatus(names) {
  const cards = LIVE_STATUS?.module_cards || [];
  for (const name of names) {
    const match = cards.find((card) => card.name === name || card.label === name);
    if (match?.status) {
      return normalizeStatus(match.status);
    }
  }
  return LIVE_STATUS ? "degraded" : "degraded";
}

function deriveStrategyStageStatus(signalCount) {
  const cards = LIVE_STATUS?.strategy_cards || [];
  if (cards.some((card) => normalizeStatus(card.status) === "failed")) {
    return "failed";
  }
  if (signalCount > 0) {
    return "healthy";
  }
  return cards.length ? "degraded" : "degraded";
}

function normalizeStatus(raw) {
  if (raw === "healthy" || raw === "good") {
    return "healthy";
  }
  if (raw === "failed" || raw === "bad") {
    return "failed";
  }
  return "degraded";
}

function sumRecordValues(record) {
  return Object.values(record || {}).reduce((sum, value) => sum + Number(value || 0), 0);
}

function renderRuntimeViewTabs() {
  const views = getRuntimeViews();
  return `
    <div class="runtime-toolbar">
      <div class="runtime-view-tabs">
        ${views
          .map(
            (view) => `
              <button type="button" class="view-tab ${view.key === selectedRuntimeView ? "active" : ""}" data-runtime-view="${escapeHtml(view.key)}">
                ${escapeHtml(view.label)}
              </button>
            `
          )
          .join("")}
      </div>
      <div class="runtime-actions">
        <a href="/api/multiagent/logs/export" class="header-link">Export Opus Logs</a>
      </div>
    </div>
  `;
}

function renderRuntimeSignalsTable(view) {
  const signals = view.signals || [];
  return `
    <article class="content-card">
      <div class="card-header">
        <div class="card-title">Signals</div>
        <div class="card-badge">${signals.length}</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Market</th>
              <th>Action</th>
              <th>Edge</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            ${
              signals.length
                ? signals
                    .map(
                      (signal) => `
                        <tr>
                          <td><span class="strat-badge">${escapeHtml(signal.source)}</span></td>
                          <td title="${escapeHtml(signal.market)}">${escapeHtml(truncate(signal.market, 36))}</td>
                          <td>${escapeHtml((signal.action || "").replaceAll("_", " "))}</td>
                          <td>${Number(signal.edge || 0).toFixed(1)}¢</td>
                          <td title="${escapeHtml(signal.reasoning || "")}">${escapeHtml(truncate(signal.reasoning || "", 44))}</td>
                        </tr>
                      `
                    )
                    .join("")
                : `<tr><td colspan="5" class="table-empty">No signals in the latest Opus scan for this view.</td></tr>`
            }
          </tbody>
        </table>
      </div>
    </article>
  `;
}

function renderRuntimeTradesTable(view) {
  const trades = view.trades || [];
  return `
    <article class="content-card">
      <div class="card-header">
        <div class="card-title">Trades</div>
        <div class="card-badge">${trades.length}</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Strategy</th>
              <th>Market</th>
              <th>Side</th>
              <th>Price</th>
              <th>USD</th>
              <th>PnL</th>
            </tr>
          </thead>
          <tbody>
            ${
              trades.length
                ? trades
                    .map(
                      (trade) => `
                        <tr>
                          <td>${escapeHtml(timeAgo(trade.time))}</td>
                          <td><span class="strat-badge">${escapeHtml(trade.source)}</span></td>
                          <td title="${escapeHtml(trade.market)}">${escapeHtml(truncate(trade.market, 28))}</td>
                          <td>${escapeHtml(trade.side || "")}</td>
                          <td>${formatPrice(trade.price)}</td>
                          <td>${formatUsd(trade.usd || 0)}</td>
                          <td class="${pnlColor(trade.pnl)}">${trade.pnl === null || trade.pnl === undefined ? "—" : formatUsd(trade.pnl)}</td>
                        </tr>
                      `
                    )
                    .join("")
                : `<tr><td colspan="7" class="table-empty">No Opus trades recorded for this view yet.</td></tr>`
            }
          </tbody>
        </table>
      </div>
    </article>
  `;
}

function renderRuntimePositionsTable(view) {
  const positions = view.portfolio?.positions || [];
  return `
    <article class="content-card">
      <div class="card-header">
        <div class="card-title">Open Positions</div>
        <div class="card-badge">${positions.length}</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Market</th>
              <th>Side</th>
              <th>Shares</th>
              <th>Entry</th>
              <th>Current</th>
              <th>PnL</th>
              <th>Strategy</th>
            </tr>
          </thead>
          <tbody>
            ${
              positions.length
                ? positions
                    .map(
                      (position) => `
                        <tr>
                          <td title="${escapeHtml(position.market)}">${escapeHtml(truncate(position.market, 34))}</td>
                          <td>${escapeHtml(position.side || "")}</td>
                          <td>${Number(position.shares || 0).toFixed(1)}</td>
                          <td>${formatPrice(position.entry)}</td>
                          <td>${formatPrice(position.current)}</td>
                          <td class="${pnlColor(position.pnl)}">${formatUsd(position.pnl || 0)}</td>
                          <td><span class="strat-badge">${escapeHtml(position.source)}</span></td>
                        </tr>
                      `
                    )
                    .join("")
                : `<tr><td colspan="7" class="table-empty">No open Opus positions for this view.</td></tr>`
            }
          </tbody>
        </table>
      </div>
    </article>
  `;
}

function renderRuntimeComparisonTable() {
  const views = getRuntimeViews();
  return `
    <article class="content-card">
      <div class="card-header">
        <div class="card-title">Strategy Comparison</div>
        <div class="card-badge">${Math.max(views.length - 1, 0)} sleeves</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>View</th>
              <th>PnL</th>
              <th>Win%</th>
              <th>Trades</th>
              <th>Open</th>
              <th>Signals</th>
              <th>Exposure</th>
            </tr>
          </thead>
          <tbody>
            ${views
              .map(
                (view) => `
                  <tr>
                    <td>${escapeHtml(view.label)}</td>
                    <td class="${pnlColor(view.performance?.total_pnl)}">${formatUsd(view.performance?.total_pnl || 0)}</td>
                    <td>${formatPct(view.performance?.win_rate || 0)}</td>
                    <td>${escapeHtml(String(view.performance?.total_trades || 0))}</td>
                    <td>${escapeHtml(String(view.performance?.open_positions || 0))}</td>
                    <td>${escapeHtml(String(view.performance?.signals || 0))}</td>
                    <td>${formatUsd(view.performance?.exposure || view.portfolio?.positions_value || 0)}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </article>
  `;
}

function renderRuntimeSection() {
  if (LIVE_STATUS_ERROR) {
    return `
      <article class="content-card">
        <div class="panel-eyebrow">Runtime error</div>
        <h3 style="margin-top:10px">Live status unavailable</h3>
        <p>${escapeHtml(LIVE_STATUS_ERROR)}</p>
      </article>
    `;
  }

  if (!LIVE_STATUS) {
    return `
      <article class="content-card">
        <div class="panel-eyebrow">Runtime</div>
        <h3 style="margin-top:10px">Waiting for Opus runtime data</h3>
        <p>The separate multi-agent route is up, but the isolated Opus runtime has not returned its first payload yet.</p>
      </article>
    `;
  }

  const health = LIVE_STATUS.health || {};
  const diagnostics = LIVE_STATUS.diagnostics || {};
  const portfolio = LIVE_STATUS.portfolio || {};
  const performance = LIVE_STATUS.performance || {};
  const blockers = LIVE_STATUS.blockers || [];
  const moduleCards = LIVE_STATUS.module_cards || [];
  const strategyCards = LIVE_STATUS.strategy_cards || [];
  const marketMix = LIVE_STATUS.market_mix || {};
  const marketPreview = LIVE_STATUS.market_preview || [];
  const activeView = getActiveRuntimeView();
  const activePortfolio = activeView.portfolio || {};
  const activePerformance = activeView.performance || {};
  const loadedAt = LIVE_STATUS_LOADED_AT
    ? new Date(LIVE_STATUS_LOADED_AT).toLocaleTimeString()
    : "unknown";

  return `
    <article class="content-card">
      <div class="panel-eyebrow">Live runtime</div>
      <h3 style="margin-top:10px">Opus runtime operator view</h3>
      <p>
        Same reading pattern as the old Oracle dashboard: switch the active sleeve, read the top PnL and trade counts first,
        inspect signals, trades, and open positions, then use the Opus-only sanity, blockers, workflow, and consult layers below.
      </p>
      ${renderRuntimeViewTabs()}
      <div class="contract-meta">
        <div class="meta-box">
          <div class="meta-label">Active view</div>
          <div class="meta-value">${escapeHtml(activeView.label)}</div>
        </div>
        <div class="meta-box">
          <div class="meta-label">View scope</div>
          <div class="meta-value">${escapeHtml(activePortfolio.scope === "aggregate" ? "full opus book" : "strategy filter on shared opus book")}</div>
        </div>
        <div class="meta-box">
          <div class="meta-label">Overall health</div>
          <div class="meta-value">${escapeHtml(health.overall_status || "unknown")}</div>
        </div>
        <div class="meta-box">
          <div class="meta-label">Last runtime refresh</div>
          <div class="meta-value">${escapeHtml(loadedAt)}</div>
        </div>
      </div>
    </article>

    <div class="content-grid three">
      <article class="content-card">
        <div class="panel-eyebrow">Selected view</div>
        <h3 style="margin-top:10px">${formatUsd(activePortfolio.total_pnl || 0)}</h3>
        <div class="soft-stack">
          <div class="soft-row">Realized PnL: ${formatUsd(activePerformance.realized_pnl || 0)}</div>
          <div class="soft-row">Unrealized PnL: ${formatUsd(activePerformance.unrealized_pnl || 0)}</div>
          <div class="soft-row">Exposure: ${formatUsd(activePerformance.exposure || activePortfolio.positions_value || 0)}</div>
        </div>
      </article>
      <article class="content-card">
        <div class="panel-eyebrow">Cycle</div>
        <h3 style="margin-top:10px">Scan ${escapeHtml(String(diagnostics.scan || LIVE_STATUS.summary?.scan_count || 0))}</h3>
        <div class="soft-stack">
          <div class="soft-row">Markets total: ${escapeHtml(String(diagnostics.markets_total || LIVE_STATUS.summary?.active_markets || 0))}</div>
          <div class="soft-row">Markets tradeable: ${escapeHtml(String(diagnostics.markets_tradeable || 0))}</div>
          <div class="soft-row">Executed this scan: ${escapeHtml(String(diagnostics.executed || 0))}</div>
        </div>
      </article>
      <article class="content-card">
        <div class="panel-eyebrow">Performance</div>
        <h3 style="margin-top:10px">${escapeHtml(String(activePerformance.total_trades || activePortfolio.total_trades || 0))} trades</h3>
        <div class="soft-stack">
          <div class="soft-row">Win rate: ${formatPct(activePerformance.win_rate || activePortfolio.win_rate || 0)}</div>
          <div class="soft-row">Signals: ${escapeHtml(String(activePerformance.signals || (activeView.signals || []).length || 0))}</div>
          <div class="soft-row">Open positions: ${escapeHtml(String(activePerformance.open_positions || activePortfolio.open_positions || 0))}</div>
        </div>
      </article>
    </div>

    <div class="content-grid two">
      ${renderRuntimeSignalsTable(activeView)}
      ${renderRuntimeTradesTable(activeView)}
    </div>

    <div class="content-grid two">
      ${renderRuntimePositionsTable(activeView)}
      ${renderRuntimeComparisonTable()}
    </div>

    ${renderSanityPanel()}

    <article class="content-card">
      <div class="panel-eyebrow">Quiet-cycle blockers</div>
      <h3 style="margin-top:10px">What is blocking trades right now</h3>
      ${
        blockers.length
          ? `<div class="content-grid two">${blockers
              .map(
                (blocker) => `
                <article class="content-card blocker-card ${escapeHtml(blocker.severity || "warn")}">
                  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
                    <h4>${escapeHtml(blocker.title)}</h4>
                    <span class="status-badge ${escapeHtml(blocker.severity || "warn")}">${escapeHtml(blocker.severity || "warn")}</span>
                  </div>
                  <p>${escapeHtml(blocker.detail)}</p>
                </article>
              `
              )
              .join("")}</div>`
          : `<p>No dominant blockers were detected in the latest isolated-runtime snapshot.</p>`
      }
    </article>

    <div class="content-grid two">
      <article class="content-card">
        <div class="panel-eyebrow">Runtime totals</div>
        <h3 style="margin-top:10px">Isolated runtime metrics</h3>
        <div class="soft-stack">
          <div class="soft-row">Total candidates: ${escapeHtml(String(performance.total_candidates || 0))}</div>
          <div class="soft-row">Total validated: ${escapeHtml(String(performance.total_validated || 0))}</div>
          <div class="soft-row">Total executed: ${escapeHtml(String(performance.total_executed || 0))}</div>
          <div class="soft-row">Execution rate: ${formatPct(performance.execution_rate || 0)}</div>
          <div class="soft-row">Quiet-cycle rate: ${formatPct(performance.quiet_cycle_rate || 0)}</div>
          <div class="soft-row">Realized PnL: ${formatUsd(performance.realized_pnl || 0)} | Unrealized PnL: ${formatUsd(performance.unrealized_pnl || 0)}</div>
          <div class="soft-row">Closed positions: ${escapeHtml(String(performance.closed_positions || 0))} | Win rate: ${formatPct(performance.win_rate || 0)}</div>
          <div class="soft-row">Average hold: ${Number(performance.avg_hold_hours || 0).toFixed(2)}h</div>
        </div>
        ${
          performance.close_reasons && Object.keys(performance.close_reasons).length
            ? `
              <div class="field-wrap">
                ${Object.entries(performance.close_reasons)
                  .map(
                    ([reason, count]) => `
                      <span class="chip">${escapeHtml(reason)} · ${escapeHtml(String(count))}</span>
                    `
                  )
                  .join("")}
              </div>
            `
            : ""
        }
      </article>
      ${renderConsultPanel()}
    </div>

    <div class="content-grid two">
      <article class="content-card">
        <div class="panel-eyebrow">Core modules</div>
        <h3 style="margin-top:10px">Module health</h3>
        <div class="soft-stack">
          ${moduleCards
            .map(
              (card) => `
                <div class="soft-row">
                  <div class="row-split">
                    <strong>${escapeHtml(card.label)}</strong>
                    <span class="status-badge ${escapeHtml(card.status)}">${escapeHtml(card.status)}</span>
                  </div>
                  <div class="inline-meta">${escapeHtml(card.detail)}</div>
                </div>
              `
            )
            .join("")}
        </div>
      </article>
      <article class="content-card">
        <div class="panel-eyebrow">Runtime mode</div>
        <h3 style="margin-top:10px">${escapeHtml(LIVE_STATUS.bridge?.mode || "unknown")}</h3>
        <div class="soft-stack">
          <div class="soft-row">State: ${escapeHtml(LIVE_STATUS.bridge?.state || "unknown")}</div>
          <div class="soft-row">Next step: ${escapeHtml(LIVE_STATUS.bridge?.next_step || "n/a")}</div>
          <div class="soft-row">Reserve target: ${formatUsd(portfolio.reserved_capital || 0)}</div>
          <div class="soft-row">Logs export: <a href="/api/multiagent/logs/export" class="inline-link">download zip</a></div>
        </div>
      </article>
    </div>

    <article class="content-card">
      <div class="panel-eyebrow">Strategies</div>
      <h3 style="margin-top:10px">Current strategy health</h3>
      <div class="content-grid three">
        ${strategyCards
          .map(
            (card) => `
              <article class="content-card">
                <div class="row-split">
                  <h4>${escapeHtml(card.name)}</h4>
                  <span class="status-badge ${escapeHtml(card.status)}">${escapeHtml(card.status)}</span>
                </div>
                <div class="soft-stack">
                  <div class="soft-row">Runs: ${escapeHtml(String(card.runs || 0))}</div>
                  <div class="soft-row">Signals: ${escapeHtml(String(card.signals || 0))}</div>
                  <div class="soft-row">Errors: ${escapeHtml(String(card.errors || 0))}</div>
                  <div class="soft-row">${card.last_error ? escapeHtml(card.last_error) : "No surfaced strategy error."}</div>
                </div>
              </article>
            `
          )
          .join("")}
      </div>
    </article>

    ${renderWorkflowIllustration()}

    <div class="content-grid two">
      <article class="content-card">
        <div class="panel-eyebrow">Market mix</div>
        <h3 style="margin-top:10px">Current working set</h3>
        <div class="field-wrap">
          ${Object.entries(marketMix)
            .map(
              ([name, count]) => `
                <span class="chip">${escapeHtml(name)} · ${escapeHtml(String(count))}</span>
              `
            )
            .join("")}
        </div>
      </article>
      <article class="content-card">
        <div class="panel-eyebrow">Top markets</div>
        <h3 style="margin-top:10px">Highest-volume sample</h3>
        <div class="soft-stack">
          ${marketPreview
            .map(
              (market) => `
                <div class="soft-row">
                  <strong>${escapeHtml(market.question)}</strong>
                  <div class="inline-meta">${escapeHtml(market.category)} | yes ${formatPrice(market.yes_price)} | liq ${formatUsd(market.liquidity || 0)} | vol ${formatUsd(market.volume_24h || 0)}</div>
                </div>
              `
            )
            .join("")}
        </div>
      </article>
    </div>
  `;
}

function renderConsultPanel() {
  return `
    <article class="content-card">
      <div class="panel-eyebrow">LLM consult</div>
      <h3 style="margin-top:10px">Ask the runtime</h3>
      <p>
        This connector sends the compact Opus diagnostics, cycle reports, blockers, errors, and closed-position history to the model.
      </p>
      <form id="consult-form" class="consult-form">
        <textarea
          id="consult-question"
          class="consult-input"
          placeholder="Why is the isolated runtime not trading enough? Which blockers matter most? What should I fix next?"
        >${escapeHtml(CONSULT_QUESTION)}</textarea>
        <div class="consult-actions">
          <button type="submit" class="button button-primary" ${CONSULT_LOADING ? "disabled" : ""}>
            ${CONSULT_LOADING ? "Consulting..." : "Consult LLM"}
          </button>
        </div>
      </form>
      ${
        CONSULT_RESPONSE
          ? `
            <div class="consult-response">
              <div class="row-split">
                <strong>${escapeHtml(CONSULT_RESPONSE.model || "LLM response")}</strong>
                <span class="inline-meta">${escapeHtml(CONSULT_RESPONSE.generated_at || "")}</span>
              </div>
              <pre class="consult-answer">${escapeHtml(CONSULT_RESPONSE.answer || "")}</pre>
            </div>
          `
          : `
            <div class="inline-meta" style="margin-top:14px">
              The model only sees compact diagnostics, not a raw log flood.
            </div>
          `
      }
    </article>
  `;
}

function renderSanityPanel() {
  const checks = buildSanityChecks();

  return `
    <article class="content-card">
      <div class="panel-eyebrow">Sanity + debug</div>
      <h3 style="margin-top:10px">Operator checks</h3>
      <p>
        Fast readouts for whether the isolated runtime is alive, ingesting fresh data, and producing anything useful.
      </p>
      <div class="content-grid three sanity-grid">
        ${checks
          .map(
            (check) => `
              <article class="content-card sanity-card ${escapeHtml(check.status)}">
                <div class="row-split">
                  <div class="sanity-title-wrap">
                    <span class="sanity-icon ${escapeHtml(check.status)}">${check.status === "healthy" ? "✓" : "✕"}</span>
                    <strong>${escapeHtml(check.title)}</strong>
                  </div>
                  <span class="status-badge ${escapeHtml(check.status)}">${escapeHtml(check.status)}</span>
                </div>
                <div class="inline-meta">${escapeHtml(check.detail)}</div>
              </article>
            `
          )
          .join("")}
      </div>
    </article>
  `;
}

function buildSanityChecks() {
  const health = LIVE_STATUS?.health || {};
  const diagnostics = LIVE_STATUS?.diagnostics || {};
  const strategyCards = LIVE_STATUS?.strategy_cards || [];
  const blockers = LIVE_STATUS?.blockers || [];
  const activeMarkets = Number(LIVE_STATUS?.summary?.active_markets || 0);
  const scanCount = Number(LIVE_STATUS?.summary?.scan_count || 0);
  const signalCount = sumRecordValues(diagnostics.signals_by_strategy || {});
  const filteredCount = sumRecordValues(diagnostics.filtered_signals || {});
  const lastScan = health.scan?.last_scan;
  const overall = normalizeStatus(health.overall_status);
  const strategyReady = strategyCards.some((card) => !String(card.name).includes("no_strategies"));

  return [
    {
      title: "Runtime loop alive",
      status: scanCount > 0 ? "healthy" : "failed",
      detail: scanCount > 0
        ? `${scanCount} isolated scan cycles completed`
        : "The isolated runtime has not completed its first scan yet",
    },
    {
      title: "Market data stream",
      status: activeMarkets > 0 ? "healthy" : "failed",
      detail: activeMarkets > 0
        ? `${activeMarkets} active markets loaded into the isolated runtime`
        : "No active markets were loaded from the shared discovery APIs",
    },
    {
      title: "Scan freshness",
      status: lastScan ? "healthy" : "failed",
      detail: lastScan
        ? `Last scan completed at ${new Date(lastScan).toLocaleTimeString()}`
        : "No completed isolated scan has been recorded",
    },
    {
      title: "Strategy engine",
      status: strategyReady ? "healthy" : "failed",
      detail: strategyReady
        ? `${signalCount} candidates proposed in the latest cycle`
        : "No Opus-native strategy has been migrated into the isolated runtime yet",
    },
    {
      title: "Validation funnel",
      status: signalCount > 0 || filteredCount > 0 ? "healthy" : "failed",
      detail: signalCount > 0 || filteredCount > 0
        ? `${filteredCount} validation/allocation blockers surfaced in the latest cycle`
        : "No candidates reached validation, so the funnel is still empty",
    },
    {
      title: "Audit visibility",
      status: overall === "failed" ? "failed" : "healthy",
      detail: overall === "failed"
        ? "The runtime reported a failed health state; check blockers and module cards below"
        : `Top blocker: ${LIVE_STATUS?.summary?.top_blocker || "none"}`,
    },
    {
      title: "Blocker clarity",
      status: blockers.length ? "healthy" : "failed",
      detail: blockers.length
        ? `${blockers.length} surfaced blocker${blockers.length === 1 ? "" : "s"} rendered in the UI`
        : "No blocker explanation is available yet",
    },
    {
      title: "Portfolio wiring",
      status: LIVE_STATUS?.portfolio ? "healthy" : "failed",
      detail: LIVE_STATUS?.portfolio
        ? `${LIVE_STATUS.portfolio.position_count || 0} open positions tracked in the isolated book`
        : "The isolated portfolio snapshot is missing",
    },
    {
      title: "Overall health",
      status: overall,
      detail: `The isolated runtime reports ${overall} overall status`,
    },
  ];
}

function formatUsd(value) {
  return `$${Number(value || 0).toFixed(2)}`;
}

function formatPct(value) {
  return `${Number(value || 0).toFixed(1)}%`;
}

function formatPrice(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function loadLiveStatus() {
  try {
    const response = await fetch("/api/multiagent/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status fetch failed: ${response.status}`);
    }
    LIVE_STATUS = await response.json();
    LIVE_STATUS_ERROR = null;
    LIVE_STATUS_LOADED_AT = new Date().toISOString();
  } catch (error) {
    LIVE_STATUS_ERROR = error instanceof Error ? error.message : String(error);
  }

  renderMetrics();
  renderActiveSection();
}

async function submitConsult(question) {
  CONSULT_LOADING = true;
  CONSULT_QUESTION = question;
  renderActiveSection();
  bindRuntimeActions();

  try {
    const response = await fetch("/api/multiagent/consult", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    CONSULT_RESPONSE = payload;
  } catch (error) {
    CONSULT_RESPONSE = {
      answer: error instanceof Error ? error.message : String(error),
      model: "connector_error",
      generated_at: new Date().toISOString(),
    };
  } finally {
    CONSULT_LOADING = false;
    renderActiveSection();
    bindRuntimeActions();
  }
}

function bindRuntimeActions() {
  document.querySelectorAll("[data-runtime-view]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedRuntimeView = button.getAttribute("data-runtime-view") || "all";
      renderMetrics();
      renderActiveSection();
    });
  });

  const form = document.getElementById("consult-form");
  if (!form) {
    return;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = document.getElementById("consult-question");
    const question = input?.value?.trim() || "";
    if (!question) {
      return;
    }
    submitConsult(question);
  });
}

function renderActiveSection() {
  const section = DATA.sections[activeSection];
  document.getElementById("active-title").textContent = section.label;
  document.getElementById("active-description").textContent = section.description;
  document.getElementById("section-content").innerHTML = section.render();
  bindRuntimeActions();
}

renderMetrics();
renderNonNegotiables();
renderSectionControls();
renderActiveSection();
loadLiveStatus();
setInterval(loadLiveStatus, 15000);
