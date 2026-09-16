const API_BASE = (window.SIP_API_URL || window.location.origin).replace(/\/$/, "");
const TIMEOUT_MS = 600000;

let threadId = crypto.randomUUID();
let homeChartReady = false;

const PIPE_COPY = [
  {
    title: "Forecast",
    body: "Builds the price path on your chart — recent closes plus a short projected stretch. Agents read this path; they don’t invent those prices.",
  },
  {
    title: "Performance analyst",
    body: "Explains what the forecast path implies: direction, how orderly it looks, and how much weight that signal deserves. The trend label has to match the numbers.",
  },
  {
    title: "Market expert",
    body: "Pulls recent headlines tied to the ticker, then writes tone, analysis, and implications. If nothing useful turns up, news shows as unavailable instead of inventing a story.",
  },
  {
    title: "Financial analyst",
    body: "Reviews company fundamentals — revenue, margins, cash, leverage, valuation — and writes health, strengths, and weaknesses from those figures only.",
  },
];

let loadStepTimer = null;
let chartRange = "1Y";
let activeAgentTab = "a1";
let analyzeInFlight = false;
let runningTicker = null;
let lastAnalyzePayload = null;

const DESK_STORE_MAX = 20;

function emptyDeskStore() {
  return { order: [], byTicker: {} };
}

// In-memory only for this page life — refresh / restart clears the desk.
// (Even with future login, this tray stays visit-scoped, not account history.)
try {
  sessionStorage.removeItem("sip-session-runs-v1");
} catch {
  /* ignore */
}

let sessionStore = emptyDeskStore();

function sessionRunCount() {
  return sessionStore.order.length;
}

function upsertSessionRun(payload) {
  if (!payload || payload.status !== "ok") return;
  const ticker = normalizeTicker(payload.ticker || "");
  if (!ticker) return;
  const existing = sessionStore.byTicker[ticker] || {};
  sessionStore.byTicker[ticker] = {
    ticker,
    savedAt: Date.now(),
    name: (payload.company && payload.company.name) || existing.name || ticker,
    trend: payload.performance_trend || existing.trend || null,
    sentiment: payload.news_sentiment || existing.sentiment || null,
    health: payload.financial_health || existing.health || null,
    cached: Boolean(payload.cached),
    payload,
  };
  sessionStore.order = [ticker, ...sessionStore.order.filter((t) => t !== ticker)].slice(
    0,
    DESK_STORE_MAX
  );
  const keep = new Set(sessionStore.order);
  for (const key of Object.keys(sessionStore.byTicker)) {
    if (!keep.has(key)) delete sessionStore.byTicker[key];
  }
  window._lastAnalyzeTicker = ticker;
  lastAnalyzePayload = payload;
}

function getSessionRun(ticker) {
  const symbol = normalizeTicker(ticker || "");
  const entry = sessionStore.byTicker[symbol];
  return entry ? entry.payload : null;
}

function formatSessionAge(ts) {
  const mins = Math.max(0, Math.round((Date.now() - Number(ts || Date.now())) / 60000));
  if (mins < 1) return "just now";
  if (mins === 1) return "1m ago";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  return hours === 1 ? "1h ago" : `${hours}h ago`;
}

function renderSessionRunsList(activeTicker) {
  const list = $("sessionRunsList");
  const count = $("sessionRunsCount");
  if (!list || !count) return;
  const active = normalizeTicker(activeTicker || window._lastAnalyzeTicker || "");
  const running = analyzeInFlight ? normalizeTicker(runningTicker || "") : "";
  const displayCount = sessionRunCount() + (running && !sessionStore.byTicker[running] ? 1 : 0);
  count.textContent = String(displayCount);
  list.innerHTML = "";

  if (running) {
    const item = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `session-run-btn is-running${running === active ? " active" : ""}`;
    btn.disabled = true;
    btn.innerHTML =
      `<span class="session-run-sym">${running}</span>` +
      `<span class="session-run-meta">Running…</span>`;
    item.appendChild(btn);
    list.appendChild(item);
  }

  if (!sessionStore.order.length && !running) {
    const empty = document.createElement("li");
    empty.className = "muted";
    empty.style.fontSize = "12px";
    empty.textContent = "Nothing on your desk yet.";
    list.appendChild(empty);
    return;
  }
  for (const ticker of sessionStore.order) {
    if (ticker === running) continue; // already shown as Running…
    const entry = sessionStore.byTicker[ticker];
    if (!entry) continue;
    const item = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `session-run-btn${ticker === active ? " active" : ""}`;
    btn.dataset.ticker = ticker;
    const bits = [entry.trend, entry.health, formatSessionAge(entry.savedAt)]
      .filter(Boolean)
      .join(" · ");
    btn.innerHTML =
      `<span class="session-run-sym">${ticker}</span>` +
      `<span class="session-run-meta">${bits}</span>`;
    item.appendChild(btn);
    list.appendChild(item);
  }
}

function renderSessionHomeChips() {
  const host = $("sessionHomeChips");
  if (!host) return;
  host.innerHTML = "";
  for (const ticker of sessionStore.order.slice(0, 10)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "session-home-chip";
    btn.dataset.ticker = ticker;
    btn.textContent = ticker;
    host.appendChild(btn);
  }
}

function downsampleSeries(points, maxPoints = 800) {
  if (!points.length || points.length <= maxPoints) return points;
  const out = [];
  const step = (points.length - 1) / (maxPoints - 1);
  for (let i = 0; i < maxPoints; i += 1) {
    out.push(points[Math.round(i * step)]);
  }
  const last = points[points.length - 1];
  if (out[out.length - 1] !== last) out[out.length - 1] = last;
  return out;
}

function filterHistoryByRange(history, rangeKey) {
  if (!history.length) return [];
  if (rangeKey === "MAX") return history.slice();
  const last = history[history.length - 1].date;
  const end = new Date(`${last}T00:00:00Z`);
  const start = new Date(end);
  if (rangeKey === "1M") start.setUTCMonth(start.getUTCMonth() - 1);
  else if (rangeKey === "6M") start.setUTCMonth(start.getUTCMonth() - 6);
  else if (rangeKey === "1Y") start.setUTCFullYear(start.getUTCFullYear() - 1);
  else if (rangeKey === "5Y") start.setUTCFullYear(start.getUTCFullYear() - 5);
  else start.setUTCFullYear(start.getUTCFullYear() - 1);
  const startIso = start.toISOString().slice(0, 10);
  return history.filter((p) => p.date >= startIso);
}

function setAgentTab(key) {
  activeAgentTab = key;
  document.querySelectorAll(".agent-tab").forEach((btn) => {
    const on = btn.dataset.agentTab === key;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll("[data-agent-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.agentPanel !== key);
  });
}

function renderCompany(company) {
  const panel = $("companyPanel");
  if (!company || (!company.summary && !company.name)) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  $("companyName").textContent = company.name || company.ticker || "—";
  const bits = [company.sector, company.industry].filter(Boolean);
  $("companyMeta").textContent = bits.join(" · ");
  const exchange = [company.exchange, company.ticker].filter(Boolean).join(" · ");
  $("companyExchange").textContent = exchange || "Company overview";
  $("companySummary").textContent =
    company.summary ||
    "No company overview was available from the market data feed for this ticker.";
}

function historyPoints(predictions) {
  return (predictions.history || [])
    .map((row) => ({
      date: String(row.date || "").slice(0, 10),
      value: Number(row.close ?? row.value),
      open: Number(row.open ?? row.close ?? row.value),
      high: Number(row.high ?? row.close ?? row.value),
      low: Number(row.low ?? row.close ?? row.value),
    }))
    .filter((row) => row.date && Number.isFinite(row.value));
}

function setLoadingStep(index) {
  document.querySelectorAll("#loadingSteps li").forEach((el) => {
    const step = Number(el.dataset.loadStep);
    el.classList.toggle("active", step === index);
    el.classList.toggle("done", step < index);
  });
}

function startLoadingUI(ticker) {
  document.body.classList.add("is-running");
  const panel = $("loadingPanel");
  panel.classList.remove("hidden", "is-complete");
  panel.setAttribute("aria-busy", "true");
  $("loadingTitle").textContent = `Analyzing ${ticker}`;
  $("loadingSub").textContent = sessionRunCount()
    ? "Still running — keep reading ready tickers on your desk while you wait."
    : "Forecast, then the three specialists. You can keep browsing this page.";
  $("analyzeBtn").textContent = "Running…";
  $("analyzeBtn").disabled = true;
  let step = 0;
  setLoadingStep(0);
  if (loadStepTimer) clearInterval(loadStepTimer);
  loadStepTimer = setInterval(() => {
    step = Math.min(step + 1, 3);
    setLoadingStep(step);
    if (step >= 3 && loadStepTimer) {
      clearInterval(loadStepTimer);
      loadStepTimer = null;
    }
  }, 12000);
}

function keepPriorResultsOpenDuringRun(nextTicker) {
  if (!sessionRunCount()) return false;
  const next = normalizeTicker(nextTicker || "");
  const viewing = normalizeTicker($("resultsTicker").textContent || "");
  const prefer =
    (viewing && viewing !== "—" && viewing !== next && getSessionRun(viewing) && viewing) ||
    sessionStore.order.find((t) => t !== next) ||
    sessionStore.order[0];
  const payload = getSessionRun(prefer);
  if (!payload) return false;
  // Restore/keep the prior packet so the sidebar stays usable during the wait.
  renderResults(payload);
  return true;
}

function completeLoadingUI(ticker) {
  if (loadStepTimer) {
    clearInterval(loadStepTimer);
    loadStepTimer = null;
  }
  document.body.classList.remove("is-running");
  $("loadingPanel").classList.add("hidden");
  $("loadingPanel").classList.remove("is-complete");
  $("loadingPanel").setAttribute("aria-busy", "false");
  $("analyzeBtn").textContent = "Run agents";
  $("analyzeBtn").disabled = false;
  void ticker;
}

function stopLoadingUI({ hide = true } = {}) {
  document.body.classList.remove("is-running");
  if (hide) {
    $("loadingPanel").classList.add("hidden");
    $("loadingPanel").classList.remove("is-complete");
  }
  $("analyzeBtn").textContent = "Run agents";
  if (!analyzeInFlight) {
    $("analyzeBtn").disabled = false;
  }
  if (loadStepTimer) {
    clearInterval(loadStepTimer);
    loadStepTimer = null;
  }
}

function $(id) {
  return document.getElementById(id);
}

function toneClass(label) {
  const value = String(label || "").toUpperCase();
  if (["BULLISH", "POSITIVE", "STRONG"].includes(value)) return "positive";
  if (["BEARISH", "NEGATIVE", "STRESSED"].includes(value)) return "negative";
  return "";
}

function money(value) {
  return `$${Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function plotLayout(extra = {}) {
  const ink = cssVar("--ink");
  const muted = cssVar("--muted");
  const grid = cssVar("--line");
  const paper = cssVar("--paper");
  return {
    margin: { l: 48, r: 16, t: 24, b: 40 },
    paper_bgcolor: paper,
    plot_bgcolor: paper,
    font: { family: "Roboto, sans-serif", color: muted, size: 11 },
    legend: { orientation: "h", y: 1.12, x: 0, font: { color: ink } },
    xaxis: {
      showgrid: true,
      gridcolor: grid,
      zeroline: false,
      color: muted,
    },
    yaxis: {
      tickprefix: "$",
      showgrid: true,
      gridcolor: grid,
      zeroline: false,
      color: muted,
    },
    hovermode: "x unified",
    dragmode: "pan",
    ...extra,
  };
}

function plotConfig() {
  return {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    toImageButtonOptions: { format: "png", filename: "sip-forecast" },
  };
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("sip-theme", theme);
  $("themeIconSun").classList.toggle("hidden", theme !== "dark");
  $("themeIconMoon").classList.toggle("hidden", theme !== "light");
  $("themeToggle").setAttribute(
    "aria-label",
    theme === "light" ? "Switch to dark theme" : "Switch to light theme"
  );
  if (homeChartReady) renderHomeDemoChart();
  if (!$("results").classList.contains("hidden") && window._lastSeries) {
    renderForecastChart(
      window._lastSeries.history,
      window._lastSeries.forecast,
      chartRange
    );
  }
}

function normalizeTicker(raw) {
  return String(raw || "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9.^=-]/g, "");
}

function forecastPoints(predictions) {
  return (predictions.forecast || [])
    .map((row) => ({
      date: String(row.date || "").slice(0, 10),
      value: Number(row.value ?? row.close),
    }))
    .filter((row) => row.date && Number.isFinite(row.value));
}

function updateSessionResultsChrome(ticker) {
  const symbol = ticker || window._lastAnalyzeTicker;
  const count = sessionRunCount();
  const onResults = !$("results").classList.contains("hidden");
  const onHome = !$("home").classList.contains("hidden");
  const pill = $("resultsReadyPill");
  const bar = $("sessionResultsBar");
  const running = analyzeInFlight ? normalizeTicker(runningTicker || "") : "";

  renderSessionRunsList(symbol);
  renderSessionHomeChips();

  if (!count && !running) {
    pill.classList.add("hidden");
    bar.classList.add("hidden");
    return;
  }

  const latest = symbol || sessionStore.order[0];
  if (running && count) {
    pill.textContent = `Running ${running} · ${count} ready`;
  } else if (running) {
    pill.textContent = `Running · ${running}`;
  } else {
    pill.textContent =
      count === 1 ? `Results · ${latest}` : `On your desk · ${count}`;
  }
  pill.classList.remove("hidden");
  $("sessionResultsLabel").textContent = running
    ? `Running ${running}${count ? ` · ${count} on your desk` : ""}`
    : count === 1
      ? `Results · ${latest}`
      : `On your desk · ${count}`;
  $("sessionResultsHint").textContent = running
    ? "Use the wait — open a ready ticker and compare while the next analysis finishes."
    : "Open analyses from this visit. They clear when you refresh or leave.";

  if (onHome && !onResults && (count || running)) {
    bar.classList.remove("hidden");
  } else {
    bar.classList.add("hidden");
  }
}

function showHome({ preserveTicker = false } = {}) {
  $("home").classList.remove("hidden");
  $("results").classList.add("hidden");
  $("errorBanner").classList.add("hidden");
  if (!analyzeInFlight) {
    $("runBanner").classList.add("hidden");
    $("loadingPanel").classList.add("hidden");
  }
  if (!preserveTicker && !analyzeInFlight) {
    $("tickerInput").value = "";
  }
  document.title = analyzeInFlight
    ? `Running · ${$("tickerInput").value || window._lastAnalyzeTicker || "…"} | Stock Intelligence`
    : "Stock Intelligence";
  updateSessionResultsChrome(window._lastAnalyzeTicker);
  window.scrollTo({ top: 0, behavior: "smooth" });
  renderHomeDemoChart();
}

function showResults(ticker) {
  const symbol = ticker || window._lastAnalyzeTicker || "—";
  $("home").classList.add("hidden");
  $("sessionResultsBar").classList.add("hidden");
  // Keep the sticky loading strip / run banner up while the next ticker executes.
  if (!analyzeInFlight) {
    $("loadingPanel").classList.add("hidden");
    $("runBanner").classList.add("hidden");
  }
  $("results").classList.remove("hidden");
  $("resultsTicker").textContent = symbol;
  document.title = analyzeInFlight && runningTicker
    ? `Running · ${runningTicker} · viewing ${symbol} | Stock Intelligence`
    : `${symbol} | Stock Intelligence`;
  updateSessionResultsChrome(symbol);
  if (window._lastSeries) {
    renderForecastChart(
      window._lastSeries.history,
      window._lastSeries.forecast,
      chartRange
    );
  }
  $("resultsTop").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHomeDemoChart() {
  if (typeof Plotly === "undefined") return;
  const histX = [
    "Aug 18", "Aug 19", "Aug 20", "Aug 21", "Aug 22",
    "Aug 25", "Aug 26", "Aug 27", "Aug 28", "Aug 29",
    "Sep 2", "Sep 3", "Sep 4",
  ];
  const histY = [
    100.0, 100.6, 100.2, 101.1, 101.8,
    101.4, 102.2, 101.9, 102.8, 103.4,
    103.0, 103.7, 104.2,
  ];
  const floor = Math.min(...histY) * 0.997;
  const fcX = ["Sep 4", "Sep 5", "Sep 6", "Sep 7", "Sep 8", "Sep 9"];
  const fcY = [104.2, 104.5, 104.9, 105.2, 105.7, 106.1];
  Plotly.newPlot(
    "homeDemoChart",
    [
      {
        x: histX,
        y: histX.map(() => floor),
        type: "scatter",
        mode: "lines",
        line: { width: 0 },
        hoverinfo: "skip",
        showlegend: false,
      },
      {
        x: histX,
        y: histY,
        type: "scatter",
        mode: "lines",
        name: "Observed",
        line: { color: "#0f9d58", width: 2.2 },
        fill: "tonexty",
        fillcolor: "rgba(15, 157, 88, 0.16)",
      },
      {
        x: fcX,
        y: fcY,
        type: "scatter",
        mode: "lines+markers",
        name: "Forecast",
        line: { color: "#1a73e8", width: 2.2, dash: "dash" },
        marker: { size: 6, color: "#1a73e8" },
      },
    ],
    plotLayout({
      height: 300,
      margin: { l: 48, r: 16, t: 12, b: 36 },
      showlegend: false,
      yaxis: {
        tickprefix: "$",
        showgrid: true,
        gridcolor: cssVar("--line"),
        zeroline: false,
        color: cssVar("--muted"),
      },
    }),
    plotConfig()
  );
  homeChartReady = true;
}

function renderForecastChart(history, forecast, rangeKey = chartRange) {
  const host = $("forecastChart");
  if (!host) return;
  if (typeof Plotly === "undefined") {
    host.innerHTML =
      "<p class='muted' style='padding:16px'>Chart library failed to load. Hard-refresh, or open http://127.0.0.1:8000 in Chrome/Edge.</p>";
    return;
  }
  window._lastSeries = { history, forecast };
  chartRange = rangeKey || chartRange;
  document.querySelectorAll("#chartRanges .range-pill").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.range === chartRange);
  });

  const visible = filterHistoryByRange(history, chartRange);
  const series = downsampleSeries(visible.length ? visible : history, 900);
  if (!series.length) {
    host.innerHTML = "<p class='muted' style='padding:16px'>No price history available for this ticker.</p>";
    return;
  }
  const first = series[0].value;
  const last = series[series.length - 1].value;
  const up = last >= first;
  const histColor = up ? "#0f9d58" : "#d93025";
  const fillColor = up
    ? "rgba(15, 157, 88, 0.18)"
    : "rgba(217, 48, 37, 0.16)";
  const floor = Math.min(...series.map((p) => p.value)) * 0.997;

  const bridge = series[series.length - 1];
  const fc = bridge && forecast.length ? [bridge, ...forecast] : forecast;

  const traces = [
    {
      x: series.map((p) => p.date),
      y: series.map(() => floor),
      type: "scatter",
      mode: "lines",
      line: { width: 0 },
      hoverinfo: "skip",
      showlegend: false,
    },
    {
      x: series.map((p) => p.date),
      y: series.map((p) => p.value),
      type: "scatter",
      mode: "lines",
      name: "Actual",
      hovertemplate: "%{x}<br>%{y:$.2f}<extra>Actual</extra>",
      line: { color: histColor, width: 2.2, shape: "linear" },
      fill: "tonexty",
      fillcolor: fillColor,
    },
  ];
  if (fc.length) {
    traces.push({
      x: fc.map((p) => p.date),
      y: fc.map((p) => p.value),
      type: "scatter",
      mode: "lines+markers",
      name: "Forecast",
      hovertemplate: "%{x}<br>%{y:$.2f}<extra>Forecast</extra>",
      line: { color: "#1a73e8", width: 2.2, dash: "dash" },
      marker: { size: 6, color: "#1a73e8" },
    });
  }

  Plotly.newPlot(
    "forecastChart",
    traces,
    plotLayout({
      height: 420,
      margin: { l: 52, r: 18, t: 18, b: 42 },
      showlegend: true,
      legend: { orientation: "h", y: 1.12, x: 0 },
      yaxis: {
        tickprefix: "$",
        showgrid: true,
        gridcolor: cssVar("--line"),
        zeroline: false,
        color: cssVar("--muted"),
      },
    }),
    plotConfig()
  );
}

function stripLabeledHeader(text, label) {
  const re = new RegExp(`^\\s*${label}\\s*:\\s*`, "i");
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => (re.test(line) ? line.replace(re, "").trim() : null))
    .find((value) => value != null) || "";
}

function proseFromAgentText(text, dropLabels) {
  const drop = new Set(dropLabels.map((label) => label.toLowerCase()));
  const lines = String(text || "").split(/\r?\n/);
  const kept = [];
  for (const line of lines) {
    const header = line.match(/^\s*([A-Za-z][A-Za-z &]+)\s*:/);
    if (header && drop.has(header[1].trim().toLowerCase())) {
      continue;
    }
    kept.push(line);
  }
  return kept.join("\n").trim();
}

function flushParagraph(target, lines) {
  const text = lines.join(" ").trim();
  if (!text) return;
  const para = document.createElement("p");
  para.textContent = text;
  target.appendChild(para);
}

function flushList(target, items) {
  if (!items.length) return;
  const list = document.createElement("ul");
  for (const itemText of items) {
    const item = document.createElement("li");
    item.textContent = itemText;
    list.appendChild(item);
  }
  target.appendChild(list);
}

function openSection(container, title, variant) {
  const card = document.createElement("div");
  card.className = `agent-section${variant ? ` ${variant}` : ""}`;
  const heading = document.createElement("h3");
  heading.textContent = title;
  card.appendChild(heading);
  const body = document.createElement("div");
  body.className = "agent-section-body";
  card.appendChild(body);
  container.appendChild(card);
  return body;
}

function renderProse(container, text) {
  container.innerHTML = "";
  const lines = String(text || "").split(/\r?\n/);
  if (!lines.some((line) => line.trim())) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No analysis returned.";
    container.appendChild(empty);
    return;
  }

  const sectionRe =
    /^(Analysis|Key points|Themes|Caveats?|Drivers|Headlines|Implications|Strengths|Weaknesses)\s*:\s*(.*)$/i;
  const titles = {
    analysis: "Analysis",
    "key points": "Key points",
    themes: "Themes",
    drivers: "Drivers",
    headlines: "Headlines",
    implications: "Implications",
    strengths: "Strengths",
    weaknesses: "Weaknesses",
    caveat: "Caveats",
    caveats: "Caveats",
  };
  const variants = {
    analysis: "is-analysis",
    "key points": "is-points",
    themes: "is-points",
    drivers: "is-points",
    headlines: "is-headlines",
    implications: "is-points",
    strengths: "is-points",
    weaknesses: "is-points",
    caveat: "is-caveat",
    caveats: "is-caveat",
  };

  let sectionBody = openSection(container, "Overview", "is-analysis");
  let paraBuf = [];
  let listBuf = [];
  let openedNamed = false;

  const flush = () => {
    flushList(sectionBody, listBuf);
    listBuf = [];
    flushParagraph(sectionBody, paraBuf);
    paraBuf = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const trimmed = line.trim();
    if (!trimmed) {
      flush();
      continue;
    }
    const section = trimmed.match(sectionRe);
    if (section) {
      flush();
      const key = section[1].toLowerCase();
      if (!openedNamed && sectionBody.childNodes.length === 0) {
        container.innerHTML = "";
      }
      openedNamed = true;
      sectionBody = openSection(container, titles[key] || section[1], variants[key] || "");
      const rest = (section[2] || "").trim();
      if (rest) {
        if (/^[-*•]\s*/.test(rest)) {
          listBuf.push(rest.replace(/^[-*•]\s*/, ""));
        } else {
          paraBuf.push(rest);
        }
      }
      continue;
    }
    if (/^[-*•]\s+/.test(trimmed)) {
      flushParagraph(sectionBody, paraBuf);
      paraBuf = [];
      listBuf.push(trimmed.replace(/^[-*•]\s+/, ""));
      continue;
    }
    flushList(sectionBody, listBuf);
    listBuf = [];
    paraBuf.push(trimmed);
  }
  flush();
  if (!openedNamed && sectionBody.childNodes.length === 0) {
    container.innerHTML = "";
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No analysis returned.";
    container.appendChild(empty);
  }
}

function renderAgentCard(prefix, text) {
  const chips = $(`${prefix}Chips`);
  chips.innerHTML = "";
  const primary =
    prefix === "agent1" ? "Trend" : prefix === "agent2" ? "Sentiment" : "Health";
  const primaryValue = stripLabeledHeader(text, primary) || "—";
  const chip = document.createElement("span");
  chip.className = `agent-chip ${toneClass(primaryValue)}`;
  chip.textContent = primaryValue.split("\n")[0];
  chips.appendChild(chip);

  if (prefix === "agent1") {
    const range = stripLabeledHeader(text, "Range");
    if (range) {
      const rangeChip = document.createElement("span");
      rangeChip.className = "agent-chip neutral";
      rangeChip.textContent = `Forecast span ${range}`;
      chips.appendChild(rangeChip);
    }
  }

  const prose = proseFromAgentText(text, ["Trend", "Range", "Sentiment", "Health"]);
  renderProse($(`${prefix}Body`), prose || text || "");
}

function renderResults(data) {
  const predictions = data.predictions || {};
  const history = historyPoints(predictions);
  const forecast = forecastPoints(predictions);
  const lastClose = predictions.last_close;
  const lastDate = predictions.last_date || "—";
  const terminal = forecast.length ? forecast[forecast.length - 1].value : null;
  const endDate = forecast.length ? forecast[forecast.length - 1].date : "—";
  const horizon = predictions.horizon || forecast.length || 5;
  let signedMove = "—";
  if (terminal != null && lastClose) {
    const pct = ((terminal / Number(lastClose)) - 1) * 100;
    signedMove = `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
  }

  const trend = data.performance_trend || data.recommendation || "—";
  const news = data.news_sentiment || "—";

  const trendEl = $("metricTrend");
  const newsEl = $("metricNews");
  const moveEl = $("metricMove");
  trendEl.textContent = String(trend).toUpperCase();
  newsEl.textContent = String(news).toUpperCase();
  trendEl.className = `metric-value ${toneClass(trend)}`;
  newsEl.className = `metric-value ${toneClass(news)}`;
  $("metricConfidence").textContent = data.confidence || "—";
  moveEl.textContent = signedMove;
  moveEl.className = `metric-value ${
    signedMove.startsWith("+") && signedMove !== "+0.00%"
      ? "positive"
      : signedMove.startsWith("-")
        ? "negative"
        : ""
  }`;

  bindMetricChipExplanations({
    explanations: data.metric_explanations || {},
    values: {
      trend: String(trend).toUpperCase(),
      news: String(news).toUpperCase(),
      confidence: data.confidence || "—",
      projected_move: signedMove,
    },
    tones: {
      trend: toneClass(trend),
      news: toneClass(news),
      confidence: "",
      projected_move: moveEl.className.replace("metric-value", "").trim(),
    },
  });

  $("forecastMeta").innerHTML = `
    <span class="meta-pill" title="Change from last close to the final forecasted session">
      Projected ${horizon}-session move · ${signedMove}
    </span>
    <span class="meta-pill">Last close ${
      lastClose != null ? money(lastClose) : "—"
    } · ${lastDate}</span>
    <span class="meta-pill">History · ${history.length.toLocaleString()} sessions</span>
    <span class="meta-pill">Forecast to ${
      terminal != null ? money(terminal) : "—"
    } · ${endDate}</span>
  `;

  renderCompany(data.company || {});
  renderAgentCard("agent1", data.performance_analysis || "");
  renderAgentCard("agent2", data.news_summary || "");
  renderAgentCard("agent3", data.financial_analysis || "");
  setAgentTab("a1");

  const newsPayload = data.news || {};
  const articles = newsPayload.articles || [];
  $("sourcesMeta").textContent =
    `${newsPayload.provider || "—"} · ${articles.length} ticker-relevant`;

  const list = $("sourcesList");
  list.innerHTML = "";
  if (!articles.length) {
    const empty = document.createElement("li");
    empty.textContent = newsPayload.error || "No sources";
    empty.style.color = "var(--muted)";
    list.appendChild(empty);
  } else {
    for (const article of articles.slice(0, 12)) {
      const item = document.createElement("li");
      const title = article.headline || "Untitled";
      const url = article.url || "";
      if (article.scope === "peer" && article.peer_ticker) {
        const badge = document.createElement("span");
        badge.className = "source-badge";
        badge.textContent = `Peer · ${article.peer_ticker}`;
        item.appendChild(badge);
      }
      if (url) {
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = title;
        item.appendChild(link);
      } else {
        const span = document.createElement("span");
        span.textContent = title;
        item.appendChild(span);
      }
      const meta = document.createElement("span");
      meta.className = "source-meta";
      meta.textContent = [article.date, article.publisher || article.source]
        .filter(Boolean)
        .join(" · ");
      item.appendChild(meta);
      list.appendChild(item);
    }
  }

  const body = $("forecastTableBody");
  body.innerHTML = "";
  for (const point of forecast) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${point.date}</td><td>${money(point.value)}</td>`;
    body.appendChild(row);
  }

  const ticker = data.ticker || $("tickerInput").value || "—";
  upsertSessionRun(data);
  const perfOk = data.performance_guardrail_ok;
  const newsOk = data.news_guardrail_ok;
  const finOk = data.financial_guardrail_ok;
  let checks = "";
  if (perfOk === false || newsOk === false || finOk === false) {
    checks = " · some labels needed extra review";
  }

  const badge = $("cacheBadge");
  if (data.cached) {
    const ageSec = Number(data.cache_age_seconds || 0);
    const ttlSec = Number(data.cache_ttl_seconds || 0);
    const ageMin = Math.max(0, Math.round(ageSec / 60));
    const leftMin =
      ttlSec > 0 ? Math.max(0, Math.ceil((ttlSec - ageSec) / 60)) : null;
    badge.classList.remove("hidden", "is-fresh");
    badge.classList.add("is-cached");
    badge.textContent =
      leftMin == null
        ? `Recent result · ${ageMin}m ago`
        : `Recent result · ${ageMin}m ago · updates in ~${leftMin}m`;
  } else {
    badge.classList.remove("hidden", "is-cached");
    badge.classList.add("is-fresh");
    badge.textContent = "Fresh analysis";
  }

  $("runMeta").textContent =
    `${data.cached ? "Showing a recent saved result" : "Fresh analysis"} · research support only${checks}`;

  showResults(ticker);
  markResultsReady(ticker);
  renderForecastChart(history, forecast, chartRange);
  renderSessionRunsList(ticker);
}

async function checkReady() {
  const pill = $("apiStatus");
  try {
    const response = await fetch(`${API_BASE}/ready`, { signal: AbortSignal.timeout(4000) });
    pill.className = `status-pill ${response.ok ? "ok" : "off"}`;
    pill.textContent = response.ok ? "API ready" : "API error";
    return response.ok;
  } catch {
    pill.className = "status-pill off";
    pill.textContent = "API down";
    return false;
  }
}

function markResultsReady(ticker) {
  window._lastAnalyzeTicker = ticker;
  updateSessionResultsChrome(ticker);
  document.title = `${ticker} | Stock Intelligence`;
  $("runBanner").classList.add("hidden");
}

async function analyze(ticker, { forceRefresh = false } = {}) {
  if (analyzeInFlight) {
    $("runBanner").classList.remove("hidden");
    $("runBanner").className = "run-banner running";
    $("runBanner").textContent =
      `Already analyzing ${runningTicker || $("tickerInput").value || "a ticker"}… use On your desk to review ready tickers while you wait.`;
    return;
  }

  analyzeInFlight = true;
  runningTicker = ticker;
  $("errorBanner").classList.add("hidden");
  $("runBanner").classList.remove("hidden");
  $("runBanner").className = "run-banner running";
  const hasPrior = sessionRunCount() > 0;
  $("runBanner").textContent = forceRefresh
    ? `Refreshing ${ticker}… starting a brand-new analysis.`
    : hasPrior
      ? `Running ${ticker}… prior results stay open — compare from On your desk while this finishes.`
      : `Running ${ticker}… you can keep browsing this page. Results appear when ready.`;
  document.title = `Running · ${ticker} | Stock Intelligence`;

  // With prior runs: keep results + sidebar so wait time is usable for comparison.
  // First run in the sitting: stay on home under the loading strip.
  if (hasPrior) {
    keepPriorResultsOpenDuringRun(ticker);
  } else {
    $("home").classList.remove("hidden");
    $("results").classList.add("hidden");
  }
  startLoadingUI(ticker);
  updateSessionResultsChrome(window._lastAnalyzeTicker);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticker,
        thread_id: threadId,
        force_refresh: Boolean(forceRefresh),
      }),
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(
        typeof payload.detail === "string"
          ? payload.detail
          : payload.detail
            ? JSON.stringify(payload.detail)
            : `HTTP ${response.status}`
      );
    }
    if (payload.status !== "ok") {
      throw new Error(payload.detail || payload.status || "Analyze failed");
    }
    lastAnalyzePayload = payload;
    window._lastAnalyzeTicker = ticker;
    runningTicker = null;
    analyzeInFlight = false;
    stopLoadingUI({ hide: true });
    completeLoadingUI(ticker);
    renderResults(payload);
  } catch (error) {
    runningTicker = null;
    analyzeInFlight = false;
    stopLoadingUI({ hide: true });
    // On failure, keep prior results if any; otherwise return home.
    if (sessionRunCount()) {
      const fallback =
        getSessionRun(window._lastAnalyzeTicker) ||
        getSessionRun(sessionStore.order[0]);
      if (fallback) renderResults(fallback);
      else {
        $("results").classList.add("hidden");
        $("home").classList.remove("hidden");
      }
    } else {
      $("results").classList.add("hidden");
      $("home").classList.remove("hidden");
      renderHomeDemoChart();
    }
    $("runBanner").classList.add("hidden");
    $("errorBanner").classList.remove("hidden");
    $("errorBanner").textContent =
      error.name === "AbortError"
        ? "Timed out waiting for analysis. Try again."
        : String(error.message || error);
    document.title = window._lastAnalyzeTicker
      ? `${window._lastAnalyzeTicker} | Stock Intelligence`
      : "Stock Intelligence";
    updateSessionResultsChrome(window._lastAnalyzeTicker);
  } finally {
    clearTimeout(timer);
    analyzeInFlight = false;
    runningTicker = null;
    $("analyzeBtn").disabled = false;
  }
}

function setPipelineStep(index) {
  document.querySelectorAll(".flow-step").forEach((el) => {
    const on = Number(el.dataset.step) === index;
    el.classList.toggle("active", on);
    el.setAttribute("aria-selected", on ? "true" : "false");
  });
  const item = PIPE_COPY[index];
  $("pipelineDetail").innerHTML = `<strong>${item.title}</strong>${item.body}`;
}

function wireHome() {
  setPipelineStep(0);

  document.querySelectorAll(".flow-step").forEach((el) => {
    el.addEventListener("click", () => setPipelineStep(Number(el.dataset.step)));
  });

  $("quickTickers").addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-ticker]");
    if (!btn) return;
    if (analyzeInFlight) {
      $("runBanner").classList.remove("hidden");
      $("runBanner").className = "run-banner running";
      $("runBanner").textContent =
        `Already analyzing ${runningTicker || "a ticker"}… open a ready ticker on your desk while you wait.`;
      return;
    }
    const ticker = btn.dataset.ticker;
    $("tickerInput").value = ticker;
    analyze(ticker);
  });

  document.querySelectorAll("#chartRanges .range-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!window._lastSeries) return;
      renderForecastChart(
        window._lastSeries.history,
        window._lastSeries.forecast,
        btn.dataset.range
      );
    });
  });

  document.querySelectorAll(".agent-tab").forEach((btn) => {
    btn.addEventListener("click", () => setAgentTab(btn.dataset.agentTab));
  });
}

function resetChartView() {
  const el = $("forecastChart");
  if (!el || typeof Plotly === "undefined" || !window._lastSeries) return;
  // newPlot alone often keeps pan/zoom; purge + redraw fully restores the view.
  try {
    Plotly.purge(el);
  } catch (_) {
    /* ignore */
  }
  renderForecastChart(
    window._lastSeries.history,
    window._lastSeries.forecast,
    chartRange
  );
}

$("analyzeForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const ticker = normalizeTicker($("tickerInput").value);
  if (!ticker) {
    $("errorBanner").classList.remove("hidden");
    $("errorBanner").textContent = "Enter a valid ticker.";
    return;
  }
  $("tickerInput").value = ticker;
  analyze(ticker);
});

$("themeToggle").addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  setTheme(current === "light" ? "dark" : "light");
});

$("brandHome").addEventListener("click", (event) => {
  event.preventDefault();
  showHome({ preserveTicker: true });
});

const resultsBackBtn = $("resultsBack");
if (resultsBackBtn) {
  resultsBackBtn.addEventListener("click", (event) => {
    event.preventDefault();
    showHome({ preserveTicker: true });
  });
}

function openStoredResults(ticker) {
  const symbol = normalizeTicker(
    ticker || window._lastAnalyzeTicker || sessionStore.order[0] || ""
  );
  if (!symbol) return;
  const payload = getSessionRun(symbol) || lastAnalyzePayload;
  if (!payload) return;
  $("tickerInput").value = symbol;
  // Re-render the exact stored packet into the same results UI — no LLM.
  renderResults(payload);
}

$("resultsReadyPill").addEventListener("click", () => {
  openStoredResults();
});

const sessionResultsOpen = $("sessionResultsOpen");
if (sessionResultsOpen) {
  sessionResultsOpen.addEventListener("click", () => openStoredResults());
}

const sessionRunsList = $("sessionRunsList");
if (sessionRunsList) {
  sessionRunsList.addEventListener("click", (event) => {
    const btn = event.target.closest(".session-run-btn");
    if (!btn || !btn.dataset.ticker) return;
    openStoredResults(btn.dataset.ticker);
  });
}

const sessionHomeChips = $("sessionHomeChips");
if (sessionHomeChips) {
  sessionHomeChips.addEventListener("click", (event) => {
    const btn = event.target.closest(".session-home-chip");
    if (!btn || !btn.dataset.ticker) return;
    openStoredResults(btn.dataset.ticker);
  });
}

const chartResetBtn = $("chartReset");
if (chartResetBtn) {
  chartResetBtn.addEventListener("click", (event) => {
    event.preventDefault();
    resetChartView();
  });
}

const refreshAnalyzeBtn = $("refreshAnalyze");
if (refreshAnalyzeBtn) {
  refreshAnalyzeBtn.addEventListener("click", (event) => {
    event.preventDefault();
    const ticker = normalizeTicker(
      window._lastAnalyzeTicker || $("tickerInput").value || $("resultsTicker").textContent
    );
    if (!ticker || ticker === "—") {
      $("errorBanner").classList.remove("hidden");
      $("errorBanner").textContent = "No ticker to refresh.";
      return;
    }
    $("tickerInput").value = ticker;
    analyze(ticker, { forceRefresh: true });
  });
}

const METRIC_CHIP_LABELS = {
  trend: "Trend",
  news: "News",
  confidence: "Confidence",
  projected_move: "Projected move",
};

const METRIC_CHIP_FALLBACKS = {
  trend:
    "Trend summarizes what the Performance Analyst read from the forecast path on this short horizon — upward, downward, or sideways — not a buy or sell call.",
  news:
    "News summarizes the Market Expert’s reading of recent headline tone for this ticker. Coverage can shift quickly and does not replace the forecast path.",
  confidence:
    "Confidence reflects how steady this run’s path reading looked. Lower confidence means treat the chips as a lighter signal and read the agent notes.",
  projected_move:
    "Projected move is the percent change from the last close to the final forecast session. It describes the path stretch, not a promised return.",
};

let metricChipState = {
  explanations: {},
  values: {},
  tones: {},
  activeKey: null,
  activeChip: null,
  hoverTimer: null,
  pinned: false,
};

function bindMetricChipExplanations({ explanations, values, tones }) {
  metricChipState.explanations = explanations || {};
  metricChipState.values = values || {};
  metricChipState.tones = tones || {};
}

function closeMetricPopover() {
  const pop = $("metricPopover");
  if (!pop) return;
  pop.classList.add("hidden");
  pop.classList.remove("is-visible");
  pop.hidden = true;
  pop.setAttribute("aria-hidden", "true");
  metricChipState.activeKey = null;
  metricChipState.pinned = false;
  if (metricChipState.activeChip) {
    metricChipState.activeChip.classList.remove("is-open");
    metricChipState.activeChip.setAttribute("aria-expanded", "false");
    metricChipState.activeChip = null;
  }
  document.querySelectorAll(".metric.is-open").forEach((el) => {
    el.classList.remove("is-open");
    el.setAttribute("aria-expanded", "false");
  });
}

function positionMetricPopover(chip) {
  const pop = $("metricPopover");
  if (!pop || !chip) return;

  pop.style.left = "0px";
  pop.style.top = "0px";
  const gap = 8;
  const margin = 12;
  const rect = chip.getBoundingClientRect();
  const popRect = pop.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  let left = rect.left + rect.width / 2 - popRect.width / 2;
  left = Math.max(margin, Math.min(left, vw - popRect.width - margin));

  let top = rect.bottom + gap;
  if (top + popRect.height > vh - margin) {
    top = rect.top - popRect.height - gap;
  }
  top = Math.max(margin, Math.min(top, vh - popRect.height - margin));

  pop.style.left = `${Math.round(left)}px`;
  pop.style.top = `${Math.round(top)}px`;
}

function openMetricPopover(key, chip, { pinned = false } = {}) {
  const pop = $("metricPopover");
  const labelEl = $("metricPopoverLabel");
  const valueEl = $("metricPopoverValue");
  const bodyEl = $("metricPopoverBody");
  if (!pop || !labelEl || !valueEl || !bodyEl || !chip) return;

  const label = METRIC_CHIP_LABELS[key] || key;
  const value = metricChipState.values[key] || "—";
  const tone = metricChipState.tones[key] || "";
  const text =
    metricChipState.explanations[key] ||
    METRIC_CHIP_FALLBACKS[key] ||
    "Explanation unavailable for this run.";

  labelEl.textContent = label;
  valueEl.textContent = value;
  valueEl.className = `metric-popover-value ${tone}`.trim();
  bodyEl.textContent = text;

  document.querySelectorAll(".metric.is-open").forEach((el) => {
    el.classList.remove("is-open");
    el.setAttribute("aria-expanded", "false");
  });
  chip.classList.add("is-open");
  chip.setAttribute("aria-expanded", "true");

  metricChipState.activeKey = key;
  metricChipState.activeChip = chip;
  metricChipState.pinned = pinned;

  pop.classList.remove("hidden");
  pop.hidden = false;
  pop.setAttribute("aria-hidden", "false");
  positionMetricPopover(chip);
  // Second pass after paint so width is accurate
  requestAnimationFrame(() => {
    positionMetricPopover(chip);
    pop.classList.add("is-visible");
  });
}

function wireMetricChips() {
  const grid = $("summaryGrid");
  const pop = $("metricPopover");
  if (!grid || !pop) return;

  const canHover =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  grid.querySelectorAll(".metric[data-metric]").forEach((chip) => {
    const key = chip.getAttribute("data-metric");
    chip.setAttribute("aria-expanded", "false");
    chip.setAttribute("aria-haspopup", "true");

    chip.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (
        metricChipState.activeKey === key &&
        !pop.hidden &&
        metricChipState.pinned
      ) {
        closeMetricPopover();
        return;
      }
      openMetricPopover(key, chip, { pinned: true });
    });

    if (canHover) {
      chip.addEventListener("mouseenter", () => {
        window.clearTimeout(metricChipState.hoverTimer);
        if (metricChipState.pinned && metricChipState.activeKey !== key) {
          metricChipState.pinned = false;
        }
        openMetricPopover(key, chip, { pinned: metricChipState.pinned });
      });
      chip.addEventListener("mouseleave", () => {
        if (metricChipState.pinned) return;
        window.clearTimeout(metricChipState.hoverTimer);
        metricChipState.hoverTimer = window.setTimeout(() => {
          if (!pop.matches(":hover")) closeMetricPopover();
        }, 120);
      });
    }

    chip.addEventListener("focus", () => {
      openMetricPopover(key, chip, { pinned: !canHover });
    });
  });

  pop.addEventListener("mouseenter", () => {
    window.clearTimeout(metricChipState.hoverTimer);
  });
  pop.addEventListener("mouseleave", () => {
    if (metricChipState.pinned) return;
    closeMetricPopover();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMetricPopover();
  });

  document.addEventListener("click", (event) => {
    if (pop.hidden) return;
    if (event.target.closest(".metric") || event.target.closest("#metricPopover")) {
      return;
    }
    closeMetricPopover();
  });

  window.addEventListener(
    "scroll",
    () => {
      if (pop.hidden || !metricChipState.activeChip) return;
      positionMetricPopover(metricChipState.activeChip);
    },
    true
  );
  window.addEventListener("resize", () => {
    if (pop.hidden || !metricChipState.activeChip) return;
    positionMetricPopover(metricChipState.activeChip);
  });
}

function escapeHelpHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function cleanHelpMarkdown(text) {
  return String(text || "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/\*([^*\n]+?)\*/g, "$1")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/`+/g, "");
}

function parseHelpNextLine(raw) {
  const lines = String(raw || "").replace(/\r\n/g, "\n").split("\n");
  let next = [];
  const kept = [];
  for (const line of lines) {
    const match = line.match(/^\s*NEXT:\s*(.+)\s*$/i);
    if (match) {
      next = match[1]
        .split("|")
        .map((part) => part.trim())
        .filter(Boolean)
        .slice(0, 4);
    } else {
      kept.push(line);
    }
  }
  return { body: kept.join("\n").trim(), next };
}

function renderHelpRichHtml(raw) {
  const parsed = parseHelpNextLine(raw);
  const body = cleanHelpMarkdown(parsed.body);
  const next = parsed.next;
  if (!body) {
    return { html: "", next };
  }
  if (/^out of scope/i.test(body)) {
    return { html: `<p class="help-lead">${escapeHelpHtml(body)}</p>`, next: [] };
  }

  const lines = body.split("\n");
  const sections = [];
  let current = null;

  function flush() {
    if (!current) return;
    sections.push(current);
    current = null;
  }

  for (const line of lines) {
    const heading = line.match(/^\s*#{1,3}\s+(.+)\s*$/);
    const boldHeading = line.match(/^\s*([^:]{2,40}):\s*(.*)$/);
    const bullet = line.match(/^\s*[-•]\s+(.+)\s*$/);
    const starBullet = line.match(/^\s*\*\s+(.+)\s*$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)\s*$/);
    if (heading) {
      flush();
      current = {
        title: cleanHelpMarkdown(heading[1].trim()),
        paras: [],
        bullets: [],
        tip: false,
      };
      const t = current.title.toLowerCase();
      current.tip = t === "tip" || t.startsWith("tip ");
      continue;
    }
    // "Agent roles: explanation" on one line → section title + body
    if (
      boldHeading &&
      !bullet &&
      !starBullet &&
      !numbered &&
      boldHeading[1].length <= 40 &&
      /^(what|agent|feature|interpreting|product|out of scope|using|how|forecast|performance|market|financial|on your desk)/i.test(
        boldHeading[1]
      )
    ) {
      flush();
      current = {
        title: cleanHelpMarkdown(boldHeading[1].trim()),
        paras: [],
        bullets: [],
        tip: false,
      };
      const rest = cleanHelpMarkdown(boldHeading[2].trim());
      if (rest) current.paras.push(rest);
      continue;
    }
    if (!current) {
      current = { title: "", paras: [], bullets: [], tip: false };
    }
    if (bullet || starBullet || numbered) {
      current.bullets.push(
        cleanHelpMarkdown((bullet || starBullet || numbered)[1].trim())
      );
      continue;
    }
    const trimmed = cleanHelpMarkdown(line.trim());
    if (!trimmed) continue;
    current.paras.push(trimmed);
  }
  flush();

  if (!sections.length) {
    return { html: `<p class="help-lead">${escapeHelpHtml(body)}</p>`, next };
  }

  const html = sections
    .map((section, index) => {
      const title = escapeHelpHtml(section.title);
      const paras = section.paras
        .map((p) => `<p>${escapeHelpHtml(p)}</p>`)
        .join("");
      const bullets = section.bullets.length
        ? `<ul>${section.bullets
            .map((b) => `<li>${escapeHelpHtml(b)}</li>`)
            .join("")}</ul>`
        : "";
      if (section.tip) {
        const tipBody = escapeHelpHtml(section.paras.join(" ")) || bullets;
        return `<p class="help-tip-line"><span>Tip</span>${tipBody}</p>`;
      }
      if (index === 0 && title) {
        return `<div class="help-block"><h4 class="help-heading">${title}</h4>${paras}${bullets}</div>`;
      }
      return `<div class="help-block">${
        title ? `<h5 class="help-subheading">${title}</h5>` : ""
      }${paras}${bullets}</div>`;
    })
    .join("");

  return { html: `<div class="help-rich">${html}</div>`, next };
}

function wireHelpChat() {
  const panel = $("helpChatPanel");
  const toggle = $("helpChatToggle");
  const closeBtn = $("helpChatClose");
  const form = $("helpChatForm");
  const input = $("helpChatInput");
  const messages = $("helpChatMessages");
  const suggestions = $("helpChatSuggestions");
  const sendBtn = $("helpChatSend");
  const stopBtn = $("helpChatStop");
  const emptyState = $("helpChatEmpty");
  if (!panel || !toggle || !form || !input || !messages) return;

  const history = [];
  let sending = false;
  let welcomed = false;
  let activeController = null;
  let activePending = null;

  const guideMarkSvg =
    '<svg class="help-diamond" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">' +
    '<g transform="rotate(-20 12 12)">' +
    '<path fill="#4285F4" d="M12 2.6L19.2 12H12z"/>' +
    '<path fill="#EA4335" d="M19.2 12L12 21.4V12z"/>' +
    '<path fill="#FBBC04" d="M12 21.4L4.8 12H12z"/>' +
    '<path fill="#34A853" d="M4.8 12L12 2.6V12z"/>' +
    "</g></svg>";

  function setEmptyVisible(show) {
    if (!emptyState) return;
    emptyState.classList.toggle("hidden", !show);
  }

  function setSuggestionsVisible(show) {
    if (!suggestions) return;
    if (show) suggestions.removeAttribute("hidden");
    else suggestions.setAttribute("hidden", "");
  }

  function setSendingUi(on) {
    sending = on;
    if (sendBtn) {
      sendBtn.disabled = on;
      sendBtn.classList.toggle("hidden", on);
    }
    if (stopBtn) {
      stopBtn.classList.toggle("hidden", !on);
    }
    input.disabled = false;
  }

  function stopGenerating() {
    if (activeController) {
      activeController.abort();
      activeController = null;
    }
  }

  function setOpen(open) {
    panel.classList.toggle("hidden", !open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) {
      if (!welcomed) {
        setSuggestionsVisible(true);
        welcomed = true;
      }
      input.focus();
    }
  }

  function appendFollowups(host, labels) {
    if (!labels || !labels.length) return;
    const wrap = document.createElement("div");
    wrap.className = "help-followups";
    for (const label of labels) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "help-suggest";
      btn.textContent = label;
      btn.addEventListener("click", () => askQuestion(label));
      wrap.appendChild(btn);
    }
    host.appendChild(wrap);
  }

  function scrollToRowStart(row) {
    if (!messages || !row) return;
    // Keep the start of this reply in view — do not jump to the end.
    const rowTop = row.offsetTop - messages.offsetTop;
    messages.scrollTop = Math.max(0, rowTop - 8);
  }

  function appendBubble(role, text, { pending = false, oos = false } = {}) {
    const row = document.createElement("div");
    row.className = `help-chat-row ${role === "user" ? "user" : "bot"}`;

    if (role !== "user") {
      const avatar = document.createElement("span");
      avatar.className = "help-chat-avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.innerHTML = guideMarkSvg;
      row.appendChild(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = `help-chat-bubble ${role === "user" ? "user" : "bot"}`;
    if (pending) {
      bubble.classList.add("pending");
      bubble.innerHTML =
        '<span class="help-typing" aria-hidden="true"><span></span><span></span><span></span></span>' +
        '<span class="help-typing-label">Generating…</span>';
    } else if (role === "user") {
      bubble.textContent = text;
    } else {
      const rendered = renderHelpRichHtml(text);
      bubble.innerHTML = rendered.html || escapeHelpHtml(text);
      if (oos || /^out of scope/i.test(String(text || ""))) {
        bubble.classList.add("oos");
      }
      appendFollowups(bubble, rendered.next);
    }
    row.appendChild(bubble);
    messages.appendChild(row);
    return { row, bubble };
  }

  async function askQuestion(question) {
    const q = (question || "").trim();
    if (!q || sending) return;
    setSendingUi(true);
    input.value = "";
    setSuggestionsVisible(false);
    setEmptyVisible(false);
    appendBubble("user", q);
    const { row, bubble: pending } = appendBubble("bot", "", { pending: true });
    activePending = { row, bubble: pending };
    scrollToRowStart(row);

    const controller = new AbortController();
    activeController = controller;
    const timeoutId = window.setTimeout(() => controller.abort("timeout"), 90000);

    try {
      const response = await fetch(`${API_BASE}/help-chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: q,
          history: history.slice(-6),
        }),
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string"
            ? payload.detail
            : `HTTP ${response.status}`
        );
      }
      const reply = payload.reply || "No reply.";
      pending.classList.remove("pending");
      const rendered = renderHelpRichHtml(reply);
      pending.innerHTML = rendered.html || escapeHelpHtml(reply);
      if (payload.out_of_scope || /^out of scope/i.test(reply)) {
        pending.classList.add("oos");
      }
      appendFollowups(pending, rendered.next);
      history.push({ role: "user", content: q });
      history.push({ role: "assistant", content: reply });
      scrollToRowStart(row);
    } catch (error) {
      if (error.name === "AbortError") {
        const timedOut = controller.signal.reason === "timeout";
        if (timedOut) {
          pending.classList.remove("pending");
          pending.innerHTML = "";
          pending.textContent = "Timed out waiting for Guide. Try again.";
          scrollToRowStart(row);
        } else if (activePending && activePending.row && activePending.row.parentNode) {
          activePending.row.remove();
        }
      } else {
        pending.classList.remove("pending");
        pending.innerHTML = "";
        pending.textContent = String(error.message || error);
        scrollToRowStart(row);
      }
    } finally {
      window.clearTimeout(timeoutId);
      activeController = null;
      activePending = null;
      setSendingUi(false);
      input.focus();
    }
  }

  toggle.addEventListener("click", () => {
    setOpen(panel.classList.contains("hidden"));
  });
  if (closeBtn) {
    closeBtn.addEventListener("click", () => setOpen(false));
  }
  if (stopBtn) {
    stopBtn.addEventListener("click", () => stopGenerating());
  }
  document.querySelectorAll("[data-open-help]").forEach((btn) => {
    btn.addEventListener("click", () => setOpen(true));
  });
  if (suggestions) {
    suggestions.addEventListener("click", (event) => {
      const btn = event.target.closest("[data-help-q]");
      if (!btn) return;
      setOpen(true);
      askQuestion(btn.dataset.helpQ);
    });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion(input.value);
  });
}

setTheme(localStorage.getItem("sip-theme") || "light");
wireHome();
wireHelpChat();
wireMetricChips();
checkReady();
if (sessionRunCount()) {
  window._lastAnalyzeTicker = sessionStore.order[0];
  lastAnalyzePayload = getSessionRun(sessionStore.order[0]);
  updateSessionResultsChrome(window._lastAnalyzeTicker);
}

let bootTries = 0;
function bootCharts() {
  if (typeof Plotly === "undefined") {
    bootTries += 1;
    if (bootTries < 80) {
      setTimeout(bootCharts, 50);
      return;
    }
    const home = $("homeDemoChart");
    if (home) {
      home.innerHTML =
        "<p class='muted' style='padding:12px'>Chart library did not load. Hard-refresh Chrome (Ctrl+Shift+R) or check network blocking of /vendor/plotly.min.js.</p>";
    }
    return;
  }
  renderHomeDemoChart();
}
bootCharts();
