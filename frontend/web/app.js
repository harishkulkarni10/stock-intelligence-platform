const API_BASE = (window.SIP_API_URL || window.location.origin).replace(/\/$/, "");
const TIMEOUT_MS = 600000;

let threadId = crypto.randomUUID();
let homeChartReady = false;

const PIPE_COPY = [
  {
    title: "Forecast (code)",
    body: "Loads the evaluation winner for the ticker: promoted child LSTM, parent, or persistence. Prices come from models — not the LLM.",
  },
  {
    title: "Agent 1 · Performance",
    body: "Reads only the forecast path and returns Trend, Range, and Caution. Guardrails block invented prices.",
  },
  {
    title: "News tool",
    body: "Fetches headlines, keeps ticker-relevant ones, drops unrelated market blurbs.",
  },
  {
    title: "Agent 2 · Market expert",
    body: "Summarizes filtered news as Sentiment, Drivers, and Caveat. Missing news → UNAVAILABLE.",
  },
];

const FEATURE_COPY = {
  model:
    "If a child model lost to persistence in training, you will see a flat path and Low confidence. That is intentional honesty, not a UI bug.",
  a1: "Agent 1 never invents upside. SIDEWAYS on a flat persistence path is the correct call.",
  a2: "Agent 2 only uses headlines that mention the ticker or company aliases after filtering.",
  guards: "Failed LLM drafts are retried, then repaired or replaced with a deterministic fallback.",
};

function $(id) {
  return document.getElementById(id);
}

function toneClass(label) {
  const value = String(label || "").toUpperCase();
  if (["BULLISH", "POSITIVE"].includes(value)) return "positive";
  if (["BEARISH", "NEGATIVE"].includes(value)) return "negative";
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
  $("themeIconSun").classList.toggle("hidden", theme !== "light");
  $("themeIconMoon").classList.toggle("hidden", theme !== "dark");
  $("themeToggle").setAttribute(
    "aria-label",
    theme === "light" ? "Switch to dark theme" : "Switch to light theme"
  );
  if (homeChartReady) renderHomeDemoChart();
  if (!$("results").classList.contains("hidden") && window._lastSeries) {
    renderForecastChart(window._lastSeries.history, window._lastSeries.forecast);
  }
}

function normalizeTicker(raw) {
  return String(raw || "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9.^=-]/g, "");
}

function historyPoints(predictions) {
  return (predictions.history || [])
    .map((row) => ({
      date: String(row.date || "").slice(0, 10),
      value: Number(row.close ?? row.value),
    }))
    .filter((row) => row.date && Number.isFinite(row.value));
}

function forecastPoints(predictions) {
  return (predictions.forecast || [])
    .map((row) => ({
      date: String(row.date || "").slice(0, 10),
      value: Number(row.value ?? row.close),
    }))
    .filter((row) => row.date && Number.isFinite(row.value));
}

function showHome() {
  $("home").classList.remove("hidden");
  $("results").classList.add("hidden");
  $("errorBanner").classList.add("hidden");
  $("statusLine").classList.add("hidden");
  $("tickerInput").value = "";
  renderHomeDemoChart();
}

function showResults() {
  $("home").classList.add("hidden");
  $("results").classList.remove("hidden");
}

function renderHomeDemoChart() {
  if (typeof Plotly === "undefined") return;
  const histX = ["Aug 25", "Aug 26", "Aug 27", "Aug 28", "Aug 29", "Sep 2", "Sep 3", "Sep 4"];
  const histY = [100, 101.2, 100.8, 102.1, 103.0, 102.4, 103.5, 104.0];
  const fcX = ["Sep 4", "Sep 5", "Sep 6", "Sep 7", "Sep 8", "Sep 9"];
  const fcY = [104.0, 104.4, 104.9, 105.2, 105.6, 106.0];
  Plotly.newPlot(
    "homeDemoChart",
    [
      {
        x: histX,
        y: histY,
        type: "scatter",
        mode: "lines",
        name: "Observed (demo)",
        line: { color: cssVar("--chart-hist"), width: 2 },
      },
      {
        x: fcX,
        y: fcY,
        type: "scatter",
        mode: "lines+markers",
        name: "Forecast (demo)",
        line: { color: cssVar("--chart-fc"), width: 2 },
        marker: { size: 6 },
      },
    ],
    plotLayout({ title: { text: "Sample observed → forecast path", font: { size: 13 } } }),
    plotConfig()
  );
  homeChartReady = true;
}

function renderForecastChart(history, forecast) {
  if (typeof Plotly === "undefined") return;
  window._lastSeries = { history, forecast };
  const bridge = history.length ? history[history.length - 1] : null;
  const fc = bridge ? [bridge, ...forecast] : forecast;
  Plotly.newPlot(
    "forecastChart",
    [
      {
        x: history.map((p) => p.date),
        y: history.map((p) => p.value),
        type: "scatter",
        mode: "lines",
        name: "Observed",
        hovertemplate: "%{x}<br>%{y:$.2f}<extra>Observed</extra>",
        line: { color: cssVar("--chart-hist"), width: 2 },
      },
      {
        x: fc.map((p) => p.date),
        y: fc.map((p) => p.value),
        type: "scatter",
        mode: "lines+markers",
        name: "Forecast",
        hovertemplate: "%{x}<br>%{y:$.2f}<extra>Forecast</extra>",
        line: { color: cssVar("--chart-fc"), width: 2 },
        marker: { size: 7 },
      },
    ],
    plotLayout(),
    plotConfig()
  );
}

function parseLabeledBlocks(text, labels) {
  const lines = String(text || "").split(/\r?\n/);
  const found = {};
  let current = null;
  for (const line of lines) {
    const match = labels.find((label) =>
      new RegExp(`^\\s*${label}\\s*:`, "i").test(line)
    );
    if (match) {
      current = match;
      found[current] = line.replace(new RegExp(`^\\s*${match}\\s*:\\s*`, "i"), "").trim();
      continue;
    }
    if (current && line.trim()) {
      found[current] = `${found[current]}\n${line}`.trim();
    }
  }
  return found;
}

function renderAgentCard(prefix, text, labels) {
  const parsed = parseLabeledBlocks(text, labels);
  const chips = $(`${prefix}Chips`);
  const sections = $(`${prefix}Sections`);
  const raw = $(`${prefix}Raw`);
  raw.textContent = text || "No response.";
  chips.innerHTML = "";
  sections.innerHTML = "";

  labels.forEach((label, index) => {
    const value = (parsed[label] || "").trim();
    if (!value) return;
    if (index === 0) {
      const chip = document.createElement("span");
      chip.className = `agent-chip ${toneClass(value)}`;
      chip.textContent = `${label}: ${value.split("\n")[0]}`;
      chips.appendChild(chip);
    }
    const details = document.createElement("details");
    details.className = "agent-block";
    details.open = index < 2;
    const firstLine = value.split("\n")[0];
    details.innerHTML = `
      <summary><span>${label}</span><span class="hint">${firstLine.slice(0, 48)}${
      firstLine.length > 48 ? "…" : ""
    }</span></summary>
      <div class="agent-block-body"></div>
    `;
    details.querySelector(".agent-block-body").textContent = value;
    sections.appendChild(details);
  });

  if (!sections.children.length) {
    const fallback = document.createElement("div");
    fallback.className = "agent-block-body";
    fallback.textContent = text || "No response.";
    sections.appendChild(fallback);
  }
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
  const perfOk = data.performance_guardrail_ok;
  const newsOk = data.news_guardrail_ok;
  let guardrails = "—";
  if (perfOk === true && newsOk === true) guardrails = "A1+A2 PASS";
  else if (perfOk === false || newsOk === false) guardrails = "CHECK";

  const trendEl = $("metricTrend");
  const newsEl = $("metricNews");
  trendEl.textContent = String(trend).toUpperCase();
  newsEl.textContent = String(news).toUpperCase();
  trendEl.className = `metric-value ${toneClass(trend)}`;
  newsEl.className = `metric-value ${toneClass(news)}`;
  $("metricConfidence").textContent = data.confidence || "—";
  $("metricGuardrails").textContent = guardrails;

  $("forecastMeta").innerHTML = `
    <span class="meta-pill">Model · ${String(predictions.model_source || "—").toUpperCase()}</span>
    <span class="meta-pill" title="Change from last close to the final forecasted session">
      Projected ${horizon}-session move · ${signedMove}
    </span>
    <span class="meta-pill">From last close ${
      lastClose != null ? money(lastClose) : "—"
    } on ${lastDate}</span>
    <span class="meta-pill">To ${
      terminal != null ? money(terminal) : "—"
    } on ${endDate}</span>
  `;

  renderAgentCard("agent1", data.performance_analysis || "", ["Trend", "Range", "Caution"]);
  renderAgentCard("agent2", data.news_summary || "", ["Sentiment", "Drivers", "Caveat"]);

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
    for (const article of articles.slice(0, 8)) {
      const item = document.createElement("li");
      const title = article.headline || "Untitled";
      const url = article.url || "";
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

  $("runMeta").textContent =
    `Mode · ${data.mode || "performance_news"} · ` +
    `${data.cached ? "cache hit" : "fresh run"} · research support only`;

  showResults();
  renderForecastChart(history, forecast);
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

async function analyze(ticker) {
  $("errorBanner").classList.add("hidden");
  $("statusLine").classList.remove("hidden");
  $("statusLine").textContent =
    `Running agents for ${ticker}… forecast → agent 1 → news → agent 2 (often ~1 min)`;
  $("analyzeBtn").disabled = true;
  $("home").classList.add("hidden");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, thread_id: threadId }),
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
    renderResults(payload);
    $("statusLine").textContent = payload.cached
      ? `${ticker} ready · cache hit`
      : `${ticker} ready`;
  } catch (error) {
    $("results").classList.add("hidden");
    $("home").classList.remove("hidden");
    $("errorBanner").classList.remove("hidden");
    $("errorBanner").textContent =
      error.name === "AbortError"
        ? "Timed out waiting for Ollama / news. Try again."
        : String(error.message || error);
    $("statusLine").classList.add("hidden");
    renderHomeDemoChart();
  } finally {
    clearTimeout(timer);
    $("analyzeBtn").disabled = false;
  }
}

function setPipelineStep(index) {
  document.querySelectorAll(".pipe-step").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.step) === index);
  });
  const item = PIPE_COPY[index];
  $("pipelineDetail").innerHTML = `<strong>${item.title}</strong><br>${item.body}`;
}

function wireHome() {
  setPipelineStep(0);
  $("featureDetail").textContent = "Select a capability card to learn more.";

  document.querySelectorAll(".pipe-step").forEach((el) => {
    el.addEventListener("click", () => setPipelineStep(Number(el.dataset.step)));
  });

  document.querySelectorAll(".feature-card").forEach((el) => {
    el.addEventListener("click", () => {
      document.querySelectorAll(".feature-card").forEach((card) => card.classList.remove("active"));
      el.classList.add("active");
      $("featureDetail").textContent = FEATURE_COPY[el.dataset.feature] || "";
    });
  });

  $("quickTickers").addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-ticker]");
    if (!btn) return;
    const ticker = btn.dataset.ticker;
    $("tickerInput").value = ticker;
    analyze(ticker);
  });
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
  showHome();
});

$("chartReset").addEventListener("click", () => {
  if (window._lastSeries) {
    renderForecastChart(window._lastSeries.history, window._lastSeries.forecast);
  }
});

document.addEventListener("click", async (event) => {
  const btn = event.target.closest("[data-copy]");
  if (!btn) return;
  const target = $(btn.dataset.copy);
  if (!target) return;
  try {
    await navigator.clipboard.writeText(target.textContent || "");
    const prev = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => {
      btn.textContent = prev;
    }, 1200);
  } catch {
    btn.textContent = "Copy failed";
  }
});

setTheme(localStorage.getItem("sip-theme") || "light");
wireHome();
checkReady();

function bootCharts() {
  if (typeof Plotly === "undefined") {
    setTimeout(bootCharts, 50);
    return;
  }
  renderHomeDemoChart();
}
bootCharts();
