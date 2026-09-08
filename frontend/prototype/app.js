/* Hardcoded demo payloads for UI review only — no API. */

const SAMPLES = {
  AAPL: {
    trend: "SIDEWAYS",
    news: "MIXED",
    confidence: "Low",
    guardrails: "A1+A2 PASS",
    model: "PERSISTENCE",
    move: "+0.00%",
    lastClose: 319.97,
    lastDate: "2026-09-04",
    history: [
      { date: "2026-08-25", value: 312.4 },
      { date: "2026-08-26", value: 314.1 },
      { date: "2026-08-27", value: 315.8 },
      { date: "2026-08-28", value: 317.2 },
      { date: "2026-08-29", value: 318.5 },
      { date: "2026-09-02", value: 319.1 },
      { date: "2026-09-03", value: 319.6 },
      { date: "2026-09-04", value: 319.97 },
    ],
    forecast: [
      { date: "2026-09-07", value: 319.97 },
      { date: "2026-09-08", value: 319.97 },
      { date: "2026-09-09", value: 319.97 },
      { date: "2026-09-10", value: 319.97 },
      { date: "2026-09-11", value: 319.97 },
    ],
    agent1:
      "Trend: SIDEWAYS\nRange: 319.9700 – 319.9700\nCaution: Persistence baseline; flat five-session path.",
    agent2:
      "Sentiment: MIXED\nDrivers:\n- App store fee pressure in the UK\n- iPhone margin pressure from memory costs\n- Foldable competition noted in coverage\nCaveat: Snapshot of recent headlines only.",
    sourcesMeta: "Provider · yahoo · 3 articles",
    sources: [
      {
        title: "Spotify-backed coalition urges CMA to speed up curbs on app store fees",
        meta: "2026-09-08 · Sky News",
        url: "#",
      },
      {
        title: "Apple's $54 Billion iPhone Engine Hits an AI Memory Squeeze",
        meta: "2026-09-07 · GuruFocus",
        url: "#",
      },
      {
        title: "Apple's $320 Stock Faces Huawei's 68% Foldable Fortress",
        meta: "2026-09-07 · GuruFocus",
        url: "#",
      },
    ],
  },
  MSFT: {
    trend: "SIDEWAYS",
    news: "MIXED",
    confidence: "Low",
    guardrails: "A1+A2 PASS",
    model: "PERSISTENCE",
    move: "+0.00%",
    lastClose: 499.7,
    lastDate: "2026-09-04",
    history: [
      { date: "2026-08-25", value: 492.1 },
      { date: "2026-08-26", value: 494.0 },
      { date: "2026-08-27", value: 495.5 },
      { date: "2026-08-28", value: 497.2 },
      { date: "2026-08-29", value: 498.0 },
      { date: "2026-09-02", value: 498.8 },
      { date: "2026-09-03", value: 499.2 },
      { date: "2026-09-04", value: 499.7 },
    ],
    forecast: [
      { date: "2026-09-07", value: 499.7 },
      { date: "2026-09-08", value: 499.7 },
      { date: "2026-09-09", value: 499.7 },
      { date: "2026-09-10", value: 499.7 },
      { date: "2026-09-11", value: 499.7 },
    ],
    agent1:
      "Trend: SIDEWAYS\nRange: 499.7000 – 499.7000\nCaution: Persistence baseline; unchanged path.",
    agent2:
      "Sentiment: MIXED\nDrivers:\n- OpenAI agent misuse acknowledgment\n- AI boom narrative with Nvidia\n- Next-gen Xbox cost concerns\nCaveat: Limited Yahoo sample.",
    sourcesMeta: "Provider · yahoo · 3 articles",
    sources: [
      {
        title: "Microsoft-Backed OpenAI Acknowledges AI Agents Misused External Wikis",
        meta: "2026-09-08 · MT Newswires",
        url: "#",
      },
      {
        title: "Nvidia, Microsoft at Center of $7 Trillion AI Boom",
        meta: "2026-09-07 · GuruFocus",
        url: "#",
      },
      {
        title: "Microsoft Has a $1,000 Problem With Its Next Xbox",
        meta: "2026-09-07 · GuruFocus",
        url: "#",
      },
    ],
  },
  NVDA: {
    trend: "SIDEWAYS",
    news: "UNAVAILABLE",
    confidence: "Low",
    guardrails: "A1+A2 PASS",
    model: "PERSISTENCE",
    move: "+0.00%",
    lastClose: 230.36,
    lastDate: "2026-09-04",
    history: [
      { date: "2026-08-25", value: 226.2 },
      { date: "2026-08-26", value: 227.5 },
      { date: "2026-08-27", value: 228.1 },
      { date: "2026-08-28", value: 229.0 },
      { date: "2026-08-29", value: 229.4 },
      { date: "2026-09-02", value: 229.9 },
      { date: "2026-09-03", value: 230.1 },
      { date: "2026-09-04", value: 230.36 },
    ],
    forecast: [
      { date: "2026-09-07", value: 230.36 },
      { date: "2026-09-08", value: 230.36 },
      { date: "2026-09-09", value: 230.36 },
      { date: "2026-09-10", value: 230.36 },
      { date: "2026-09-11", value: 230.36 },
    ],
    agent1:
      "Trend: SIDEWAYS\nRange: 230.3600 – 230.3600\nCaution: Persistence baseline.",
    agent2:
      "Sentiment: UNAVAILABLE\nDrivers: No ticker-relevant headlines after filtering.\nCaveat: Do not invent news.",
    sourcesMeta: "Provider · yahoo · 0 ticker-relevant articles",
    sources: [],
  },
  TSLA: {
    trend: "BULLISH",
    news: "MIXED",
    confidence: "Medium",
    guardrails: "A1+A2 PASS",
    model: "PARENT",
    move: "+0.77%",
    lastClose: 354.08,
    lastDate: "2026-09-04",
    history: [
      { date: "2026-08-25", value: 342.0 },
      { date: "2026-08-26", value: 345.2 },
      { date: "2026-08-27", value: 348.1 },
      { date: "2026-08-28", value: 350.4 },
      { date: "2026-08-29", value: 351.8 },
      { date: "2026-09-02", value: 352.5 },
      { date: "2026-09-03", value: 353.2 },
      { date: "2026-09-04", value: 354.08 },
    ],
    forecast: [
      { date: "2026-09-07", value: 355.58 },
      { date: "2026-09-08", value: 356.21 },
      { date: "2026-09-09", value: 356.9 },
      { date: "2026-09-10", value: 357.45 },
      { date: "2026-09-11", value: 358.14 },
    ],
    agent1:
      "Trend: BULLISH\nRange: 355.5771 – 358.1395\nCaution: Parent model path; treat as uncertain.",
    agent2:
      "Sentiment: MIXED\nDrivers:\n- Robotaxi commentary\n- European catalyst coverage\n- Macro tape also naming TSLA\nCaveat: Headline mix is uneven.",
    sourcesMeta: "Provider · yahoo · 3 articles",
    sources: [
      {
        title: "Fantastic News for Tesla Stock Investors!",
        meta: "2026-09-08 · Motley Fool",
        url: "#",
      },
      {
        title: "Ross Gerber Recommends Tesla Owners Let TSLA Take The Risk In Robotaxi Program",
        meta: "2026-09-08 · Benzinga",
        url: "#",
      },
      {
        title: "Tesla Stock May Have A European Catalyst Investors Are Missing",
        meta: "2026-09-08",
        url: "#",
      },
    ],
  },
};

let chart;

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

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("sip-prototype-theme", theme);
  const toggle = document.getElementById("themeToggle");
  toggle.textContent = theme === "light" ? "Dark" : "Light";
  if (chart) {
    const ink = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
    const grid = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();
    chart.options.scales.x.ticks.color = ink;
    chart.options.scales.y.ticks.color = ink;
    chart.options.scales.x.grid.color = grid;
    chart.options.scales.y.grid.color = grid;
    chart.data.datasets[0].borderColor = getComputedStyle(document.documentElement)
      .getPropertyValue("--chart-hist")
      .trim();
    chart.data.datasets[1].borderColor = getComputedStyle(document.documentElement)
      .getPropertyValue("--chart-fc")
      .trim();
    chart.data.datasets[1].backgroundColor = getComputedStyle(document.documentElement)
      .getPropertyValue("--chart-fc")
      .trim();
    chart.update();
  }
}

function renderChart(sample) {
  const histColor = getComputedStyle(document.documentElement)
    .getPropertyValue("--chart-hist")
    .trim();
  const fcColor = getComputedStyle(document.documentElement).getPropertyValue("--chart-fc").trim();
  const ink = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  const grid = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();

  const historyLabels = sample.history.map((point) => point.date);
  const historyValues = sample.history.map((point) => point.value);
  const forecastLabels = [
    sample.history[sample.history.length - 1].date,
    ...sample.forecast.map((point) => point.date),
  ];
  const forecastValues = [
    sample.history[sample.history.length - 1].value,
    ...sample.forecast.map((point) => point.value),
  ];

  const labels = [...new Set([...historyLabels, ...forecastLabels])];
  const histSeries = labels.map((label) => {
    const hit = sample.history.find((point) => point.date === label);
    return hit ? hit.value : null;
  });
  const fcSeries = labels.map((label) => {
    const idx = forecastLabels.indexOf(label);
    return idx >= 0 ? forecastValues[idx] : null;
  });

  const ctx = document.getElementById("forecastChart");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Observed",
          data: histSeries,
          borderColor: histColor,
          borderWidth: 2,
          pointRadius: 0,
          spanGaps: false,
          tension: 0.15,
        },
        {
          label: "Forecast",
          data: fcSeries,
          borderColor: fcColor,
          backgroundColor: fcColor,
          borderWidth: 2,
          pointRadius: 3,
          spanGaps: false,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: ink, boxWidth: 12, font: { family: "Roboto", size: 12 } },
        },
        tooltip: {
          callbacks: {
            label: (item) => (item.parsed.y == null ? "" : money(item.parsed.y)),
          },
        },
      },
      scales: {
        x: {
          ticks: { color: ink, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
          grid: { color: grid },
        },
        y: {
          ticks: {
            color: ink,
            callback: (value) => `$${value}`,
          },
          grid: { color: grid },
        },
      },
    },
  });
}

function render(ticker) {
  const sample = SAMPLES[ticker];
  if (!sample) return;

  const trendEl = document.getElementById("metricTrend");
  const newsEl = document.getElementById("metricNews");
  trendEl.textContent = sample.trend;
  newsEl.textContent = sample.news;
  trendEl.className = `metric-value ${toneClass(sample.trend)}`;
  newsEl.className = `metric-value ${toneClass(sample.news)}`;
  document.getElementById("metricConfidence").textContent = sample.confidence;
  document.getElementById("metricGuardrails").textContent = sample.guardrails;

  document.getElementById("forecastModel").textContent = `Model · ${sample.model}`;
  document.getElementById("forecastMove").textContent = `Move · ${sample.move}`;
  document.getElementById("forecastClose").textContent = `Last close · ${money(sample.lastClose)}`;
  document.getElementById("forecastDate").textContent = `As of · ${sample.lastDate}`;

  document.getElementById("agent1Text").textContent = sample.agent1;
  document.getElementById("agent2Text").textContent = sample.agent2;
  document.getElementById("sourcesMeta").textContent = sample.sourcesMeta;

  const list = document.getElementById("sourcesList");
  list.innerHTML = "";
  if (!sample.sources.length) {
    const empty = document.createElement("li");
    empty.textContent = "No sources";
    empty.style.color = "var(--muted)";
    list.appendChild(empty);
  } else {
    for (const source of sample.sources) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = source.url;
      link.textContent = source.title;
      const meta = document.createElement("span");
      meta.className = "source-meta";
      meta.textContent = source.meta;
      item.append(link, meta);
      list.appendChild(item);
    }
  }

  const body = document.getElementById("forecastTableBody");
  body.innerHTML = "";
  for (const point of sample.forecast) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${point.date}</td><td>${money(point.value)}</td>`;
    body.appendChild(row);
  }

  renderChart(sample);
}

document.getElementById("demoForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const raw = document.getElementById("tickerInput").value.trim().toUpperCase();
  const ticker = SAMPLES[raw] ? raw : "AAPL";
  document.getElementById("tickerInput").value = ticker;
  render(ticker);
});

document.getElementById("themeToggle").addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  setTheme(current === "light" ? "dark" : "light");
});

const saved = localStorage.getItem("sip-prototype-theme") || "light";
setTheme(saved);
render("AAPL");
