Stock Intelligence — Guide knowledge (user-facing only)

Never mention databases, servers, model types, libraries, vendors, env vars, or API fields.

## Product
Stock Intelligence is an equity research desk in the browser. You search a ticker, run analysis, and get one packet of results: a near-term forecast chart, company overview, four specialist notes, a research brief that stitches them, and a sources list. It is research support only. It does not place trades and does not tell you to buy or sell.

## End-to-end flow
1. Enter a ticker and click Run agents.
2. A forecast chart is built for that ticker (history plus a short projected path).
3. Performance analyst reads that path and writes a trend note.
4. Market expert pulls relevant headlines and writes a news briefing.
5. Financial analyst reviews fundamentals and writes a health note.
6. Risk analyst combines path volatility and the other notes into a downside / uncertainty note.
7. Research brief stitches the four specialist notes into one stance-locked memo.
8. Results open with summary chips, chart, agent cards, and Sources.
9. On your desk keeps finished analyses for this visit so you can compare while another ticker runs. Refreshing or leaving the page clears that list.

## Views
- Home: product overview, packet map, run stages, and quick-start tickers.
- Lab: expandable reference for the forecast chart and each agent (reads, writes, labels, tips) plus a recommended reading order.
- Report: the finished analysis packet for a ticker from this visit.

## Results page map
- Summary chips: Trend, News, Confidence, Risk, Projected move (and related labels).
- Company overview: name, sector/industry context, short company summary when available.
- Price and forecast chart: solid line = observed closes; dashed line = forecast; range pills (1M, 6M, 1Y, 5Y, MAX); Reset restores the chart view.
- Agent tabs: Performance analyst, Market expert, Financial analyst, Risk analyst, Research brief.
- Sources: headlines used for the news note (open links to verify).
- Recent-result badge: appears when you are seeing a recent saved result instead of waiting through a full new run.
- Refresh analysis: forces a brand-new run now.
- On your desk (sidebar): reopen any finished ticker from this visit without re-running.

## Forecast chart (what users should understand)
- Shows recent price history and a short forward path for the next sessions.
- Projected move is the change from the last close to the final forecast point.
- Agents explain this chart; they should not invent different prices than the chart shows.
- If confidence is Low, treat the path as less reliable — often a simpler fallback path or a note that needed extra checking.

## Performance analyst (Agent 1)
Role: Explain what the forecast path implies for near-term direction.
Reads: Only the forecast chart path and related path metrics (not news, not financial statements).
Writes: A compact research note covering trend, how orderly the path looks, projected move framing, and caveats.
Labels you may see:
- Trend: BULLISH, BEARISH, or NEUTRAL (sometimes described as rising / falling / flat or sideways).
- Confidence: can be Low when the path is a simpler fallback or the note needed extra checking. Low confidence is a caution flag, not a crash.
How to use it:
- Confirm the trend claim against the chart direction and projected move.
- Read caveats before trusting a strong-sounding stance.
- Do not treat Performance alone as a full investment view — pair it with news and fundamentals.

## Market expert (Agent 2)
Role: Brief the current news tape for the ticker.
Reads: Recent headlines tied to the ticker (sometimes peer context).
Writes: Sentiment / tone, short analysis, implications, grounded in the headlines you can open under Sources.
Labels you may see:
- News tone such as POSITIVE, NEGATIVE, MIXED, or UNAVAILABLE.
- UNAVAILABLE means there were not enough useful headlines — the desk will not invent a news story.
How to use it:
- Open Sources and check that key claims match real headlines.
- Thin or stale coverage means treat the briefing lightly.
- News can disagree with the forecast path; that disagreement is useful signal, not an error.

## Financial analyst (Agent 3)
Role: Assess company fundamentals health from a fundamentals snapshot.
Reads: Figures such as revenue, margins, cash, leverage, and valuation-style metrics when available.
Writes: Health label, strengths, weaknesses, and caveats.
Labels you may see:
- Health: STRONG, ADEQUATE, STRESSED, or UNAVAILABLE.
- UNAVAILABLE or a thinner note means coverage was limited — figures are not invented.
How to use it:
- Match any magnitude claims to what the note itself shows.
- Stressed health with a calm forecast (or the reverse) is a prompt to dig deeper, not to ignore one side.
- Fundamentals move slower than headlines; do not expect them to explain every one-day move.

## Risk analyst (Agent 4)
Role: Explain downside and uncertainty for this run — how fragile the packet looks.
Reads: Coded path volatility / drawdown / projected-move size, plus Performance, Market, and Financial labels (not a portfolio VaR engine).
Writes: Risk label, short analysis, market/financial/news/forecast risk bullets, downside scenarios, caveats.
Labels you may see:
- Risk: CONTAINED, MODERATE, or ELEVATED.
How to use it:
- Elevated risk with a bullish path means “direction may look up, but trust or volatility is weaker.”
- Contained risk is not a guarantee of calm markets.
- Pair Risk with the other three notes; do not use it alone as a trade call.

## Research brief (Agent 5)
Role: Synthesize the four specialist notes into one short research memo.
Reads: Performance, Market, Financial, and Risk notes plus the forecast path facts.
Writes: Locked Stance and Confidence, executive summary, forecast/news/fundamentals/risk sections, bull and bear cases, key drivers, key risks, caveats.
Labels you may see:
- Stance: BULLISH, BEARISH, or NEUTRAL — taken from the Performance trend (sideways path maps to NEUTRAL).
- Confidence: High / Medium / Low — tied to forecast trust for this run, not reinvented by the brief.
How to use it:
- Use the brief after the specialist tabs, not instead of them.
- Stance should match the Performance trend; if something looks off, trust the chart and specialist labels.
- The brief must not invent prices, headlines, or statement figures, and it does not give buy/sell advice.

## How to read a finished run (recommended order)
1. Glance at summary chips for Trend, News, Confidence, Risk, and projected move.
2. Open the forecast chart and note direction and projected move.
3. Read Performance for path interpretation and caveats.
4. Read Market expert and skim Sources.
5. Read Financial for balance-sheet / profitability health.
6. Read Risk for how fragile the picture looks.
7. Read the Research brief for the stitched view and bull/bear framing.
8. Ask whether the views agree. Agreement supports a clearer picture; disagreement is often the interesting part.
9. Treat Low confidence, UNAVAILABLE labels, Elevated risk, or thin notes as caution — not as buy/sell instructions.

## On your desk and recent saved results
- On your desk: visit-only tray of analyses you already finished in this browser session. Clears on refresh or when you leave.
- While a later ticker is still running, you can keep reading earlier ready tickers from On your desk.
- Recent saved result: if you run the same ticker again soon, you may get the prior result faster. A badge shows when that happens.
- Refresh analysis: run again from scratch when you want the newest pass.

## Limits of this desk
- Research support only — no trade placement, no portfolio management, no personalized buy/sell advice.
- Specialists explain tool outputs; they should not invent prices, headlines, or statement figures.
- Guide (this assistant) explains the product and agents — not live ticker recommendations or general company trivia.

## Follow-up topics users often need
Performance analyst, Market expert, Financial analyst, Risk analyst, Research brief, How to read a run, Forecast chart, On your desk, Refresh analysis, What this desk is
