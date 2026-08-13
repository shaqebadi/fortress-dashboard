# FORTRESS — live cloud dashboard

The live paper record of **FORTRESS**, a halal, long-only, unlevered systematic US equity
strategy: the compliant halal-Nasdaq core with a frozen dual-trigger risk clamp on top
("beats the S&P with a shield, not a rocket"). This page rebuilds itself **once a day on
GitHub's servers** and is served via GitHub Pages.

Sibling product: the Deen Capital dashboard (QQQ-beater, same operator). FORTRESS is a
distinct product with a shallower-drawdown mandate; the two records are kept separate on
purpose.

## Strategy (all parameters FROZEN — audited backtest values, never refit)

- **Core**: top-100 halal Nasdaq book (deployed top-60 form), cap-weighted, 20% single-name
  cap, AAOIFI screen + 130-name exclusion list, reconstituted monthly. Identical to the
  Deen Capital production CORE sleeve; synced after each monthly formation.
- **Overlay** (2 parameters, σ\*=0.20 / D\*=0.12): equity fraction
  `f = clip(min(σ*/max(σ20,σ60), drawdown-brake), 0.25, 1)`, 5pp drift-band execution,
  t+2 application. De-risked capital: 50% GLD, 50% cash at 0% (no interest, halal).
- Daily portfolio return: `r = f·r_core + 0.5(1−f)·r_GLD`.

## How it updates

A scheduled GitHub Action (`.github/workflows/update.yml`) runs every weekday ~2h after
the US close: fetches adjusted closes (yfinance), extends the core return series
(buy-and-hold intra-book drift), recomputes the frozen clamp on the full series, marks
NAV + benchmarks (SPY, QQQ, SPUS, HLAL), regenerates `index.html`, commits. A failed
fetch never blanks the page.

## Monthly core sync (manual, ~once a month)

After the local Deen Capital monthly production run, refresh the core holdings here:
update `core_weights` in `data/paper_state.json` and `data/target_book.json` from the new
production CORE sleeve, and commit. The clamp state carries over untouched (it depends
only on the core return series, which is continuous across reconstitutions).

## Files

| Path | Purpose |
|---|---|
| `build_dashboard.py` | fetch + mark + clamp + regenerate `index.html` (`--offline` renders only) |
| `data/core_returns.csv` | the core's daily return series: research seed → live record (`source` column) |
| `data/paper_track.csv` | daily NAV record: FORTRESS + benchmarks + executed f and weights |
| `data/paper_state.json` | drifting core weights, NAV, benchmark levels, provenance |
| `data/target_book.json` | current holdings + shield state (for display) |
| `data/tear_sheet_stats.json` | static 2010–2025 backtest stats (delisting-safe re-derivation) |

## Disclosures

Paper NAV is **simulated** (adjusted closing prices, not broker fills), gross of costs.
The 2010–2025 figures are **hypothetical backtested** results; the 2020–2025 window is
**semi-out-of-sample** — this live record is the strategy's first true forward test.
The clamp's pre-2026 estimator seed splices the audited research core (through 2025) with
a current-constituent bridge for 2026 (validated: 0.983 daily correlation vs the true core
over 2025); the seed only initializes the vol/drawdown estimators and never enters the
NAV record. Past performance does not predict future results. For the operator's use —
not an offer or solicitation.
