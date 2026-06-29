# Institutional Investment Research Platform

This project is a Python research platform that combines equity valuation,
fixed-income curve modelling, GenAI-assisted research narration, calibration
tracking, real-market data adapters, and walk-forward backtesting.

I built it around one rule: the system can use language models to explain
research, but every financial number must come from deterministic,
auditable Python code. That makes the project easy to defend in an interview:
the valuation math, rate-risk model, backtest windows, and calibration metrics
can all be opened, tested, and explained line by line.

## What It Does

- Runs a CFA-style valuation stack: FCFF, FCFE, WACC, multi-stage DCF,
  EV/EBITDA and P/E comps, residual income, bear/base/bull scenarios, and
  margin of safety.
- Fits a Nelson-Siegel Treasury curve and uses a Kalman filter to track
  level, slope, and curvature through time.
- Flags equities with duration-like sensitivity to 10Y Treasury yield changes.
- Combines valuation and rate-risk signals through an investment committee
  orchestrator.
- Logs recommendations and confidence scores, then evaluates calibration with
  Brier score, Expected Calibration Error, realized accuracy, precision/recall,
  and decision entropy.
- Runs expanding-window walk-forward backtests to avoid look-ahead bias.
- Supports synthetic data by default and real-data mode through yfinance and
  FRED Treasury yield curves.

## Quick Start

```bash
pip install -r requirements.txt

# Full offline demo using synthetic market and yield-curve data
python demo.py

# Real-data mode for a US-listed ticker
python demo.py --real --ticker AAPL --peers MSFT,GOOGL,META,AMZN

# Test suite
python -m pytest tests/ -v
```

If real-data mode cannot reach a provider or a ticker is missing required
fundamentals, the demo prints the reason and falls back to synthetic data so
the rest of the pipeline still runs.

## Architecture

```text
quant-research-platform/
|-- core/
|   |-- valuation.py        # FCFF/FCFE/WACC/DCF/comps/residual income/scenarios
|   |-- fixed_income.py     # Nelson-Siegel, Kalman filter, duration, convexity, KRD
|   |-- calibration.py      # Brier score, ECE, precision/recall, decision entropy
|   `-- metrics.py          # Sharpe, Sortino, Calmar, IR, alpha/beta, drawdown
|-- agents/
|   |-- llm_client.py       # Provider-agnostic research narration interface
|   |-- valuation_agent.py  # Wraps the valuation core and narrates the output
|   |-- fixed_income_agent.py
|   `-- orchestrator.py     # Investment committee and calibration logging
|-- backtest/
|   `-- walk_forward.py     # Expanding/rolling walk-forward engine
|-- data/
|   |-- synthetic_market.py # Offline market/yield-curve generator
|   `-- real_market.py      # yfinance prices/fundamentals + FRED yield curves
|-- tests/                  # Unit and integration tests
`-- demo.py                 # Single-command end-to-end demo
```

## Design Choices

**1. Financial math first, narration second.**  
`core/valuation.py`, `core/fixed_income.py`, `core/metrics.py`, and
`core/calibration.py` contain the auditable calculations. The narration layer
only receives already-computed outputs and turns them into analyst-style text.

**2. Confidence is model-derived, not guessed.**  
The valuation confidence score is based on agreement across DCF, comps, and
residual income. The fixed-income confidence score is based on Kalman filter
state uncertainty. The calibration framework then checks whether those
confidence scores match realized outcomes.

**3. Backtesting uses forward-only windows.**  
`backtest/walk_forward.py` trains on one window, predicts the next unseen
window, then repeats. This is intentionally more realistic than a single static
train/test split.

**4. Real data and offline reproducibility both matter.**  
The project can run fully offline with synthetic data, which makes testing and
presentation reliable. It can also pull delayed equity prices/fundamentals from
yfinance and Treasury curves from FRED when internet access is available.

## GenAI Integration

The platform exposes a small `LLMClient` interface. The default
`MockLLMClient` is deterministic so the test suite and demo remain reproducible.
For live narration, `AnthropicLLMClient` is already implemented:

```bash
pip install anthropic
set ANTHROPIC_API_KEY=your_key_here
```

Then construct the agents with `AnthropicLLMClient()` instead of
`MockLLMClient()`. No valuation, risk, calibration, or backtesting code needs
to change.

## Verification

The test suite covers:

| Suite | Focus |
|---|---|
| `test_valuation.py` | WACC, FCFF, DCF monotonicity, comps, residual income, scenarios, recommendation logic |
| `test_fixed_income.py` | Nelson-Siegel parameter recovery, Kalman tracking, bond pricing, duration/convexity, rate sensitivity |
| `test_calibration_and_metrics.py` | Brier score, ECE, overconfidence flagging, drawdown, Calmar, alpha/beta, performance reporting |
| `test_end_to_end.py` | Synthetic data, both research agents, committee decisioning, walk-forward backtest, calibration logging |
| `test_real_market.py` | Mocked yfinance/FRED normalization for prices, statements, peer multiples, and Treasury curves |

One useful regression test is `test_residual_income_is_per_share_not_total`.
It guards against a real bug found during development: residual income was
initially computed as a *total equity value* while DCF and comps are
*per-share* values, so the two were silently off by a factor equal to shares
outstanding. That mismatch was corrupting the valuation agent's cross-method
confidence score (it read as strong disagreement between methods purely
because of a units error, not because the methods actually disagreed). It
was caught by comparing intrinsic-value outputs across methods, fixed by
dividing by `shares_outstanding`, and pinned down with this regression test
so it cannot silently reappear.

## Interview Walkthrough

1. Start with `demo.py` and show the full pipeline running in one command.
2. Open `core/valuation.py` and explain how FCFF, WACC, DCF, comps, residual
   income, scenarios, and margin of safety are calculated.
3. Open `core/fixed_income.py` and explain the Nelson-Siegel factors plus the
   Kalman filter state update.
4. Open `agents/orchestrator.py` and show how a BUY can be downgraded to HOLD
   by the fixed-income risk overlay.
5. Open `core/calibration.py` and explain how confidence is checked against
   realized outcomes.
6. Open `backtest/walk_forward.py` and show why `test_start == train_end`
   prevents look-ahead bias.

## Suggested CV Bullet

Built an institutional investment research platform in Python combining a
CFA-style equity valuation engine (FCFF/FCFE/WACC/DCF/comps/residual income),
a Kalman-filtered Nelson-Siegel fixed-income risk overlay, GenAI-assisted
research narration, yfinance/FRED real-data adapters, agent calibration
metrics (Brier score and Expected Calibration Error), and walk-forward
backtesting with institutional performance statistics.

## Methodology References

- CFA Institute Level I/II curriculum: Equity Valuation, Corporate Issuers,
  Fixed Income, and Portfolio Management
- Damodaran, A., *Investment Valuation*
- Penman, S., *Financial Statement Analysis and Security Valuation*
- Nelson, C.R. and Siegel, A.F. (1987), "Parsimonious Modeling of Yield Curves"
- Diebold, F.X. and Li, C. (2006), "Forecasting the Term Structure of
  Government Bond Yields"
- Brier, G.W. (1950), "Verification of Forecasts Expressed in Terms of
  Probability"
- Guo, C. et al. (2017), "On Calibration of Modern Neural Networks"
- Sharpe, W.F. (1994), "The Sharpe Ratio"

## Scope

This is a research and interview project, not a broker-connected trading
system. It does not place trades and should not be used as financial advice.
Synthetic data is used by default; real-data mode depends on delayed or
third-party provider data.

## License

MIT, for educational and research purposes.
