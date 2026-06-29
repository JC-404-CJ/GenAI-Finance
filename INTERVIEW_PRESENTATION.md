# Interview Presentation Guide

## 60-Second Pitch

I built an institutional-style investment research platform that combines
fundamental equity valuation, fixed-income curve modelling, GenAI-assisted
research narration, calibration tracking, and walk-forward backtesting.

The main design choice is that language models never create the financial
numbers. Valuation, yield-curve modelling, performance metrics, and calibration
are deterministic Python modules. The narration layer only explains outputs
that have already been calculated and tested.

## Demo Flow

1. Run the offline demo:

```bash
python demo.py
```

2. Point out the six sections:

- Synthetic or real market data loading
- CFA valuation agent
- Fixed-income / credit risk overlay
- Investment committee decision
- Walk-forward backtest
- Calibration report

3. If internet access is available, run:

```bash
python demo.py --real --ticker AAPL --peers MSFT,GOOGL,META,AMZN
```

## Strong Talking Points

- The valuation engine includes FCFF, FCFE, WACC, multi-stage DCF, comparable
  multiples, residual income, and scenario analysis.
- The fixed-income module fits Nelson-Siegel curves and tracks level/slope/
  curvature with a Kalman filter.
- The risk overlay can downgrade an attractive equity call when rate
  sensitivity is high.
- Confidence is checked with calibration metrics instead of being taken at
  face value.
- The backtester uses expanding walk-forward windows, so the strategy never
  trains on future data.
- The real-data adapter normalizes yfinance statements and FRED Treasury
  curves into the same contracts used by the synthetic data path.

## Files To Open Live

- `demo.py` for the complete pipeline
- `core/valuation.py` for the valuation math
- `core/fixed_income.py` for Nelson-Siegel and Kalman filtering
- `agents/orchestrator.py` for the committee decision rule
- `core/calibration.py` for Brier score, ECE, and overconfidence detection
- `backtest/walk_forward.py` for no-look-ahead evaluation

## Likely Questions

**Why use GenAI here?**  
For analyst-style narration and communication, not for unverified calculation.
That keeps the outputs explainable while still showing how GenAI fits into a
finance workflow.

**How do you know the agents are reliable?**  
Each decision can be logged with its confidence and realized outcome. The
calibration tracker computes Brier score, Expected Calibration Error, realized
accuracy, and an overconfidence flag.

**What would you add next?**  
Portfolio construction with Black-Litterman or HRP, a filings-based RAG layer,
options Greeks, and a dashboard for scenario and calibration visualization.

**What is the biggest engineering lesson?**  
Keep the numerical core deterministic and tested. Then add GenAI as an
interface layer around the analysis rather than as the source of truth.
