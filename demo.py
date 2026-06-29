"""
demo.py

Single-command end-to-end demo of the platform. Run with:
    python3 demo.py
    python3 demo.py --real --ticker AAPL --peers MSFT,GOOGL,META,AMZN

This script:
    1. Generates synthetic (clearly labeled) market + yield curve data
    2. Runs the CFA Valuation Agent (FCFF/FCFE/WACC/DCF/comps/RI/scenarios)
    3. Runs the Fixed-Income Agent (Nelson-Siegel + Kalman filter + rate sensitivity)
    4. Combines them via the Investment Committee orchestrator
    5. Runs a walk-forward backtest of a simple signal-driven strategy
    6. Reports full institutional performance metrics
    7. Simulates and reports agent calibration (confidence vs. accuracy)

This is meant as a reference entry point -- read it top to bottom to see
how all the modules connect, then dig into the individual core/ and agents/
files for the actual implementations.
"""

import argparse
from datetime import date

import numpy as np
import pandas as pd

from data.synthetic_market import SyntheticMarketDataSource
from data.real_market import RealMarketDataSource
from core.valuation import FinancialInputs, PeerMultiples
from core.metrics import full_performance_report
from core.calibration import AgentCalibrationTracker, simulate_agent_calls
from backtest.walk_forward import run_walk_forward
from agents.llm_client import MockLLMClient
from agents.valuation_agent import ValuationAgent
from agents.fixed_income_agent import FixedIncomeAgent
from agents.orchestrator import InvestmentCommittee


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the investment research platform demo.")
    parser.add_argument("--real", action="store_true", help="Use real market data from yfinance and FRED.")
    parser.add_argument("--ticker", default="DEMOCORP", help="Ticker to analyze in real-data mode.")
    parser.add_argument("--peers", default="", help="Comma-separated peer tickers for comparable-company multiples.")
    parser.add_argument("--start", default="2018-01-01", help="Price-history start date.")
    parser.add_argument("--end", default=date.today().isoformat(), help="Price-history end date.")
    parser.add_argument("--yield-start", default="2022-01-01", help="Treasury yield-panel start date.")
    parser.add_argument("--yield-end", default=date.today().isoformat(), help="Treasury yield-panel end date.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    if args.real:
        section("1. REAL MARKET DATA (yfinance prices/fundamentals + FRED Treasury curve)")
        source = RealMarketDataSource()
        peer_arg = [p.strip().upper() for p in args.peers.split(",") if p.strip()]
        try:
            bundle = source.build_data_bundle(
                args.ticker,
                args.start,
                args.end,
                args.yield_start,
                args.yield_end,
                peers=peer_arg or None,
            )
        except (ConnectionError, ValueError, ImportError) as exc:
            print(f"Real-data mode failed: {exc}")
            print("Falling back to synthetic data so the rest of the demo can still run. "
                  "Check your internet connection and that 'yfinance' is installed "
                  "(pip install -r requirements.txt), then retry --real.")
            args.real = False
        else:
            ticker = bundle.ticker
            price_df = bundle.price_history
            yield_panel_df = bundle.yield_curve_panel
            inputs = bundle.financial_inputs
            peers = bundle.peer_multiples
            print(f"Loaded {len(price_df)} trading days for {ticker} "
                  f"(latest close: {price_df['close'].iloc[-1]:.2f})")
            print(f"Loaded Treasury curve panel: {yield_panel_df.shape[0]} dates x "
                  f"{yield_panel_df.shape[1]} maturities {list(yield_panel_df.columns)}")
            print(f"Financial inputs from yfinance statements/info: revenue={inputs.revenue:,.0f}, "
                  f"EBIT={inputs.ebit:,.0f}, shares={inputs.shares_outstanding:,.0f}")
            print(f"Peer multiples from: {', '.join(bundle.peers_used) if bundle.peers_used else 'none'}")

    if not args.real:
        section("1. SYNTHETIC MARKET DATA (clearly labeled placeholder -- "
                "run with --real to use yfinance/FRED)")
        source = SyntheticMarketDataSource(seed=42)
        ticker = "DEMOCORP"
        price_df = source.get_price_history(ticker, "2018-01-01", "2023-12-31")
        print(f"Generated {len(price_df)} trading days for {ticker} "
              f"(price range: {price_df['close'].min():.2f} - {price_df['close'].max():.2f})")

        yield_panel_df = source.get_yield_curve_panel("2022-01-01", "2023-12-31")
        print(f"Generated yield curve panel: {yield_panel_df.shape[0]} dates x "
              f"{yield_panel_df.shape[1]} maturities {list(yield_panel_df.columns)}")

        inputs = FinancialInputs(
            ticker=ticker, revenue=5000, ebit=900, ebitda=1100, tax_rate=0.22,
            capex=350, depreciation_amortization=200, change_in_nwc=50,
            net_debt=1200, shares_outstanding=400, cost_of_debt=0.045,
            cost_of_equity=0.095, market_cap=8000, total_debt=1800,
            revenue_growth_rate=0.07, terminal_growth_rate=0.025,
            net_income=600, book_value_equity=3500,
            current_price=float(price_df["close"].iloc[-1] / 5),
        )
        peers = PeerMultiples(ev_ebitda_peers=[9.5, 10.5, 11.5, 12.0], pe_peers=[17, 19, 21, 23])

    # ------------------------------------------------------------------
    # 2. CFA Valuation Agent
    # ------------------------------------------------------------------
    section("2. CFA VALUATION AGENT")
    llm = MockLLMClient()  # swap for AnthropicLLMClient() with ANTHROPIC_API_KEY set
    valuation_agent = ValuationAgent(llm)

    val_output = valuation_agent.analyze(inputs, peers)
    print(f"Recommendation:        {val_output.decision}")
    print(f"Confidence:            {val_output.confidence:.1%} (derived from cross-method agreement)")
    print(f"DCF intrinsic value:   ${val_output.raw_report.dcf.intrinsic_value_per_share:.2f}/share")
    print(f"Comps (EV/EBITDA):     ${val_output.raw_report.comps.implied_value_ev_ebitda:.2f}/share")
    print(f"Comps (P/E):           ${val_output.raw_report.comps.implied_value_pe:.2f}/share")
    print(f"Residual income:       ${val_output.raw_report.residual_income:.2f}/share")
    print(f"Bear/Base/Bull:        ${val_output.raw_report.scenarios.bear_value:.2f} / "
          f"${val_output.raw_report.scenarios.base_value:.2f} / ${val_output.raw_report.scenarios.bull_value:.2f}")
    print(f"Margin of safety:      {val_output.margin_of_safety:.1%}")
    print(f"Narrative: {val_output.narrative}")

    # ------------------------------------------------------------------
    # 3. Fixed-Income / Credit Agent
    # ------------------------------------------------------------------
    section("3. FIXED-INCOME / CREDIT AGENT (Nelson-Siegel + Kalman filter)")
    fi_agent = FixedIncomeAgent(llm)

    maturities = yield_panel_df.columns.values.astype(float)
    current_yields = yield_panel_df.iloc[-1].values
    yield_panel = yield_panel_df.values

    aligned_rates = pd.concat(
        [price_df["return"].rename("equity_return"), yield_panel_df[10.0].diff().rename("ten_year_change")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned_rates) >= 20:
        equity_returns_sample = aligned_rates["equity_return"].values
        ten_year_changes = aligned_rates["ten_year_change"].values
    else:
        equity_returns_sample = price_df["return"].iloc[-len(yield_panel_df):].values
        ten_year_changes = yield_panel_df[10.0].diff().fillna(0).values

    fi_output = fi_agent.analyze(maturities, current_yields, yield_panel, equity_returns_sample, ten_year_changes)
    print(f"Decision:              {fi_output.decision}")
    print(f"Confidence:            {fi_output.confidence:.1%} (derived from Kalman filter state uncertainty)")
    print(f"NS Level/Slope/Curve:  {fi_output.ns_level:.4f} / {fi_output.ns_slope:.4f} / {fi_output.ns_curvature:.4f}")
    print(f"Rate sensitivity beta: {fi_output.rate_sensitivity.beta_to_10y:.2f} (R^2={fi_output.rate_sensitivity.r_squared:.2f})")
    print(f"Flag level:            {fi_output.rate_sensitivity.flag_level}")
    print(f"Narrative: {fi_output.narrative}")

    # ------------------------------------------------------------------
    # 4. Investment Committee
    # ------------------------------------------------------------------
    section("4. INVESTMENT COMMITTEE (combined decision)")
    tracker = AgentCalibrationTracker()
    committee = InvestmentCommittee(valuation_agent, fi_agent, tracker)
    decision = committee.decide(
        ticker, inputs, peers, maturities, current_yields, yield_panel,
        equity_returns_sample, ten_year_changes
    )
    print(f"FINAL DECISION: {decision.final_decision}")
    print(f"Rationale: {decision.rationale}")

    # ------------------------------------------------------------------
    # 5. Walk-forward backtest of a simple signal strategy
    # ------------------------------------------------------------------
    section("5. WALK-FORWARD BACKTEST (no look-ahead bias)")

    def fit_fn(train_df):
        # Simple momentum signal fit on the training window: sign of trailing
        # 20-day mean return. This is intentionally simple -- the point of
        # this demo section is to exercise the walk-forward ENGINE end-to-end
        # with full institutional metrics, not to claim this is alpha.
        trailing_mean = train_df["return"].tail(20).mean()
        return {"trailing_mean": trailing_mean}

    def decide_fn(fitted_state, current_row):
        return "BUY" if fitted_state["trailing_mean"] > 0 else "HOLD"

    initial_train_size = min(500, max(126, len(price_df) // 2))
    test_size = min(126, max(21, (len(price_df) - initial_train_size) // 4))
    wf_result = run_walk_forward(
        price_df, fit_fn, decide_fn,
        initial_train_size=initial_train_size, test_size=test_size,
        expanding=True,
    )
    print(f"Walk-forward windows: {len(wf_result.windows)}")
    print(f"Out-of-sample periods: {len(wf_result.strategy_returns)}")

    perf = full_performance_report(wf_result.strategy_returns, wf_result.benchmark_returns)
    print(f"\n--- INSTITUTIONAL PERFORMANCE METRICS (strategy vs. buy-and-hold benchmark) ---")
    print(f"CAGR:               {perf.cagr:.2%}")
    print(f"Annualized Return:  {perf.annualized_return:.2%}")
    print(f"Annualized Vol:     {perf.annualized_vol:.2%}")
    print(f"Sharpe Ratio:       {perf.sharpe:.2f}")
    print(f"Sortino Ratio:      {perf.sortino:.2f}")
    print(f"Calmar Ratio:       {perf.calmar:.2f}")
    print(f"Max Drawdown:       {perf.max_drawdown:.2%}")
    print(f"Information Ratio: {perf.information_ratio:.2f}")
    print(f"Alpha (annualized): {perf.alpha:.2%}")
    print(f"Beta:               {perf.beta:.2f}")
    print(f"Win Rate:           {perf.win_rate:.2%}")

    # ------------------------------------------------------------------
    # 6. Agent calibration report
    # ------------------------------------------------------------------
    section("6. AGENT CALIBRATION REPORT (confidence vs. realized accuracy)")
    rng2 = np.random.default_rng(123)
    overconfident_calls = simulate_agent_calls("FundamentalAgent_Demo", n_calls=300, true_skill=0.61, confidence_bias=0.32, rng=rng2)
    calibrated_calls = simulate_agent_calls("RiskAgent_Demo", n_calls=300, true_skill=0.58, confidence_bias=0.0, rng=rng2)
    tracker.log_many(overconfident_calls)
    tracker.log_many(calibrated_calls)

    for name in ["FundamentalAgent_Demo", "RiskAgent_Demo"]:
        rep = tracker.report(name)
        print(f"\nAgent: {name}")
        print(f"  N calls:              {rep.n_calls}")
        print(f"  Mean confidence:      {rep.mean_confidence:.1%}")
        print(f"  Realized accuracy:    {rep.overall_accuracy:.1%}")
        print(f"  Brier score:          {rep.brier_score:.3f}  (0=perfect, 1=worst)")
        print(f"  Expected Cal. Error:  {rep.expected_calibration_error:.3f}")
        print(f"  Decision entropy:     {rep.decision_entropy:.3f}  (max={np.log2(3):.3f})")
        print(f"  >>> OVERCONFIDENT FLAG: {rep.is_overconfident}")

    section("DONE -- see README.md for architecture notes and resource links")


if __name__ == "__main__":
    main()
