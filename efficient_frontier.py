"""Compute an approximate efficient frontier for the OHLCV dataset in ``temp.csv``.

The script reuses the CSV parsing helpers from :mod:`analyze_temp_csv` to
extract closing prices, converts them into daily returns and then explores a
simple long-only portfolio space to approximate the efficient frontier.  The
search is grid based which keeps the implementation dependency free while still
providing a useful visualisation for a small number of tickers.

Example usage::

    python efficient_frontier.py              # Analyse temp.csv (default)
    python efficient_frontier.py data.csv --step 0.02

The output lists the expected return and risk (standard deviation of returns)
for each portfolio that lies on the frontier along with the corresponding
weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple
import argparse

from analyze_temp_csv import compute_daily_returns, load_ohlcv


@dataclass(frozen=True)
class PortfolioPoint:
    """A single portfolio on the efficient frontier."""

    expected_return: float
    risk: float
    weights: Dict[str, float]

    def weight_string(self) -> str:
        ordered = sorted(self.weights.items())
        return ", ".join(f"{ticker}={weight:5.2f}" for ticker, weight in ordered)


def _align_returns(
    returns_by_ticker: Dict[str, List[Tuple[str, float]]]
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Align return series so that each ticker has values for the same dates."""

    if not returns_by_ticker:
        return [], {}

    overlapping_dates = None
    for series in returns_by_ticker.values():
        series_dates = {date for date, _ in series}
        if overlapping_dates is None:
            overlapping_dates = series_dates
        else:
            overlapping_dates &= series_dates
    if not overlapping_dates:
        return [], {}

    ordered_dates = sorted(overlapping_dates)
    aligned_returns: Dict[str, List[float]] = {}
    for ticker, series in returns_by_ticker.items():
        value_map = {date: value for date, value in series}
        aligned_returns[ticker] = [value_map[date] for date in ordered_dates]
    return ordered_dates, aligned_returns


def _covariance(sample_x: Sequence[float], sample_y: Sequence[float]) -> float:
    if len(sample_x) != len(sample_y):
        raise ValueError("Return series must have identical length to compute covariance.")
    if len(sample_x) < 2:
        raise ValueError("Need at least two observations to compute covariance.")
    mean_x = mean(sample_x)
    mean_y = mean(sample_y)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(sample_x, sample_y)) / (len(sample_x) - 1)


def _portfolio_variance(weights: Sequence[float], cov_matrix: List[List[float]]) -> float:
    variance = 0.0
    for i, weight_i in enumerate(weights):
        for j, weight_j in enumerate(weights):
            variance += weight_i * weight_j * cov_matrix[i][j]
    return variance


def _generate_weight_grid(tickers: Sequence[str], step: float) -> Iterable[Dict[str, float]]:
    if len(tickers) == 0:
        return []
    if len(tickers) == 1:
        yield {tickers[0]: 1.0}
        return

    def recurse(index: int, remaining: float, current: Dict[str, float]):
        if index == len(tickers) - 1:
            current[tickers[index]] = remaining
            yield dict(current)
            return
        ticker = tickers[index]
        next_value = 0.0
        while next_value <= remaining + 1e-9:
            current[ticker] = round(next_value, 10)
            yield from recurse(index + 1, remaining - next_value, current)
            next_value = round(next_value + step, 10)

    yield from recurse(0, 1.0, {})


def compute_efficient_frontier(
    csv_path: Path, step: float = 0.01
) -> Tuple[List[PortfolioPoint], List[PortfolioPoint]]:
    dates, dataset = load_ohlcv(csv_path)
    if not dates:
        raise ValueError("CSV file does not contain any data rows.")

    close_returns = {
        ticker: compute_daily_returns(metrics["Close"])
        for ticker, metrics in dataset.items()
        if "Close" in metrics
    }
    if len(close_returns) < 2:
        raise ValueError("At least two tickers with closing prices are required to build a frontier.")

    _, aligned_returns = _align_returns(close_returns)
    if not aligned_returns:
        raise ValueError("Could not find overlapping dates across tickers to compute returns.")

    tickers = sorted(aligned_returns.keys())
    means = [mean(aligned_returns[ticker]) for ticker in tickers]
    cov_matrix: List[List[float]] = []
    for ticker_x in tickers:
        row = []
        for ticker_y in tickers:
            row.append(_covariance(aligned_returns[ticker_x], aligned_returns[ticker_y]))
        cov_matrix.append(row)

    all_portfolios: List[PortfolioPoint] = []
    for weights in _generate_weight_grid(tickers, step):
        weight_vector = [weights[ticker] for ticker in tickers]
        expected_return = sum(weight * avg for weight, avg in zip(weight_vector, means))
        variance = _portfolio_variance(weight_vector, cov_matrix)
        risk = sqrt(max(variance, 0.0))
        all_portfolios.append(
            PortfolioPoint(expected_return=expected_return, risk=risk, weights=weights)
        )

    sorted_portfolios = sorted(all_portfolios, key=lambda entry: entry.risk)
    efficient: List[PortfolioPoint] = []
    best_return = float("-inf")
    for point in sorted_portfolios:
        if point.expected_return > best_return + 1e-9:
            efficient.append(point)
            best_return = point.expected_return

    return efficient, sorted_portfolios


def plot_efficient_frontier(
    efficient: Sequence[PortfolioPoint],
    all_portfolios: Sequence[PortfolioPoint],
    output_path: Path,
) -> None:
    """Create a scatter plot that highlights the efficient frontier."""

    from matplotlib import pyplot as plt

    if not efficient or not all_portfolios:
        raise ValueError("Need portfolio data to plot the efficient frontier.")

    all_risk = [point.risk for point in all_portfolios]
    all_return = [point.expected_return for point in all_portfolios]
    frontier_risk = [point.risk for point in efficient]
    frontier_return = [point.expected_return for point in efficient]

    plt.figure(figsize=(8, 6))
    plt.scatter(all_risk, all_return, s=15, alpha=0.3, label="Portfolios")
    plt.plot(
        frontier_risk,
        frontier_return,
        color="tab:orange",
        marker="o",
        linewidth=2,
        markersize=4,
        label="Efficient frontier",
    )
    plt.title("Efficient Frontier (Daily Returns)")
    plt.xlabel("Risk (standard deviation)")
    plt.ylabel("Expected return")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute an efficient frontier for temp.csv")
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("temp.csv"),
        help="Path to the OHLCV CSV file (defaults to temp.csv)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Grid resolution for weights. Smaller values improve accuracy at the cost of runtime.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("efficient_frontier.png"),
        help="Path where the efficient frontier plot will be saved.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the efficient frontier plot.",
    )
    args = parser.parse_args()

    if args.step <= 0 or args.step > 1:
        raise ValueError("Step size must be between 0 and 1.")

    frontier, all_portfolios = compute_efficient_frontier(args.csv_path, step=args.step)
    if not frontier:
        print("No efficient portfolios were found.")
        return

    print("Efficient frontier (daily returns)")
    print("===============================")
    print("Return      Risk        Weights")
    for point in frontier:
        print(f"{point.expected_return:8.5f}  {point.risk:8.5f}  {point.weight_string()}")

    if not args.no_plot:
        output_path = args.plot_output
        plot_efficient_frontier(frontier, all_portfolios, output_path)
        print(f"Saved efficient frontier plot to {output_path}")


if __name__ == "__main__":
    main()
