"""Utility functions to analyze the multi-ticker OHLCV dataset stored in ``temp.csv``.

The script is intentionally dependency free so that it can run in a bare Python
environment.  It normalises the repeated two-row header into a structured
representation and then prints a compact statistical summary that can serve as a
starting point for deeper analysis.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple
import argparse
import csv


@dataclass
class MetricSummary:
    metric: str
    count: int
    minimum: float
    maximum: float
    average: float


def _parse_float(value: str) -> float:
    if value == "":
        raise ValueError("Encountered an empty numeric cell in the CSV file.")
    return float(value)


def _pairwise(iterable: Iterable[Tuple[str, float]]) -> Iterable[Tuple[Tuple[str, float], Tuple[str, float]]]:
    iterator = iter(iterable)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    for current in iterator:
        yield previous, current
        previous = current


def _correlation(sample_x: List[float], sample_y: List[float]) -> float | None:
    if len(sample_x) != len(sample_y) or len(sample_x) < 2:
        return None
    mean_x = mean(sample_x)
    mean_y = mean(sample_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(sample_x, sample_y))
    denominator_x = sum((x - mean_x) ** 2 for x in sample_x)
    denominator_y = sum((y - mean_y) ** 2 for y in sample_y)
    denominator = sqrt(denominator_x * denominator_y)
    if denominator == 0:
        return None
    return numerator / denominator


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:6.2f}%"


def _aligned_correlation(
    series_x: List[Tuple[str, float]], series_y: List[Tuple[str, float]]
) -> float | None:
    if not series_x or not series_y:
        return None
    map_x = {date: value for date, value in series_x}
    map_y = {date: value for date, value in series_y}
    overlapping_dates = sorted(set(map_x.keys()) & set(map_y.keys()))
    if len(overlapping_dates) < 2:
        return None
    aligned_x = [map_x[date] for date in overlapping_dates]
    aligned_y = [map_y[date] for date in overlapping_dates]
    return _correlation(aligned_x, aligned_y)


def load_ohlcv(path: Path) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if any(cell for cell in row)]

    if len(rows) < 4:
        raise ValueError("Expected at least four rows (two headers, a label row and data).")

    header_metrics = rows[0][1:]
    header_tickers = rows[1][1:]
    if len(header_metrics) != len(header_tickers):
        raise ValueError("Header rows have different lengths, cannot align metrics with tickers.")

    column_meta = list(zip(header_tickers, header_metrics))

    dates: List[str] = []
    dataset: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for raw_row in rows[3:]:
        if not raw_row:
            continue
        date = raw_row[0]
        dates.append(date)
        for (ticker, metric), value in zip(column_meta, raw_row[1:]):
            value = value.strip()
            if value == "":
                # Skip over missing entries so that later statistics remain
                # well-defined.  The dataset is dense, so this branch is
                # primarily defensive programming for unexpected gaps.
                continue
            dataset[ticker][metric].append((date, _parse_float(value)))

    return dates, dataset


def build_metric_summary(values: Sequence[Tuple[str, float]], metric_name: str) -> MetricSummary:
    only_values = [value for _, value in values]
    return MetricSummary(
        metric=metric_name,
        count=len(only_values),
        minimum=min(only_values),
        maximum=max(only_values),
        average=mean(only_values),
    )


def compute_daily_returns(values: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    if not values:
        return []
    sorted_values = sorted(values, key=lambda entry: entry[0])
    returns: List[Tuple[str, float]] = []
    for (prev_date, previous), (curr_date, current) in _pairwise(sorted_values):
        if previous == 0:
            returns.append((curr_date, 0.0))
        else:
            returns.append((curr_date, (current - previous) / previous))
    return returns


def describe_dataset(path: Path) -> str:
    dates, dataset = load_ohlcv(path)
    if not dates:
        raise ValueError("No data rows were found in the CSV file.")

    tickers = sorted(dataset.keys())
    metrics = sorted({metric for ticker in tickers for metric in dataset[ticker].keys()})

    lines: List[str] = []
    lines.append("Dataset overview")
    lines.append("=================")
    lines.append(f"Rows (trading days): {len(dates)}")
    lines.append(f"Date range         : {dates[0]} → {dates[-1]}")
    lines.append(f"Tickers            : {', '.join(tickers)}")
    lines.append(f"Metrics            : {', '.join(metrics)}")
    lines.append("")

    for ticker in tickers:
        lines.append(f"Ticker {ticker}")
        lines.append("-" * (7 + len(ticker)))
        metrics_for_ticker = dataset[ticker]
        summaries = [
            build_metric_summary(sorted(values, key=lambda entry: entry[0]), metric)
            for metric, values in sorted(metrics_for_ticker.items())
        ]
        for summary in summaries:
            lines.append(
                f"{summary.metric:<6} → count={summary.count:3d}, min={summary.minimum:12.4f}, "
                f"max={summary.maximum:12.4f}, mean={summary.average:12.4f}"
            )
        if "Close" in metrics_for_ticker:
            closes = metrics_for_ticker["Close"]
            returns = compute_daily_returns(closes)
            if returns:
                best_day = max(returns, key=lambda entry: entry[1])
                worst_day = min(returns, key=lambda entry: entry[1])
                lines.append(
                    f"Best daily close return : {_format_percentage(best_day[1])} on {best_day[0]}"
                )
                lines.append(
                    f"Worst daily close return: {_format_percentage(worst_day[1])} on {worst_day[0]}"
                )
                lines.append(
                    f"Average daily close return: {_format_percentage(mean(value for _, value in returns))}"
                )
        lines.append("")

    # Correlation matrix across tickers for closing returns
    lines.append("Close-to-close return correlations")
    lines.append("==================================")
    close_returns = {
        ticker: compute_daily_returns(metrics["Close"])
        for ticker, metrics in dataset.items()
        if "Close" in metrics
    }

    ordered_tickers = sorted(close_returns.keys())
    if ordered_tickers:
        header_row = "Ticker".ljust(8) + "".join(t.ljust(12) for t in ordered_tickers)
        lines.append(header_row)
        for ticker_x in ordered_tickers:
            row = ticker_x.ljust(8)
            for ticker_y in ordered_tickers:
                corr = _aligned_correlation(close_returns[ticker_x], close_returns[ticker_y])
                if corr is None:
                    cell = "   n/a     "
                else:
                    cell = f"{corr:10.4f}  "
                row += cell
            lines.append(row)
    else:
        lines.append("No closing price data available to compute correlations.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the dataset stored in temp.csv")
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("temp.csv"),
        help="Path to the OHLCV CSV file (defaults to temp.csv in the project root)",
    )
    args = parser.parse_args()

    report = describe_dataset(args.csv_path)
    print(report)


if __name__ == "__main__":
    main()
