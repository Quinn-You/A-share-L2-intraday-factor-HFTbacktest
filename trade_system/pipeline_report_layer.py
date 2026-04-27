#!/usr/bin/env python3
"""汇总层：根据策略层输出生成图表与 HTML 报表。

这一层不再负责总入口调度，只接收策略层产出的结果对象，
然后补充：
1. 全市场维度的汇总指标。
2. 各类图表文件。
3. HTML 报告和对外展示用 CSV/JSON。
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pipeline_strategy_layer import StrategyRunResult


# 图表优先使用可显示中文的字体，避免 DejaVu Sans 缺字告警。
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
	"DejaVu Sans",
	"Arial",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def _safe_sharpe(realized_returns: list[float]) -> float | None:
	"""基于单笔真实收益率序列计算全市场层面的 Sharpe。"""
	if len(realized_returns) < 2:
		return None
	arr = np.asarray(realized_returns, dtype=np.float64)
	std = float(arr.std(ddof=1))
	if not np.isfinite(std) or std <= 0:
		return None
	return float(math.sqrt(arr.size) * arr.mean() / std)


def _report_dir(repo_dir: Path) -> Path:
	"""报表层输出目录。"""
	return repo_dir / "report_layer"


def _symbol_chart_dir(report_dir: Path) -> Path:
	"""个股图表输出目录。"""
	return report_dir / "symbol_charts"


def _plot_cumulative_pnl(all_trades: pl.DataFrame, out_path: Path) -> None:
	"""按平仓时间在同一张图中绘制 Long/Short/Total 三条累计 PnL 子图。"""
	plot_df = (
		all_trades.select(["exit_time", "side", "pnl"])
		.with_columns(pl.col("exit_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False))
		.drop_nulls(["exit_time", "side", "pnl"])
		.sort("exit_time")
	)
	if plot_df.height == 0:
		return

	def _draw_subplot(ax: plt.Axes, trades_df: pl.DataFrame, title: str, color: str) -> None:
		ax.set_title(title)
		ax.set_ylabel("Cumulative PnL")
		ax.axhline(0.0, color="#334155", linewidth=1.0)
		ax.grid(alpha=0.25)
		if trades_df.height == 0:
			ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes, color="#64748b")
			return
		x_values = trades_df.get_column("exit_time").to_list()
		y_values = np.cumsum(trades_df.get_column("pnl").cast(pl.Float64).to_numpy())
		ax.plot(x_values, y_values, color=color, linewidth=2.0)

	long_df = plot_df.filter(pl.col("side") == "long")
	short_df = plot_df.filter(pl.col("side") == "short")

	fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
	_draw_subplot(axes[0], long_df, "Long Cumulative PnL", "#1d4ed8")
	_draw_subplot(axes[1], short_df, "Short Cumulative PnL", "#b91c1c")
	_draw_subplot(axes[2], plot_df, "Total Cumulative PnL", "#0f766e")
	axes[2].set_xlabel("Exit Time")
	fig.autofmt_xdate()
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def _plot_rank_chart(summary_df: pl.DataFrame, metric: str, title: str, out_path: Path) -> None:
	"""绘制指定指标的头部/尾部股票排名图。"""
	usable = summary_df.filter(pl.col(metric).is_not_null()).sort(metric)
	if usable.height == 0:
		return
	bottom = usable.head(10)
	top = usable.tail(10)
	plot_df = pl.concat([bottom, top]).unique(subset=["symbol"], keep="last")
	labels = plot_df.get_column("symbol").to_list()
	values = plot_df.get_column(metric).cast(pl.Float64).to_list()
	colors = ["#b91c1c" if value < 0 else "#15803d" for value in values]

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.bar(labels, values, color=colors)
	ax.axhline(0.0, color="#334155", linewidth=1.0)
	ax.set_title(title)
	ax.set_ylabel(metric)
	ax.tick_params(axis="x", rotation=45)
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def _plot_symbol_net_position(symbol: str, symbol_trades: pl.DataFrame, out_path: Path) -> bool:
	"""根据开平仓事件绘制单只股票的日内净仓位曲线。"""
	trade_rows = (
		symbol_trades.select(["side", "shares", "entry_time", "exit_time"])
		.with_columns(
			pl.col("entry_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("entry_dt"),
			pl.col("exit_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("exit_dt"),
		)
		.drop_nulls(["entry_dt", "exit_dt", "shares", "side"])
	)
	if trade_rows.height == 0:
		return False

	events: list[tuple[object, float]] = []
	for side, shares, entry_dt, exit_dt in trade_rows.select(["side", "shares", "entry_dt", "exit_dt"]).iter_rows():
		direction = 1.0 if side == "long" else -1.0
		qty = float(shares)
		events.append((entry_dt, direction * qty))
		events.append((exit_dt, -direction * qty))

	if not events:
		return False

	events.sort(key=lambda item: item[0])
	x_values = [events[0][0]]
	y_values = [0.0]
	current_position = 0.0
	for ts, delta in events:
		x_values.append(ts)
		y_values.append(current_position)
		current_position += delta
		x_values.append(ts)
		y_values.append(current_position)

	fig, ax = plt.subplots(figsize=(12, 5))
	ax.step(x_values, y_values, where="post", color="#1d4ed8", linewidth=1.8)
	ax.axhline(0.0, color="#334155", linewidth=1.0)
	ax.set_title(f"{symbol} Net Position")
	ax.set_xlabel("Time")
	ax.set_ylabel("Net Position")
	ax.grid(alpha=0.25)
	fig.autofmt_xdate()
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)
	return True


def _plot_pnl_distribution(summary_df: pl.DataFrame, out_path: Path) -> None:
	"""绘制股票总 PnL 的分布直方图。"""
	usable = summary_df.filter(pl.col("n_trades") > 0)
	if usable.height == 0:
		return
	pnl_values = usable.get_column("total_pnl").cast(pl.Float64).to_numpy()

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.hist(pnl_values, bins=min(30, max(10, usable.height // 2)), color="#1d4ed8", alpha=0.85)
	ax.set_title("Symbol Total PnL Distribution")
	ax.set_xlabel("Total PnL")
	ax.set_ylabel("Count")
	ax.grid(alpha=0.2)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def _metric_top_bottom_symbols(summary_df: pl.DataFrame, metric: str, top_n: int = 10) -> tuple[list[str], list[str]]:
	"""返回指定指标的最优/最差股票列表。"""
	usable = summary_df.filter(pl.col(metric).is_not_null())
	if usable.height == 0:
		return [], []
	top_symbols = usable.sort(metric, descending=True).head(top_n).get_column("symbol").to_list()
	bottom_symbols = usable.sort(metric).head(top_n).get_column("symbol").to_list()
	return top_symbols, bottom_symbols


def _build_symbol_chart_manifest(result: StrategyRunResult, report_dir: Path) -> pl.DataFrame:
	"""为每只股票生成净仓位图，并输出索引表。"""
	chart_dir = _symbol_chart_dir(report_dir)
	chart_dir.mkdir(parents=True, exist_ok=True)

	manifest_rows: list[dict[str, object]] = []
	if result.all_trades_df.height == 0:
		return pl.DataFrame(
			{
				"symbol": [],
				"n_trades": [],
				"net_position_path": [],
			}
		)

	for symbol in sorted(result.all_trades_df.get_column("symbol").unique().to_list()):
		symbol_trades = result.all_trades_df.filter(pl.col("symbol") == symbol)
		symbol_dir = chart_dir / symbol
		symbol_dir.mkdir(parents=True, exist_ok=True)
		net_position_path = symbol_dir / "net_position.png"

		has_net_position = _plot_symbol_net_position(symbol, symbol_trades, net_position_path)
		manifest_rows.append(
			{
				"symbol": symbol,
				"n_trades": int(symbol_trades.height),
				"net_position_path": str(net_position_path.relative_to(report_dir)) if has_net_position else None,
			}
		)

	return pl.DataFrame(manifest_rows)


def _html_chart_link(path_text: str | None, label: str) -> str:
	"""生成 HTML 图表链接。"""
	if not path_text:
		return ""
	escaped_path = html.escape(path_text)
	escaped_label = html.escape(label)
	return f'<a href="{escaped_path}">{escaped_label}</a>'


def _write_html_report(
	out_dir: Path,
	summary_row: dict[str, object],
	chart_paths: list[Path],
	symbol_chart_manifest: pl.DataFrame,
) -> None:
	"""把汇总指标和图表文件拼成一份静态 HTML 报告。"""
	html_path = out_dir / "report.html"
	rows = "\n".join(
		f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value))}</td></tr>"
		for key, value in summary_row.items()
	)
	images = "\n".join(
		f'<section><img src="{chart_path.name}" alt="{chart_path.stem}" style="max-width:100%;height:auto;border:1px solid #ddd;"></section>'
		for chart_path in chart_paths
		if chart_path.exists()
	)
	symbol_rows = "\n".join(
		"<tr>"
		f"<td>{html.escape(str(symbol))}</td>"
		f"<td>{int(n_trades)}</td>"
		f"<td>{_html_chart_link(net_position_path, 'net_position')}</td>"
		"</tr>"
		for symbol, n_trades, net_position_path in symbol_chart_manifest.select(
			["symbol", "n_trades", "net_position_path"]
		).iter_rows()
	)
	html_text = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>pipeline report</title>
  <style>
    body {{ font-family: "DejaVu Sans", sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; margin: 16px 0 24px; width: 760px; max-width: 100%; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 10px 12px; text-align: left; }}
    th {{ width: 280px; background: #e2e8f0; }}
    section {{ margin: 24px 0; background: white; padding: 12px; }}
  </style>
</head>
<body>
  <h1>全市场回测报表</h1>
  <table>
    {rows}
  </table>
	<h2>个股图表索引</h2>
	<table>
		<tr><th>symbol</th><th>n_trades</th><th>Net position</th></tr>
		{symbol_rows}
	</table>
  {images}
</body>
</html>
""".strip()
	html_path.write_text(html_text, encoding="utf-8")


def generate_report(result: StrategyRunResult) -> dict[str, object]:
	"""根据策略层结果生成最终汇总和图表。

	输入是策略层已经完成回测后的结果对象；
	输出是一份 overall_summary 字典，同时把 csv/json/png/html 文件写入 report_layer 目录。
	"""

	report_dir = _report_dir(result.repo_dir)
	report_dir.mkdir(parents=True, exist_ok=True)

	all_trades_df = result.all_trades_df
	summary_df = result.summary_df
	failures_df = result.failures_df
	symbol_chart_manifest = _build_symbol_chart_manifest(result, report_dir)
	best_symbols_by_pnl, worst_symbols_by_pnl = _metric_top_bottom_symbols(summary_df, "total_pnl")
	best_symbols_by_sharpe, worst_symbols_by_sharpe = _metric_top_bottom_symbols(summary_df, "sharpe_ratio")
	realized_returns = (
		all_trades_df.with_columns(
			pl.when(pl.col("entry_notional") > 0)
			.then(pl.col("pnl") / pl.col("entry_notional"))
			.otherwise(0.0)
			.alias("realized_return")
		)
		.get_column("realized_return")
		.cast(pl.Float64)
		.to_list()
		if all_trades_df.height > 0
		else []
	)

	# 这里把“逐股票结果”提升成“全市场结果”，供最终报告展示。
	overall_sharpe = _safe_sharpe(realized_returns) if all_trades_df.height > 0 else None
	overall_summary = {
		"date": result.trade_date,
		"repo_dir": str(result.repo_dir),
		"hold_minutes": result.hold_minutes,
		"n_symbols_completed": int(summary_df.height),
		"n_symbols_failed": int(failures_df.height),
		"n_trades": int(all_trades_df.height),
		"total_pnl": float(all_trades_df.get_column("pnl").cast(pl.Float64).sum()) if all_trades_df.height > 0 else 0.0,
		"avg_trade_pnl": float(all_trades_df.get_column("pnl").cast(pl.Float64).mean()) if all_trades_df.height > 0 else 0.0,
		"sharpe_ratio": overall_sharpe,
		"best_symbols_by_pnl": best_symbols_by_pnl,
		"worst_symbols_by_pnl": worst_symbols_by_pnl,
		"best_symbols_by_sharpe": best_symbols_by_sharpe,
		"worst_symbols_by_sharpe": worst_symbols_by_sharpe,
		"n_symbol_net_position_charts": int(symbol_chart_manifest.filter(pl.col("net_position_path").is_not_null()).height),
	}

	# 策略层已经落盘 summary/all_trades/failures，这里不再重复输出同类 CSV。
	symbol_chart_manifest.write_csv(report_dir / "symbol_chart_manifest.csv")
	(report_dir / "overall_summary.json").write_text(
		json.dumps(overall_summary, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	# 固定输出几类基础图表，后续如果要扩展报表，只需要在这一层继续追加。
	chart_paths = [
		report_dir / "cumulative_pnl.png",
		report_dir / "rank_total_pnl.png",
		report_dir / "rank_sharpe_ratio.png",
		report_dir / "pnl_distribution.png",
	]
	_plot_cumulative_pnl(all_trades_df, chart_paths[0])
	_plot_rank_chart(summary_df, "total_pnl", "Total PnL Top 10 / Bottom 10", chart_paths[1])
	_plot_rank_chart(summary_df, "sharpe_ratio", "Sharpe Top 10 / Bottom 10", chart_paths[2])
	_plot_pnl_distribution(summary_df, chart_paths[3])
	_write_html_report(report_dir, overall_summary, chart_paths, symbol_chart_manifest)
	return overall_summary