#!/usr/bin/env python3
"""策略层：消费数据准备层结果，逐股票运行 hftbacktest 策略。

这一层的定位是“计算层”：
1. 接收数据层已经准备好的 bundle。
2. 逐股票调用 pipeline_core 中的单票回测逻辑。
3. 输出交易明细和股票级 summary。

它不负责原始数据转换，也不负责图表渲染。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from pipeline_data_layer import PreparedUniverseBundle
from pipeline_data_layer import _default_repo_dir
from pipeline_data_layer import _normalize_requested_symbols
from pipeline_data_layer import prepare_universe_data
from pipeline_core import run_backtest_for_alpha_records
from pipeline_core import trades_to_dataframe


@dataclass
class StrategyConfig:
	"""策略层运行所需的全部参数配置。"""

	alpha_threshold: float
	hold_minutes: int
	notional: float
	lot_size: int
	step_ns: int
	tick_size: float
	order_latency_ns: int
	roi_lb: float
	roi_ub: float
	commission_rate: float
	stamp_duty_rate: float


@dataclass
class StrategyRunResult:
	"""策略层对外输出的统一结果对象。"""

	trade_date: str
	repo_dir: Path
	hold_minutes: int
	summary_df: pl.DataFrame
	all_trades_df: pl.DataFrame
	failures_df: pl.DataFrame


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="策略层：运行全市场 hbt 回测")
	p.add_argument("--date", type=str, required=True, help="交易日 YYYY-MM-DD")
	p.add_argument(
		"--alpha",
		type=Path,
		default=Path("/data/sihang/AlphaPROBETick/alpha_monthly"),
		help="alpha parquet 路径或 alpha_monthly 根目录",
	)
	p.add_argument("--threshold", type=float, default=0.3, help="alpha 开仓阈值")
	p.add_argument("--hold-min", type=int, default=5, help="固定持有分钟数")
	p.add_argument("--notional", type=float, default=1_000.0, help="每笔名义资金")
	p.add_argument("--lot-size", type=int, default=100, help="最小交易数")
	p.add_argument("--step-ns", type=int, default=10_000_000, help="引擎时间推进步长（纳秒）")
	p.add_argument("--tick-size", type=float, default=0.01, help="最小价格变动")
	p.add_argument("--order-latency-ns", type=int, default=0, help="单边下单延迟（纳秒）")
	p.add_argument("--roi-lb", type=float, default=1.0, help="hbt 价格回放下界")
	p.add_argument("--roi-ub", type=float, default=100.0, help="hbt 价格回放上界")
	p.add_argument("--commission-rate", type=float, default=0.00015, help="双边手续费率")
	p.add_argument("--stamp-duty-rate", type=float, default=0.0005, help="印花税率（仅卖出）")
	p.add_argument("--symbols", type=str, default=None, help="可选，逗号分隔的股票代码")
	p.add_argument("--max-symbols", type=int, default=None, help="可选，仅运行前 N 只股票")
	p.add_argument("--repo-dir", type=Path, default=None, help="输出 repo 目录")
	p.add_argument("--force-regenerate", action="store_true", help="强制重建 event stream")
	return p


def _safe_sharpe(realized_returns: list[float]) -> float | None:
	"""基于单笔真实收益率序列计算简单版 Sharpe；样本不足或标准差无效时返回 None。"""
	if len(realized_returns) < 2:
		return None
	arr = np.asarray(realized_returns, dtype=np.float64)
	std = float(arr.std(ddof=1))
	if not np.isfinite(std) or std <= 0:
		return None
	return float(math.sqrt(arr.size) * arr.mean() / std)


def _strategy_dir(repo_dir: Path) -> Path:
	"""策略层输出目录。"""
	return repo_dir / "strategy_layer"


def _build_config_from_args(args: argparse.Namespace) -> StrategyConfig:
	"""把命令行参数打包成 StrategyConfig，方便在层间传递。"""
	return StrategyConfig(
		alpha_threshold=args.threshold,
		hold_minutes=args.hold_min,
		notional=args.notional,
		lot_size=args.lot_size,
		step_ns=args.step_ns,
		tick_size=args.tick_size,
		order_latency_ns=args.order_latency_ns,
		roi_lb=args.roi_lb,
		roi_ub=args.roi_ub,
		commission_rate=args.commission_rate,
		stamp_duty_rate=args.stamp_duty_rate,
	)


def run_universe_strategy(bundle: PreparedUniverseBundle, config: StrategyConfig) -> StrategyRunResult:
	"""逐股票运行策略。

	这一层只做回测，不负责生成 event stream，也不负责图表和报表。
	每只股票独立执行，失败的股票会记录到 failures，而不会中断整个批次。
	"""

	strategy_dir = _strategy_dir(bundle.repo_dir)
	trades_dir = strategy_dir / "trades"
	strategy_dir.mkdir(parents=True, exist_ok=True)
	trades_dir.mkdir(parents=True, exist_ok=True)

	summary_rows: list[dict[str, object]] = []
	failure_rows: list[dict[str, str]] = []
	all_trade_frames: list[pl.DataFrame] = []

	# 逐股票调用单票回测内核，把结果落到 strategy_layer 目录下。
	for prepared in bundle.prepared_symbols:
		alpha_records = bundle.alpha_records_by_symbol[prepared.symbol]
		try:
			trades, summary = run_backtest_for_alpha_records(
				eventstream_path=prepared.eventstream_path,
				alpha_records=alpha_records,
				alpha_threshold=config.alpha_threshold,
				hold_minutes=config.hold_minutes,
				notional=config.notional,
				lot_size=config.lot_size,
				step_ns=config.step_ns,
				tick_size=config.tick_size,
				order_latency_ns=config.order_latency_ns,
				roi_lb=config.roi_lb,
				roi_ub=config.roi_ub,
				commission_rate=config.commission_rate,
				stamp_duty_rate=config.stamp_duty_rate,
			)
		except Exception as exc:
			failure_rows.append({"symbol": prepared.symbol, "reason": f"策略执行失败: {exc}"})
			continue

		trades_df = trades_to_dataframe(trades)
		trades_path = trades_dir / f"{prepared.symbol}_{bundle.trade_date}.csv"
		trades_df.write_csv(trades_path)

		# all_trades 是跨股票汇总表，所以这里补一列 symbol，便于后面报表聚合。
		if trades_df.height > 0:
			all_trade_frames.append(trades_df.with_columns(pl.lit(prepared.symbol).alias("symbol")))

		realized_return_values = (
			trades_df.with_columns(
				pl.when(pl.col("entry_notional") > 0)
				.then(pl.col("pnl") / pl.col("entry_notional"))
				.otherwise(0.0)
				.alias("realized_return")
			)
			.get_column("realized_return")
			.cast(pl.Float64)
			.to_list()
			if trades_df.height > 0
			else []
		)
		summary_rows.append(
			{
				"symbol": prepared.symbol,
				"market": prepared.market,
				"eventstream_path": str(prepared.eventstream_path),
				"trades_path": str(trades_path),
				"alpha_points": prepared.alpha_points,
				"n_trades": int(summary["n_trades"]),
				"n_long": int(summary["n_long"]),
				"n_short": int(summary["n_short"]),
				"win_rate": float(summary["win_rate"]),
				"total_pnl": float(summary["total_pnl"]),
				"avg_pnl": float(summary["avg_pnl"]),
				"avg_ret": float(summary["avg_ret"]),
				"sharpe_ratio": _safe_sharpe(realized_return_values),
			}
		)

	all_trades_df = pl.concat(all_trade_frames, how="diagonal_relaxed") if all_trade_frames else pl.DataFrame(
		{
			"symbol": [],
			"side": [],
			"entry_time": [],
			"exit_time": [],
			"entry_px": [],
			"exit_px": [],
			"shares": [],
			"entry_notional": [],
			"exit_notional": [],
			"pnl_gross": [],
			"commission": [],
			"stamp_duty": [],
			"total_cost": [],
			"pnl": [],
			"ret": [],
			"alpha_signal": [],
		}
	)
	summary_df = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame(
		{
			"symbol": [],
			"market": [],
			"eventstream_path": [],
			"trades_path": [],
			"alpha_points": [],
			"n_trades": [],
			"n_long": [],
			"n_short": [],
			"win_rate": [],
			"total_pnl": [],
			"avg_pnl": [],
			"avg_ret": [],
			"sharpe_ratio": [],
		}
	)
	failures_df = pl.DataFrame(failure_rows or {"symbol": [], "reason": []})

	# 先把结果落盘，再把内存中的 DataFrame 返回给上层继续汇总。
	summary_df.sort("total_pnl", descending=True).write_csv(strategy_dir / "symbol_summary.csv")
	all_trades_df.write_csv(strategy_dir / "all_trades.csv")
	failures_df.write_csv(strategy_dir / "strategy_failures.csv")

	return StrategyRunResult(
		trade_date=bundle.trade_date,
		repo_dir=bundle.repo_dir,
		hold_minutes=config.hold_minutes,
		summary_df=summary_df,
		all_trades_df=all_trades_df,
		failures_df=failures_df,
	)


def main() -> None:
	"""支持独立执行策略层，方便只做数据准备 + 回测，不生成报表。"""
	args = _build_parser().parse_args()
	repo_dir = args.repo_dir or _default_repo_dir(args.date)
	bundle = prepare_universe_data(
		trade_date=args.date,
		alpha_path=args.alpha,
		repo_dir=repo_dir,
		requested_symbols=_normalize_requested_symbols(args.symbols),
		max_symbols=args.max_symbols,
		force_regenerate=args.force_regenerate,
	)
	result = run_universe_strategy(bundle, _build_config_from_args(args))

	print("=" * 72)
	print("策略执行完成")
	print(f"date              : {result.trade_date}")
	print(f"repo_dir          : {result.repo_dir}")
	print(f"summary_rows      : {result.summary_df.height}")
	print(f"all_trade_rows    : {result.all_trades_df.height}")
	print(f"failure_rows      : {result.failures_df.height}")
	print("=" * 72)


if __name__ == "__main__":
	main()