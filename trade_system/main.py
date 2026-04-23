#!/usr/bin/env python3
"""Pipeline 总入口：负责串联数据层、策略层和报表层。

这个文件只做三件事：
1. 解析命令行参数。
2. 依次调用数据准备层、策略层、报表层。
3. 在终端打印一份简明的执行摘要。

它不关心某只股票如何生成 event stream，也不关心单笔交易如何撮合；
这些细节都下沉到各自层内处理。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_data_layer import _default_repo_dir
from pipeline_data_layer import _normalize_requested_symbols
from pipeline_data_layer import prepare_universe_data
from pipeline_report_layer import generate_report
from pipeline_strategy_layer import StrategyConfig
from pipeline_strategy_layer import run_universe_strategy


def _build_parser() -> argparse.ArgumentParser:
	"""统一定义完整 pipeline 的命令行参数。

	这里的参数会原样分发给下游三个层次，因此 main.py 是唯一的总入口。
	"""
	p = argparse.ArgumentParser(description="完整 pipeline：准备数据、运行策略、输出报表")
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
	p.add_argument("--roi-lb", type=float, default=0.1, help="hbt 价格回放下界")
	p.add_argument("--roi-ub", type=float, default=10000.0, help="hbt 价格回放上界")
	p.add_argument("--commission-rate", type=float, default=0.00015, help="双边手续费率")
	p.add_argument("--stamp-duty-rate", type=float, default=0.0005, help="印花税率（仅卖出）")
	p.add_argument("--symbols", type=str, default=None, help="可选，逗号分隔的股票代码")
	p.add_argument("--max-symbols", type=int, default=None, help="可选，仅运行前 N 只股票")
	p.add_argument("--repo-dir", type=Path, default=None, help="输出 repo 目录")
	p.add_argument("--force-regenerate", action="store_true", help="强制重建 event stream")
	return p


def main() -> None:
	"""执行完整流水线。

	调用顺序固定为：
	1. 数据层：准备 alpha 和 event stream。
	2. 策略层：逐股票回测并输出交易明细。
	3. 报表层：汇总结果并生成可视化报告。
	"""
	args = _build_parser().parse_args()
	repo_dir = args.repo_dir or _default_repo_dir(args.date)

	# 第一步：由数据准备层统一处理 alpha 和 event stream。
	bundle = prepare_universe_data(
		trade_date=args.date,
		alpha_path=args.alpha,
		repo_dir=repo_dir,
		requested_symbols=_normalize_requested_symbols(args.symbols),
		max_symbols=args.max_symbols,
		force_regenerate=args.force_regenerate,
	)

	# 第二步：策略层只负责消费数据并运行 hbt。
	strategy_result = run_universe_strategy(
		bundle,
		StrategyConfig(
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
		),
	)

	# 第三步：报表层只做汇总、图表和 HTML 报告。
	overall_summary = generate_report(strategy_result)

	print("=" * 72)
	print("完整 pipeline 完成")
	print(f"date              : {args.date}")
	print(f"repo_dir          : {repo_dir}")
	print(f"n_symbols_done    : {overall_summary['n_symbols_completed']}")
	print(f"n_symbols_failed  : {overall_summary['n_symbols_failed']}")
	print(f"n_trades          : {overall_summary['n_trades']}")
	print(f"total_pnl         : {overall_summary['total_pnl']:.8f}")
	print(f"sharpe_ratio      : {overall_summary['sharpe_ratio']}")
	print(f"best_by_pnl       : {', '.join(overall_summary['best_symbols_by_pnl'])}")
	print(f"worst_by_pnl      : {', '.join(overall_summary['worst_symbols_by_pnl'])}")
	print(f"best_by_sharpe    : {', '.join(overall_summary['best_symbols_by_sharpe'])}")
	print(f"worst_by_sharpe   : {', '.join(overall_summary['worst_symbols_by_sharpe'])}")
	print(f"symbol_pos_charts : {overall_summary['n_symbol_net_position_charts']}")
	print("=" * 72)


if __name__ == "__main__":
	main()