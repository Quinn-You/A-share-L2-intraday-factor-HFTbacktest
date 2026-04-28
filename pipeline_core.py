#!/usr/bin/env python3
"""Pipeline 内核模块：沉淀从 alpha_bt 拆出来的通用策略能力。

这个文件不直接面向用户，而是给其他 pipeline 层提供稳定能力，主要包括：
1. alpha 文件定位与读取。
2. 时间字段与回测时间戳转换。
3. hftbacktest 引擎初始化。
4. 单只股票的策略主循环。
5. 成交结果的汇总与表格化。

设计上，这里是替代 alpha_bt 的“模板内核”，这样后续即使删除 alpha_bt，
数据层、策略层、报表层仍然可以独立运行。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np
import polars as pl


# 这里复制 alpha_bt 的底层依赖和回测模板，保证 pipeline 后续不依赖 alpha_bt 文件本身。
try:
	from hftbacktest import BUY_EVENT
	from hftbacktest import DEPTH_EVENT
	from hftbacktest import GTC
	from hftbacktest import MARKET
	from hftbacktest import SELL_EVENT
	from hftbacktest import BacktestAsset
	from hftbacktest import ROIVectorMarketDepthBacktest
except Exception:
	GTC = None
	MARKET = None
	BacktestAsset = None
	ROIVectorMarketDepthBacktest = None
	DEPTH_EVENT = np.uint64(0x1)
	BUY_EVENT = np.uint64(0x20000000)
	SELL_EVENT = np.uint64(0x10000000)


@dataclass
class Trade:
	"""单笔已完成交易的标准结果结构。

	策略层和报表层都依赖这个统一结构，避免各层重复定义字段。
	"""

	side: str
	entry_ts_ns: int
	exit_ts_ns: int
	entry_px: float
	exit_px: float
	shares: int
	entry_notional: float
	exit_notional: float
	pnl_gross: float
	commission: float
	stamp_duty: float
	total_cost: float
	pnl: float
	ret: float
	alpha_signal: float


@dataclass
class OpenPosition:
	"""策略主循环中的持仓状态。

	它只在回测过程中存在，用于记录开仓后、平仓前的临时状态。
	"""

	side: int
	shares: int
	alpha_signal: float
	entry_order_id: int
	entry_ts_ns: int
	entry_px: float
	target_exit_ts_ns: int
	exit_order_id: int | None = None


def to_ns_utc(date_str: str, hhmmss: str) -> int:
	"""把 YYYY-MM-DD + HH:MM:SS 转成 UTC 纳秒时间戳。"""
	dt = datetime.strptime(f"{date_str} {hhmmss}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
	return int(dt.timestamp() * 1_000_000_000)


def hold_target_ts_excluding_lunch(entry_ts_ns: int, hold_minutes: int) -> int:
	"""计算平仓目标时间：11:30-13:00 午休不计入持仓时间。"""
	hold_ns = int(hold_minutes * 60 * 1_000_000_000)
	raw_target = entry_ts_ns + hold_ns

	entry_dt = datetime.fromtimestamp(entry_ts_ns / 1e9, tz=timezone.utc)
	lunch_start = entry_dt.replace(hour=11, minute=30, second=0, microsecond=0)
	lunch_end = entry_dt.replace(hour=13, minute=0, second=0, microsecond=0)
	lunch_start_ns = int(lunch_start.timestamp() * 1_000_000_000)
	lunch_end_ns = int(lunch_end.timestamp() * 1_000_000_000)

	if entry_ts_ns < lunch_start_ns and raw_target > lunch_start_ns:
		return raw_target + (lunch_end_ns - lunch_start_ns)
	return raw_target


def ns_to_text(ts_ns: int) -> str:
	"""把纳秒时间戳转为报表可读的文本时间。"""
	return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def resolve_alpha_path(alpha_path: Path, trade_date: str) -> Path:
	"""兼容直接传 parquet 文件或传 alpha_monthly 根目录两种用法。"""
	if alpha_path.is_file():
		return alpha_path

	target_name = f"{trade_date}.parquet"
	search_root = alpha_path.parent if alpha_path.suffix == ".parquet" else alpha_path
	if search_root.name != "alpha_monthly":
		for parent in [search_root, *search_root.parents]:
			if parent.name == "alpha_monthly":
				search_root = parent
				break

	if search_root.exists():
		matches = sorted(search_root.rglob(target_name))
		if len(matches) == 1:
			return matches[0]
		if len(matches) > 1:
			raise ValueError(f"找到多个 alpha 文件，请手工指定更精确路径：{[str(item) for item in matches[:5]]}")

	raise FileNotFoundError(f"未找到 alpha 文件：{alpha_path}，且在 {search_root} 下也未找到 {target_name}")


def load_alpha_records_grouped(
	alpha_path: Path,
	trade_date: str,
	symbols: Iterable[str] | None = None,
) -> dict[str, list[tuple[int, float]]]:
	"""一次性读取当日 alpha，并按股票分组。"""
	filters = None
	if symbols is not None:
		symbol_list = sorted({str(symbol) for symbol in symbols if symbol is not None})
		if len(symbol_list) == 0:
			return {}
		filters = pl.col("symbol").is_in(symbol_list)

	query = pl.read_parquet(alpha_path)
	if filters is not None:
		query = query.filter(filters)

	df = query.select(["symbol", "time", "alpha"])
	if df.height == 0:
		return {}

	records_by_symbol: dict[str, list[tuple[int, float]]] = defaultdict(list)
	for symbol, time_text, alpha_value in df.iter_rows():
		if symbol is None or time_text is None or alpha_value is None:
			continue
		records_by_symbol[str(symbol)].append((to_ns_utc(trade_date, str(time_text)), float(alpha_value)))

	for records in records_by_symbol.values():
		records.sort(key=lambda item: item[0])
	return dict(records_by_symbol)


def build_hbt(
	eventstream_path: Path,
	lot_size: int,
	tick_size: float,
	order_latency_ns: int,
	roi_lb: float,
	roi_ub: float,
) -> Any:
	"""根据 event stream 和撮合参数构造单资产回测引擎。"""

	if BacktestAsset is None or ROIVectorMarketDepthBacktest is None:
		raise RuntimeError("当前环境未安装 hftbacktest，无法使用引擎撮合回测")

	asset = (
		BacktestAsset()
		.data(str(eventstream_path))
		.linear_asset(1.0)
		.constant_order_latency(order_latency_ns, order_latency_ns)
		.power_prob_queue_model(2.0)
		.no_partial_fill_exchange()
		.tick_size(tick_size)
		.lot_size(lot_size)
		.roi_lb(roi_lb)
		.roi_ub(roi_ub)
	)
	return ROIVectorMarketDepthBacktest([asset])


def is_order_filled(order: object) -> bool:
	"""判断订单是否已经完全成交。"""
	return (order is not None) and (float(order.leaves_qty) <= 0.0) and (float(order.exec_qty) > 0.0)


def submit_entry_order(
	hbt: Any,
	order_id: int,
	side: int,
	shares: int,
) -> None:
	"""按方向提交市价开仓单。"""
	if side > 0:
		hbt.submit_buy_order(0, order_id, 0.0, float(shares), GTC, MARKET, False)
	else:
		hbt.submit_sell_order(0, order_id, 0.0, float(shares), GTC, MARKET, False)


def submit_exit_order(
	hbt: Any,
	order_id: int,
	pos: OpenPosition,
) -> None:
	"""针对现有持仓提交对手方向的市价平仓单。"""
	if pos.side > 0:
		hbt.submit_sell_order(0, order_id, 0.0, float(pos.shares), GTC, MARKET, False)
	else:
		hbt.submit_buy_order(0, order_id, 0.0, float(pos.shares), GTC, MARKET, False)


def summarize_trades(trades: Iterable[Trade]) -> dict[str, float]:
	"""把交易列表压缩成策略层和报表层可直接使用的摘要指标。"""
	trade_list = list(trades)
	if not trade_list:
		return {
			"n_trades": 0,
			"n_long": 0,
			"n_short": 0,
			"win_rate": 0.0,
			"total_pnl": 0.0,
			"avg_pnl": 0.0,
			"avg_ret": 0.0,
		}

	pnls = np.array([item.pnl for item in trade_list], dtype=np.float64)
	rets = np.array([item.ret for item in trade_list], dtype=np.float64)
	sides = [item.side for item in trade_list]
	return {
		"n_trades": int(len(trade_list)),
		"n_long": int(sum(1 for side in sides if side == "long")),
		"n_short": int(sum(1 for side in sides if side == "short")),
		"win_rate": float((pnls > 0).mean()),
		"total_pnl": float(pnls.sum()),
		"avg_pnl": float(pnls.mean()),
		"avg_ret": float(rets.mean()),
	}


def trades_to_dataframe(trades: list[Trade]) -> pl.DataFrame:
	"""把内部 Trade 对象列表转换成可落盘的 Polars DataFrame。"""
	rows = []
	for trade in trades:
		row = asdict(trade)
		row["entry_time"] = ns_to_text(trade.entry_ts_ns)
		row["exit_time"] = ns_to_text(trade.exit_ts_ns)
		rows.append(row)
	if not rows:
		return pl.DataFrame(
			{
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
	return pl.DataFrame(rows).select(
		[
			"side",
			"entry_time",
			"exit_time",
			"entry_px",
			"exit_px",
			"shares",
			"entry_notional",
			"exit_notional",
			"pnl_gross",
			"commission",
			"stamp_duty",
			"total_cost",
			"pnl",
			"ret",
			"alpha_signal",
		]
	)


def run_backtest_for_alpha_records(
	eventstream_path: Path,
	alpha_records: list[tuple[int, float]],
	alpha_threshold: float,
	hold_minutes: int,
	notional: float,
	lot_size: int,
	step_ns: int,
	tick_size: float,
	order_latency_ns: int,
	roi_lb: float,
	roi_ub: float,
	commission_rate: float,
	stamp_duty_rate: float,
) -> tuple[list[Trade], dict[str, float]]:
	"""执行单只股票的回测主循环。

	输入：
	- eventstream_path: 该股票该交易日的盘口事件流。
	- alpha_records: [(信号时间, alpha 值)] 序列，且已按时间排序。
	- 其余参数均为回测和成本模型参数。

	输出：
	- trades: 单笔交易明细。
	- summary: 从 trades 进一步汇总出的摘要统计。
	"""

	if MARKET is None:
		raise RuntimeError("当前 hftbacktest 版本不支持 MARKET 订单类型")
	if commission_rate < 0 or stamp_duty_rate < 0:
		raise ValueError("commission_rate 和 stamp_duty_rate 不能为负数")
	if len(alpha_records) == 0:
		return [], summarize_trades([])

	hbt = build_hbt(
		eventstream_path=eventstream_path,
		lot_size=lot_size,
		tick_size=tick_size,
		order_latency_ns=order_latency_ns,
		roi_lb=roi_lb,
		roi_ub=roi_ub,
	)

	trades: list[Trade] = []
	open_positions: list[OpenPosition] = []
	pending_entry: dict[int, tuple[int, float, int]] = {}
	pending_exit: dict[int, int] = {}
	next_signal_idx = 0
	next_order_id = 1

	try:
		while True:
			# 推进撮合引擎到下一个离散时间点，并读取当前盘口快照。
			status = hbt.elapse(step_ns)
			now_ts = int(hbt.current_timestamp)
			depth = hbt.depth(0)

			# 只有买一卖一都有效时，才允许根据信号开仓或根据时间平仓。
			has_book = (
				np.isfinite(depth.best_bid)
				and np.isfinite(depth.best_ask)
				and float(depth.best_bid) > 0
				and float(depth.best_ask) > 0
				and float(depth.best_ask) >= float(depth.best_bid)
			)

			# 先检查之前挂出的开仓单是否已经成交；成交后把状态转成 open_positions。
			filled_entry_ids: list[int] = []
			for oid, (side, alpha_val, shares) in pending_entry.items():
				order = hbt.orders(0).get(oid)
				if not is_order_filled(order):
					continue
				entry_ts_ns = int(order.exch_timestamp)
				entry_px = float(order.exec_price)
				open_positions.append(
					OpenPosition(
						side=side,
						shares=shares,
						alpha_signal=alpha_val,
						entry_order_id=oid,
						entry_ts_ns=entry_ts_ns,
						entry_px=entry_px,
						target_exit_ts_ns=hold_target_ts_excluding_lunch(entry_ts_ns, hold_minutes),
					)
				)
				filled_entry_ids.append(oid)
			for oid in filled_entry_ids:
				pending_entry.pop(oid, None)

			# 再检查持仓是否到达目标持有时长；到时就发送市价平仓单。
			if has_book:
				for idx, pos in enumerate(open_positions):
					if pos.exit_order_id is not None or now_ts < pos.target_exit_ts_ns:
						continue
					oid = next_order_id
					next_order_id += 1
					submit_exit_order(hbt, oid, pos)
					pos.exit_order_id = oid
					pending_exit[oid] = idx

			# 平仓单成交后，生成最终 Trade，并在这里统一结算手续费、印花税和收益率。
			filled_exit_ids: list[int] = []
			for oid, pos_idx in pending_exit.items():
				order = hbt.orders(0).get(oid)
				if not is_order_filled(order):
					continue
				pos = open_positions[pos_idx]
				exit_ts_ns = int(order.exch_timestamp)
				exit_px = float(order.exec_price)
				entry_notional = float(pos.entry_px * pos.shares)
				exit_notional = float(exit_px * pos.shares)
				forward_return = (exit_px - pos.entry_px) / pos.entry_px if pos.entry_px > 0 else 0.0
				pnl_gross = pos.side * (exit_px - pos.entry_px) * pos.shares
				commission = (entry_notional + exit_notional) * commission_rate
				sell_notional = exit_notional if pos.side > 0 else entry_notional
				stamp_duty = sell_notional * stamp_duty_rate
				total_cost = commission + stamp_duty
				pnl = pnl_gross - total_cost
				ret = float(forward_return)
				trades.append(
					Trade(
						side="long" if pos.side > 0 else "short",
						entry_ts_ns=pos.entry_ts_ns,
						exit_ts_ns=exit_ts_ns,
						entry_px=pos.entry_px,
						exit_px=exit_px,
						shares=pos.shares,
						entry_notional=entry_notional,
						exit_notional=exit_notional,
						pnl_gross=float(pnl_gross),
						commission=float(commission),
						stamp_duty=float(stamp_duty),
						total_cost=float(total_cost),
						pnl=float(pnl),
						ret=float(ret),
						alpha_signal=pos.alpha_signal,
					)
				)
				filled_exit_ids.append(oid)
			for oid in filled_exit_ids:
				pending_exit.pop(oid, None)

			# 消费当前时点之前已经到达的 alpha 信号，并尝试根据阈值开仓。
			while next_signal_idx < len(alpha_records) and alpha_records[next_signal_idx][0] <= now_ts:
				sig_ts_ns, alpha_val = alpha_records[next_signal_idx]
				next_signal_idx += 1
				side = 0
				if alpha_val >= alpha_threshold:
					side = 1
				elif alpha_val <= -alpha_threshold:
					side = -1
				if side == 0 or not has_book:
					continue

				entry_ref_px = float(depth.best_ask) if side > 0 else float(depth.best_bid)
				shares = int((notional // entry_ref_px) // lot_size) * lot_size
				if shares < lot_size:
					continue

				oid = next_order_id
				next_order_id += 1
				submit_entry_order(hbt, oid, side, shares)
				pending_entry[oid] = (side, alpha_val, shares)

			# 回放结束后，只要后续没有未消费信号、未成交开仓单和未完成平仓流程，就退出。
			if status != 0:
				all_exited = all(pos.exit_order_id is not None for pos in open_positions) and len(pending_exit) == 0
				if next_signal_idx >= len(alpha_records) and len(pending_entry) == 0 and all_exited:
					break
				break

			hbt.clear_inactive_orders(0)
	finally:
		hbt.close()

	return trades, summarize_trades(trades)
