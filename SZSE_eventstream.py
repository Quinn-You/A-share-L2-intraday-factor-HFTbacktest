#!/usr/bin/env python3
"""
基于 SZL2_ORDER 和 SZL2_TRADE 的深交所事件流转换脚本。

规则：
1. 将 ORDER 和 TRADE 直接按统一顺序回放，不再单独按 SetID 分组处理。
2. 限价单进入虚拟订单簿；市价单也入簿，但价格固定取对手盘一档（买取最优卖、卖取最优买）；U 单在本方簿非空时按本方最优价进入簿，否则直接跳过。
3. 成交事件输出方式参考 version2：先按簿面扣减生成 depth_event，再输出一条带主动方方向的 TRADE_EVENT。
4. F 成交固定双边扣减：买卖两边只要能定位到订单状态都执行扣减；在簿内则同步更新 depth，不在簿内仅扣 remaining。
5. 撤单事件通过 TRADE.TradeType=4 表达，并输出一条 depth_event 更新对应价位聚合量。
6. 输出字段沿用 HFTBT_DTYPE: ev, exch_ts, local_ts, px, qty, order_id, ival, fval。

示例：
python /home/haoranyou/reconstruction/SZSE/SZSE_eventstream.py --symbol 002594 --date 2026-03-25
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEPTH_EVENT = np.uint64(0x1)
TRADE_EVENT = np.uint64(0x2)
BUY_EVENT = np.uint64(0x20000000)
SELL_EVENT = np.uint64(0x10000000)
EXCH_EVENT = np.uint64(0x80000000)
LOCAL_EVENT = np.uint64(0x40000000)

HFTBT_DTYPE = np.dtype([
	("ev", "<u8"),
	("exch_ts", "<i8"),
	("local_ts", "<i8"),
	("px", "<f8"),
	("qty", "<f8"),
	("order_id", "<u8"),
	("ival", "<i8"),
	("fval", "<f8"),
])

DATA_ROOT = Path("/home/haoranyou/shared_data")

_ORDER_MORNING_START_US = (9 * 60 + 15) * 60 * 1_000_000
_TRADE_MORNING_START_US = (9 * 60 + 15) * 60 * 1_000_000
_MORNING_END_US = (11 * 60 + 30) * 60 * 1_000_000
_AFTERNOON_START_US = (13 * 60) * 60 * 1_000_000
_AFTERNOON_END_US = (15 * 60) * 60 * 1_000_000
_OPEN_CALL_AUCTION_END_US = (9 * 60 + 30) * 60 * 1_000_000
_CLOSE_CALL_AUCTION_START_US = (14 * 60 + 57) * 60 * 1_000_000

ORDER_ACTION = tuple[str, int, int, int, int, np.uint64, str, float, float]
TRADE_ACTION = tuple[str, int, int, int, int, int, float, float, str]
MERGED_ACTION = ORDER_ACTION | TRADE_ACTION


@dataclass
class OrderState:
	side: np.uint64
	order_type: str
	ts_ns: int
	arrival_seq: int
	remaining_qty: float
	book_price: float | None
	in_book: bool
	displayed_in_depth: bool


def _make_row(
	ev: np.uint64,
	ts_ns: int,
	px: float,
	qty: float,
	order_id: int = 0,
	ival: int = 0,
	fval: float = 0.0,
) -> tuple:
	return (
		ev,
		np.int64(ts_ns),
		np.int64(ts_ns),
		float(px),
		float(qty),
		np.uint64(order_id),
		np.int64(ival),
		float(fval),
	)


def _make_depth_row(ts_ns: int, px: float, qty: float, side: np.uint64, order_id: int) -> tuple:
	return _make_row(EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | side, ts_ns, px, qty, order_id=order_id)


def _make_trade_row(ts_ns: int, px: float, qty: float, side: np.uint64, order_id: int) -> tuple:
	return _make_row(EXCH_EVENT | LOCAL_EVENT | TRADE_EVENT | side, ts_ns, px, qty, order_id=order_id)


def _time_of_day_us_from_ns(ts_ns: int) -> int:
	return (ts_ns // 1_000) % (24 * 3_600_000_000)


def _is_call_auction_trade(ts_ns: int) -> bool:
	tod_us = _time_of_day_us_from_ns(ts_ns)
	in_open_call = _TRADE_MORNING_START_US <= tod_us < _OPEN_CALL_AUCTION_END_US
	in_close_call = _CLOSE_CALL_AUCTION_START_US <= tod_us <= _AFTERNOON_END_US
	return in_open_call or in_close_call


def _time_of_day_us_expr(col: str = "TradingTime") -> pl.Expr:
	h = pl.col(col).dt.hour().cast(pl.Int64)
	m = pl.col(col).dt.minute().cast(pl.Int64)
	s = pl.col(col).dt.second().cast(pl.Int64)
	us = pl.col(col).dt.microsecond().cast(pl.Int64)
	return h * 3_600_000_000 + m * 60_000_000 + s * 1_000_000 + us


def _in_order_sessions_expr(col: str = "TradingTime") -> pl.Expr:
	tod = _time_of_day_us_expr(col)
	in_morning = (tod >= _ORDER_MORNING_START_US) & (tod <= _MORNING_END_US)
	in_afternoon = (tod >= _AFTERNOON_START_US) & (tod <= _AFTERNOON_END_US)
	return in_morning | in_afternoon


def _in_trade_sessions_expr(col: str = "TradingTime") -> pl.Expr:
	tod = _time_of_day_us_expr(col)
	in_morning = (tod >= _TRADE_MORNING_START_US) & (tod <= _MORNING_END_US)
	in_afternoon = (tod >= _AFTERNOON_START_US) & (tod <= _AFTERNOON_END_US)
	return in_morning | in_afternoon


def _load_szse_table(table: str, date: str, symbol: str) -> pl.DataFrame:
	path = DATA_ROOT / table / f"{date}.parquet"
	if not path.exists():
		raise FileNotFoundError(path)

	if table == "SZL2_TRADE":
		time_filter = _in_trade_sessions_expr("TradingTime")
	else:
		time_filter = _in_order_sessions_expr("TradingTime")

	return (
		pl.scan_parquet(path)
		.filter(pl.col("Symbol") == symbol)
		.filter(time_filter)
		.collect()
	)


def _normalize_order_type(value: object) -> str:
	if value is None:
		return ""
	return str(value).strip().upper()


def _normalize_trade_type(value: object) -> str:
	if value is None:
		return ""
	return str(value).strip().upper()


def _side_from_order_code(value: object) -> np.uint64 | None:
	text = "" if value is None else str(value).strip().upper()
	if text == "1":
		return BUY_EVENT
	if text == "2":
		return SELL_EVENT
	return None


def _pack_order_id(set_id: int, rec_id: int) -> int:
	return (int(set_id) << 48) | int(rec_id)


def sz_order_actions(df_order: pl.DataFrame) -> list[ORDER_ACTION]:
	actions: list[ORDER_ACTION] = []
	if len(df_order) == 0:
		return actions

	ts_ns_col = df_order.select(pl.col("TradingTime").dt.epoch("ns")).to_series().to_numpy()
	for idx, row in enumerate(df_order.iter_rows(named=True)):
		side = _side_from_order_code(row.get("OrderCode"))
		if side is None:
			continue

		order_type = _normalize_order_type(row.get("OrderType"))
		if order_type not in {"1", "2", "U"}:
			continue

		set_id = int(row["SetID"])
		rec_id = int(row["RecID"])
		order_id = _pack_order_id(set_id, rec_id)
		price = float(row["OrderPrice"]) if row["OrderPrice"] is not None else 0.0
		qty = float(row["OrderVolume"]) if row["OrderVolume"] is not None else 0.0
		actions.append((
			"order",
			set_id,
			rec_id,
			int(ts_ns_col[idx]),
			order_id,
			side,
			order_type,
			price,
			qty,
		))
	return actions


def sz_trade_actions(df_trade: pl.DataFrame) -> list[TRADE_ACTION]:
	actions: list[TRADE_ACTION] = []
	if len(df_trade) == 0:
		return actions

	ts_ns_col = df_trade.select(pl.col("TradingTime").dt.epoch("ns")).to_series().to_numpy()
	for idx, row in enumerate(df_trade.iter_rows(named=True)):
		trade_type = _normalize_trade_type(row.get("TradeType"))
		if trade_type not in {"F", "4"}:
			continue

		set_id = int(row["SetID"])
		rec_id = int(row["RecID"])
		trade_id = _pack_order_id(set_id, rec_id)
		buy_order_id = int(row["BuyOrderID"]) if row["BuyOrderID"] is not None else 0
		sell_order_id = int(row["SellOrderID"]) if row["SellOrderID"] is not None else 0
		price = float(row["TradePrice"]) if row["TradePrice"] is not None else 0.0
		qty = float(row["TradeVolume"]) if row["TradeVolume"] is not None else 0.0
		actions.append((
			"trade",
			set_id,
			rec_id,
			int(ts_ns_col[idx]),
			trade_id,
			buy_order_id,
			sell_order_id,
			price,
			qty,
			trade_type,
		))
	return actions


def _merge_actions(order_actions: list[ORDER_ACTION], trade_actions: list[TRADE_ACTION]) -> list[MERGED_ACTION]:
	merged: list[MERGED_ACTION] = []
	merged.extend(order_actions)
	merged.extend(trade_actions)
	merged.sort(key=lambda item: (item[2], item[3], 0 if item[0] == "order" else 1))
	return merged


def _update_price_level(levels: dict[float, float], price: float, delta: float) -> float:
	new_qty = levels.get(price, 0.0) + delta
	if new_qty <= 0:
		levels.pop(price, None)
		return 0.0
	levels[price] = new_qty
	return new_qty


def _best_same_side_price(levels: dict[float, float], side: np.uint64) -> float | None:
	if not levels:
		return None
	if side == BUY_EVENT:
		return max(levels)
	return min(levels)


def _book_price_for_trade(state: OrderState, fallback_price: float) -> float:
	if state.book_price is not None and state.book_price > 0:
		return state.book_price
	return fallback_price


def _reduce_order_remaining(state: OrderState | None, trade_qty: float) -> None:
	if state is None:
		return
	state.remaining_qty = max(0.0, state.remaining_qty - max(0.0, trade_qty))


def _apply_passive_fill(
	rows: list[tuple],
	state: OrderState,
	levels: dict[float, float],
	ts_ns: int,
	order_id: int,
	trade_qty: float,
) -> None:
	if not state.in_book or state.book_price is None:
		return

	fill_qty = min(state.remaining_qty, max(0.0, trade_qty))
	if fill_qty <= 0:
		return

	state.remaining_qty -= fill_qty
	aggregate_qty = _update_price_level(levels, state.book_price, -fill_qty)
	rows.append(_make_depth_row(ts_ns, state.book_price, aggregate_qty, state.side, order_id))
	if state.remaining_qty <= 0:
		state.in_book = False
		state.book_price = None
		state.displayed_in_depth = False


def _apply_trade_fill(
	rows: list[tuple],
	state: OrderState | None,
	levels: dict[float, float],
	ts_ns: int,
	order_id: int,
	trade_qty: float,
) -> None:
	if state is None or order_id == 0:
		return
	if state.in_book and state.displayed_in_depth:
		_apply_passive_fill(rows, state, levels, ts_ns, order_id, trade_qty)
		return
	_reduce_order_remaining(state, trade_qty)


def _infer_aggressor_side(
	buy_state: OrderState | None,
	sell_state: OrderState | None,
	trade_price: float,
) -> np.uint64:
	if buy_state is not None and sell_state is not None:
		if buy_state.arrival_seq > sell_state.arrival_seq:
			return BUY_EVENT
		if sell_state.arrival_seq > buy_state.arrival_seq:
			return SELL_EVENT

	if sell_state is not None:
		sell_price = _book_price_for_trade(sell_state, trade_price)
		if sell_price > 0 and np.isclose(trade_price, sell_price):
			return BUY_EVENT

	if buy_state is not None:
		buy_price = _book_price_for_trade(buy_state, trade_price)
		if buy_price > 0 and np.isclose(trade_price, buy_price):
			return SELL_EVENT

	return np.uint64(0)


def _actions_to_events_one_set(order_actions: list[ORDER_ACTION], trade_actions: list[TRADE_ACTION]) -> list[tuple]:
	rows: list[tuple] = []
	order_states: dict[int, OrderState] = {}
	bid_levels: dict[float, float] = {}
	ask_levels: dict[float, float] = {}

	for action in _merge_actions(order_actions, trade_actions):
		if action[0] == "order":
			_, set_id, rec_id, ts_ns, order_id, side, order_type, raw_price, qty = action
			if qty <= 0:
				continue

			state = OrderState(
				side=side,
				order_type=order_type,
				ts_ns=ts_ns,
				arrival_seq=rec_id,
				remaining_qty=qty,
				book_price=None,
				in_book=False,
				displayed_in_depth=False,
			)
			order_states[order_id] = state

			if order_type == "1":
				if side == BUY_EVENT:
					effective_price = min(ask_levels) if ask_levels else None
				else:
					effective_price = max(bid_levels) if bid_levels else None
				if effective_price is None or effective_price <= 0:
					state.remaining_qty = 0.0
					continue
				state.book_price = effective_price
				state.in_book = True
				state.displayed_in_depth = False
				continue
			elif order_type == "U":
				same_side_levels = bid_levels if side == BUY_EVENT else ask_levels
				effective_price = _best_same_side_price(same_side_levels, side)
				if effective_price is None or effective_price <= 0:
					state.remaining_qty = 0.0
					continue
			else:
				effective_price = raw_price
				if effective_price <= 0:
					state.remaining_qty = 0.0
					continue

			state.book_price = effective_price
			state.in_book = True
			state.displayed_in_depth = True
			levels = bid_levels if side == BUY_EVENT else ask_levels
			aggregate_qty = _update_price_level(levels, effective_price, qty)
			rows.append(_make_depth_row(ts_ns, effective_price, aggregate_qty, side, order_id))
			continue

		_, set_id, rec_id, ts_ns, trade_id, buy_rec_id, sell_rec_id, trade_price, trade_qty, trade_type = action
		buy_order_id = _pack_order_id(set_id, buy_rec_id) if buy_rec_id > 0 else 0
		sell_order_id = _pack_order_id(set_id, sell_rec_id) if sell_rec_id > 0 else 0
		buy_state = order_states.get(buy_order_id) if buy_order_id else None
		sell_state = order_states.get(sell_order_id) if sell_order_id else None

		if trade_type == "4":
			cancel_order_id = buy_order_id or sell_order_id
			state = order_states.get(cancel_order_id) if cancel_order_id else None
			if state is None:
				continue

			cancel_qty = min(state.remaining_qty, max(0.0, trade_qty))
			if cancel_qty <= 0:
				continue

			state.remaining_qty -= cancel_qty
			if state.in_book and state.book_price is not None and state.displayed_in_depth:
				levels = bid_levels if state.side == BUY_EVENT else ask_levels
				aggregate_qty = _update_price_level(levels, state.book_price, -cancel_qty)
				rows.append(_make_depth_row(ts_ns, state.book_price, aggregate_qty, state.side, cancel_order_id))
				if state.remaining_qty <= 0:
					state.in_book = False
					state.book_price = None
					state.displayed_in_depth = False
			elif state.remaining_qty <= 0:
				state.in_book = False
				state.book_price = None
				state.displayed_in_depth = False
			continue

		if trade_qty <= 0 or trade_price <= 0:
			continue

		aggressor_side = _infer_aggressor_side(buy_state, sell_state, trade_price)
		_apply_trade_fill(rows, buy_state, bid_levels, ts_ns, buy_order_id, trade_qty)
		_apply_trade_fill(rows, sell_state, ask_levels, ts_ns, sell_order_id, trade_qty)

		rows.append(_make_trade_row(ts_ns, trade_price, trade_qty, aggressor_side, trade_id))

	return rows


def actions_to_events(order_actions: list[ORDER_ACTION], trade_actions: list[TRADE_ACTION]) -> list[tuple]:
	rows = _actions_to_events_one_set(order_actions, trade_actions)
	rows.sort(key=lambda row: (row[1], row[2]))
	return rows


def convert_szse_symbol(symbol: str, date: str, outdir: Path) -> Path:
	logger.info("Converting SZSE %s %s", symbol, date)
	df_order = _load_szse_table("SZL2_ORDER", date, symbol)
	df_trade = _load_szse_table("SZL2_TRADE", date, symbol)
	logger.info("Rows after session filter ORDER=%d TRADE=%d", len(df_order), len(df_trade))

	events = actions_to_events(sz_order_actions(df_order), sz_trade_actions(df_trade))
	arr = np.array(events, dtype=HFTBT_DTYPE)

	outdir.mkdir(parents=True, exist_ok=True)
	out_path = outdir / f"{symbol}_{date}.npz"
	np.savez_compressed(out_path, data=arr)
	logger.info("Saved %d events to %s", len(arr), out_path)
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(description="SZSE tick to hft event stream converter (ORDER/TRADE only)")
	parser.add_argument("--symbol", required=True, help="SZSE symbol, e.g. 002594")
	parser.add_argument("--date", required=True, help="Trading date, e.g. 2026-03-26")
	parser.add_argument(
		"--outdir",
		default="/home/haoranyou/trade_system/output/pipeline_repo/2024-08-30/data_layer/eventstream/2024-08-30/szse",
		help="Output directory for generated npz",
	)
	args = parser.parse_args()
	convert_szse_symbol(args.symbol, args.date, Path(args.outdir))


if __name__ == "__main__":
	main()


'''PY="/home/haoranyou/miniconda3/envs/guyuefangyuan/bin/python"
SCRIPT="/home/haoranyou/trade_system/SZSE_eventstream.py"
CODES="/home/haoranyou/trade_system/check/unique_codes.csv"
DATE="2024-08-27"
OUTDIR="/home/haoranyou/trade_system/output/pipeline_repo/${DATE}/data_layer/eventstream/${DATE}/szse"
FAIL_FILE="${OUTDIR}/failed_symbols.txt"

mkdir -p "${OUTDIR}"
: > "${FAIL_FILE}"

export PY SCRIPT DATE OUTDIR FAIL_FILE
tail -n +2 "${CODES}" | tr -d '\r' | awk 'NF' | xargs -I{} -P 8 bash -lc '
symbol="{}"
out="${OUTDIR}/${symbol}_${DATE}.npz"

if [[ -f "${out}" ]]; then
  echo "[SKIP] ${symbol}"
  exit 0
fi

echo "[RUN ] ${symbol}"
if ! "${PY}" "${SCRIPT}" --symbol "${symbol}" --date "${DATE}" --outdir "${OUTDIR}"; then
  echo "${symbol}" >> "${FAIL_FILE}"
fi
'
'''
