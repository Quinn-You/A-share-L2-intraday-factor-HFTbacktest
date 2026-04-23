#!/usr/bin/env python3
"""
基于 ORDER 和 TRANSACTION 的 SSE 事件流转换脚本。

规则：
1. 从空簿开始，不再使用 TAQ 参与构建。
2. ORDER 取 09:15-11:30、13:00-14:57，TRANSACTION 取 09:25-11:30、13:00-14:57。
3. ORDER 和 TRANSACTION 都先转成 action，再按共用 RecID 全局升序回放。
4. ORDER 更新订单余额，TX 同时打印成交并根据orderid扣减被动方，集合竞价期间的N单扣减双方。
python SSE_eventstream.py  --symbol 600519 --date 2024-08-30
"""
'''
order:
如果这个 order_id 以前不存在：
直接写入 order_book
然后把对应价位的聚合量加上 new_qty
再输出一个 DEPTH_EVENT
如果这个 order_id 已存在，而且 side 和 px 没变：
认为是“同一订单剩余量更新”
只把价位聚合量加上 delta = new_qty - prev_qty
再输出一个 DEPTH_EVENT
如果这个 order_id 已存在，但 side 或 px 变了：
先把旧价位的 prev_qty 从聚合量里扣掉，输出一个旧价位 DEPTH_EVENT
再把新价位 new_qty 加进去，输出一个新价位 DEPTH_EVENT
所以这里的“挂单”其实不只是新单，也包含同一个 order_id 的改价、改量。
cancel:
找到旧订单
从对应价位聚合量里扣掉该订单剩余量
输出一个 DEPTH_EVENT，qty 是撤单后的该价位剩余聚合量
从 order_book 删除这个 order_id
也就是说，撤单不会生成单独的 cancel event；它只体现在一个深度更新事件里。
transaction:
先在 order_book 里找这个 order_id
找不到就跳过
side 不匹配也跳过
如果找到，就把剩余量减去 min(qty, trade_qty)
然后继续走 _set_order_state 做真正的订单簿更新
所以一笔成交最多会生成 3 个事件：

买侧一个 DEPTH_EVENT
卖侧一个 DEPTH_EVENT
一个 TRADE_EVENT
'''

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEPTH_EVENT = np.uint64(0x1)
TRADE_EVENT = np.uint64(0x2)
'''
从空簿直接开始建立,所以不用clear和snapshot事件了
DEPTH_CLEAR_EVENT = np.uint64(0x3)
DEPTH_SNAPSHOT_EVENT = np.uint64(0x4)
'''
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

_ORDER_MORNING_START_US = (9 * 60 + 14) * 60 * 1_000_000
_TX_MORNING_START_US = (9 * 60 + 25) * 60 * 1_000_000
_MORNING_END_US = (11 * 60 + 30) * 60 * 1_000_000
_AFTERNOON_START_US = (13 * 60) * 60 * 1_000_000
#这里多加了一秒钟，确保重建的订单簿包含尾盘集合竞价的最后一条taq信息
_AFTERNOON_END_US = (15 * 60 + 1) * 60 * 1_000_000

ORDER_ACTION = tuple[int, int, int, int, np.uint64, float, float, str]
TX_ACTION = tuple[int, int, int, int, int, np.uint64, float, tuple]
MERGED_ACTION = tuple[str, tuple]


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

def _in_transaction_sessions_expr(col: str = "TradingTime") -> pl.Expr:
    tod = _time_of_day_us_expr(col)
    in_morning = (tod >= _TX_MORNING_START_US) & (tod <= _MORNING_END_US)
    in_afternoon = (tod >= _AFTERNOON_START_US) & (tod <= _AFTERNOON_END_US)
    return in_morning | in_afternoon

def _load_sse_table(table: str, date: str, symbol: str) -> pl.DataFrame:
    path = DATA_ROOT / table / f"{date}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    time_filter = _in_transaction_sessions_expr("TradingTime") if table == "SEL2_TRANSACTION" else _in_order_sessions_expr("TradingTime")
    return (
        pl.scan_parquet(path)
        .filter(pl.col("Symbol") == symbol)
        .filter(time_filter)
        .collect()
    )


def _make_depth_row(ts_ns: int, px: float, qty: float, side: np.uint64, order_id: int = 0) -> tuple:
    return _make_row(EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | side, ts_ns, px, qty, order_id=order_id)


def se_order_actions(df_order: pl.DataFrame) -> list[ORDER_ACTION]:
    actions: list[ORDER_ACTION] = []
    if len(df_order) == 0:
        return actions

    ts_ns_col = df_order.select(pl.col("TradingTime").dt.epoch("ns")).to_series().to_numpy()
    for i, row in enumerate(df_order.iter_rows(named=True)):
        order_type = row["OrderType"]
        if order_type == "A":
            action = "set"
        elif order_type == "D":
            action = "cancel"
        else:
            continue

        order_code = row["OrderCode"]
        if order_code == "B":
            side = BUY_EVENT
        elif order_code == "S":
            side = SELL_EVENT
        else:
            continue

        rec_id = int(row["RecID"]) if row["RecID"] is not None else -1
        order_id = int(row["OrderID"]) if row["OrderID"] and row["OrderID"] > 0 else rec_id
        px = float(row["OrderPrice"]) if row["OrderPrice"] is not None else 0.0
        qty = float(row["Balance"]) if row["Balance"] is not None else 0.0
        actions.append((int(ts_ns_col[i]), rec_id, i, order_id, side, px, qty, action))
    return actions


def se_transaction_actions(df_tx: pl.DataFrame) -> list[TX_ACTION]:
    flag_map = {"B": BUY_EVENT, "S": SELL_EVENT, "N": np.uint64(0)}
    actions: list[TX_ACTION] = []
    if len(df_tx) == 0:
        return actions

    ts_ns_col = df_tx.select(pl.col("TradingTime").dt.epoch("ns")).to_series().to_numpy()
    for i, row in enumerate(df_tx.iter_rows(named=True)):
        px = row["TradePrice"]
        qty = row["TradeVolume"]
        if px is None or px <= 0 or qty is None or qty <= 0:
            continue

        rec_id = int(row["RecID"]) if row["RecID"] is not None else -1
        side = flag_map.get(row.get("BuySellFlag", "N"), np.uint64(0))
        buy_order_id = int(row["BuyRecID"]) if row["BuyRecID"] is not None and row["BuyRecID"] > 0 else 0
        sell_order_id = int(row["SellRecID"]) if row["SellRecID"] is not None and row["SellRecID"] > 0 else 0
        trade_row = _make_row(EXCH_EVENT | LOCAL_EVENT | TRADE_EVENT | side, int(ts_ns_col[i]), float(px), float(qty), order_id=rec_id)
        actions.append((int(ts_ns_col[i]), rec_id, i, buy_order_id, sell_order_id, side, float(qty), trade_row))
    return actions


def _merge_actions_by_recid(order_actions: list[ORDER_ACTION], tx_actions: list[TX_ACTION]) -> list[MERGED_ACTION]:
    merged: list[MERGED_ACTION] = [("order", action) for action in order_actions]
    merged.extend(("tx", action) for action in tx_actions)
    merged.sort(key=lambda item: item[1][1])
    return merged


def _update_price_level(levels: dict[float, float], price: float, delta: float) -> float:
    new_qty = levels.get(price, 0.0) + delta
    if new_qty <= 0:
        levels.pop(price, None)
        return 0.0
    levels[price] = new_qty
    return new_qty


def _set_order_state(
    rows: list[tuple],
    order_book: dict[int, tuple[np.uint64, float, float]],
    bid_levels: dict[float, float],
    ask_levels: dict[float, float],
    ts_ns: int,
    order_id: int,
    side: np.uint64 | None,
    px: float | None,
    new_qty: float,
) -> None:
    prev = order_book.get(order_id)
    #处理orderid不是第一次出现的情况
    if prev is not None:
        prev_side, prev_px, prev_qty = prev
        if new_qty <= 0:
            levels = bid_levels if prev_side == BUY_EVENT else ask_levels
            agg_qty = _update_price_level(levels, prev_px, -prev_qty)
            rows.append(_make_depth_row(ts_ns, prev_px, agg_qty, prev_side, order_id=order_id))
            order_book.pop(order_id, None)
            return

        if side == prev_side and px == prev_px:
            if new_qty != prev_qty:
                levels = bid_levels if side == BUY_EVENT else ask_levels
                agg_qty = _update_price_level(levels, px, new_qty - prev_qty)
                rows.append(_make_depth_row(ts_ns, px, agg_qty, side, order_id=order_id))
                order_book[order_id] = (side, px, new_qty)
            return

        prev_levels = bid_levels if prev_side == BUY_EVENT else ask_levels
        prev_agg = _update_price_level(prev_levels, prev_px, -prev_qty)
        rows.append(_make_depth_row(ts_ns, prev_px, prev_agg, prev_side, order_id=order_id))

    if side is None or px is None or new_qty <= 0 or px <= 0:
        order_book.pop(order_id, None)
        return
    #这里是orderid第一次出现的情况
    order_book[order_id] = (side, px, new_qty)
    levels = bid_levels if side == BUY_EVENT else ask_levels
    agg_qty = _update_price_level(levels, px, new_qty)
    rows.append(_make_depth_row(ts_ns, px, agg_qty, side, order_id=order_id))


def _reduce_order_by_trade(
    rows: list[tuple],
    order_book: dict[int, tuple[np.uint64, float, float]],
    bid_levels: dict[float, float],
    ask_levels: dict[float, float],
    ts_ns: int,
    order_id: int,
    expected_side: np.uint64,
    trade_qty: float,
) -> None:
    order = order_book.get(order_id)
    if order is None:
        return

    side, px, qty = order
    if side != expected_side:
        return

    remaining = qty - min(qty, trade_qty)
    _set_order_state(rows, order_book, bid_levels, ask_levels, ts_ns, order_id, side, px, remaining)


def actions_to_events(order_actions: list[ORDER_ACTION], tx_actions: list[TX_ACTION]) -> list[tuple]:
    rows: list[tuple] = []
    order_book: dict[int, tuple[np.uint64, float, float]] = {}
    bid_levels: dict[float, float] = {}
    ask_levels: dict[float, float] = {}

    for kind, payload in _merge_actions_by_recid(order_actions, tx_actions):
        if kind == "order":
            ts_ns, _, _, order_id, side, px, qty, action = payload
            if action == "cancel":
                #cancel就置零
                _set_order_state(rows, order_book, bid_levels, ask_levels, ts_ns, order_id, None, None, 0.0)
            else:
                #order就挂单或者修改订单数目
                _set_order_state(rows, order_book, bid_levels, ask_levels, ts_ns, order_id, side, px, qty)
            continue
       
        ts_ns, _, _, buy_order_id, sell_order_id, aggressor_side, trade_qty, trade_row = payload
        # B/S 只扣减被动方，N 扣减双方
        if aggressor_side == BUY_EVENT:
            _reduce_order_by_trade(rows, order_book, bid_levels, ask_levels, ts_ns, sell_order_id, SELL_EVENT, trade_qty)
        elif aggressor_side == SELL_EVENT:
            _reduce_order_by_trade(rows, order_book, bid_levels, ask_levels, ts_ns, buy_order_id, BUY_EVENT, trade_qty)
        else:
            _reduce_order_by_trade(rows, order_book, bid_levels, ask_levels, ts_ns, buy_order_id, BUY_EVENT, trade_qty)
            _reduce_order_by_trade(rows, order_book, bid_levels, ask_levels, ts_ns, sell_order_id, SELL_EVENT, trade_qty)
        rows.append(trade_row)

    return rows


def convert_sse_symbol(symbol: str, date: str, outdir: Path) -> Path:
    logger.info("Converting SSE(version3) %s %s", symbol, date)

    df_order = _load_sse_table("SEL2_ORDER", date, symbol)
    df_tx = _load_sse_table("SEL2_TRANSACTION", date, symbol)
    logger.info("Rows after session filter ORDER=%d TX=%d", len(df_order), len(df_tx))

    events = actions_to_events(se_order_actions(df_order), se_transaction_actions(df_tx))
    arr = np.array(events, dtype=HFTBT_DTYPE)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{symbol}_{date}.npz"
    np.savez_compressed(out_path, data=arr)
    logger.info("Saved %d events to %s", len(arr), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SSE tick to hft event stream converter (ORDER/TX only)")
    parser.add_argument("--symbol", required=True, help="SSE symbol, e.g. 600519")
    parser.add_argument("--date", required=True, help="Trading date, e.g. 2026-04-08")
    parser.add_argument(
        "--outdir",
        default="/home/haoranyou/trade_system/output/pipeline_repo/2024-08-30/data_layer/eventstream/2024-08-30/sse",
        help="Output directory for generated npz",
    )
    args = parser.parse_args()
    convert_sse_symbol(args.symbol, args.date, Path(args.outdir))


if __name__ == "__main__":
    main()