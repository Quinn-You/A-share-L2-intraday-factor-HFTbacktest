#!/usr/bin/env python3
"""数据准备层：生成 event stream，并整理当日 alpha 输入。

这一层是完整 pipeline 的数据入口，职责明确限制在两块：
1. 找到并读取某个交易日的 alpha 文件，按股票分组。
2. 调用 SSE / SZSE 转换器，为每只股票生成 event stream。

它不做策略计算，也不做收益汇总，只负责把“可回测输入”准备好。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from SSE_eventstream import convert_sse_symbol
from SZSE_eventstream import convert_szse_symbol
from pipeline_core import load_alpha_records_grouped
from pipeline_core import resolve_alpha_path


@dataclass
class PreparedSymbol:
	"""单只股票的数据准备结果。

	这里记录的是“是否准备成功、输入文件在哪里、alpha 有多少点”等元信息，
	供策略层逐股票消费。
	"""

	symbol: str
	market: str
	eventstream_path: Path
	alpha_points: int
	status: str
	message: str = ""


@dataclass
class PreparedUniverseBundle:
	"""策略层消费的数据包。

	可以把它理解成数据层对外暴露的统一返回对象：
	策略层不需要重新理解 alpha 或 event stream 的生成细节，只要消费这个 bundle 即可。
	"""

	trade_date: str
	alpha_file: Path
	repo_dir: Path
	prepared_symbols: list[PreparedSymbol]
	alpha_records_by_symbol: dict[str, list[tuple[int, float]]]
	skipped_symbols: list[dict[str, str]]


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="数据准备层：生成 event stream 并整理 alpha")
	p.add_argument("--date", type=str, required=True, help="交易日 YYYY-MM-DD")
	p.add_argument(
		"--alpha",
		type=Path,
		default=Path("/data/sihang/AlphaPROBETick/alpha_monthly"),
		help="alpha parquet 路径或 alpha_monthly 根目录",
	)
	p.add_argument(
		"--symbols",
		type=str,
		default=None,
		help="可选，逗号分隔的股票代码；不传则使用当日 alpha 中全部股票",
	)
	p.add_argument(
		"--max-symbols",
		type=int,
		default=None,
		help="可选，仅处理前 N 只股票，便于调试",
	)
	p.add_argument(
		"--repo-dir",
		type=Path,
		default=None,
		help="输出 repo 目录；默认 trade_system/output/pipeline_repo/{date}",
	)
	p.add_argument(
		"--force-regenerate",
		action="store_true",
		help="若 event stream 已存在，是否强制重新生成",
	)
	return p


def _default_repo_dir(trade_date: str) -> Path:
	"""生成默认输出目录。

	不同交易日会落在不同目录下，避免结果互相覆盖。
	"""
	return Path(__file__).resolve().parent / "output" / "pipeline_repo" / trade_date


def _normalize_requested_symbols(symbols: str | None) -> list[str] | None:
	"""把命令行传入的逗号分隔股票代码转成列表。"""
	if not symbols:
		return None
	items = [item.strip() for item in symbols.split(",") if item.strip()]
	return items or None


def _resolve_market(symbol: str) -> str | None:
	"""根据股票代码前缀推断交易所。"""
	if symbol.startswith(("00", "001", "002", "003", "200", "300", "301")):
		return "SZSE"
	if symbol.startswith(("60", "68", "90")):
		return "SSE"
	return None


def _eventstream_out_dir(repo_dir: Path, trade_date: str, market: str) -> Path:
	"""给定交易日和交易所，定位 event stream 的输出目录。"""
	return repo_dir / "data_layer" / "eventstream" / trade_date / market.lower()


def _manifest_dir(repo_dir: Path) -> Path:
	"""保存数据层清单文件的目录。"""
	return repo_dir / "data_layer" / "manifest"


def _write_manifest_files(
	repo_dir: Path,
	prepared_symbols: list[PreparedSymbol],
	skipped_symbols: list[dict[str, str]],
) -> None:
	"""把准备成功/失败的股票清单写到 manifest，便于排查和复盘。"""
	manifest_dir = _manifest_dir(repo_dir)
	manifest_dir.mkdir(parents=True, exist_ok=True)

	prepared_df = pl.DataFrame(
		[
			{
				"symbol": item.symbol,
				"market": item.market,
				"eventstream_path": str(item.eventstream_path),
				"alpha_points": item.alpha_points,
				"status": item.status,
				"message": item.message,
			}
			for item in prepared_symbols
		]
		or {
			"symbol": [],
			"market": [],
			"eventstream_path": [],
			"alpha_points": [],
			"status": [],
			"message": [],
		}
	)
	prepared_df.write_csv(manifest_dir / "prepared_symbols.csv")

	skipped_df = pl.DataFrame(skipped_symbols or {"symbol": [], "reason": []})
	skipped_df.write_csv(manifest_dir / "skipped_symbols.csv")


def prepare_universe_data(
	trade_date: str,
	alpha_path: Path,
	repo_dir: Path | None = None,
	requested_symbols: list[str] | None = None,
	max_symbols: int | None = None,
	force_regenerate: bool = False,
) -> PreparedUniverseBundle:
	"""准备全市场回测输入。

	职责只包含两件事：
	1. 读取当日 alpha，并整理成按股票分组的信号列表。
	2. 按股票代码调用 SSE / SZSE 转换脚本，生成 event stream 文件。

	返回值 PreparedUniverseBundle 是后续策略层唯一需要依赖的数据入口。
	"""

	repo_root = repo_dir or _default_repo_dir(trade_date)
	repo_root.mkdir(parents=True, exist_ok=True)

	alpha_file = resolve_alpha_path(alpha_path, trade_date)
	alpha_records_by_symbol = load_alpha_records_grouped(alpha_file, trade_date, requested_symbols)
	symbols = sorted(alpha_records_by_symbol)
	if max_symbols is not None:
		symbols = symbols[:max_symbols]

	prepared_symbols: list[PreparedSymbol] = []
	skipped_symbols: list[dict[str, str]] = []

	# 按股票逐个准备输入文件；某只股票失败不影响其他股票继续准备。
	for symbol in symbols:
		market = _resolve_market(symbol)
		if market is None:
			skipped_symbols.append({"symbol": symbol, "reason": "无法根据代码前缀识别交易所"})
			continue

		out_dir = _eventstream_out_dir(repo_root, trade_date, market)
		out_dir.mkdir(parents=True, exist_ok=True)
		eventstream_path = out_dir / f"{symbol}_{trade_date}.npz"

		try:
			if force_regenerate or (not eventstream_path.exists()):
				if market == "SSE":
					convert_sse_symbol(symbol, trade_date, out_dir)
				else:
					convert_szse_symbol(symbol, trade_date, out_dir)
				status = "generated"
				message = ""
			else:
				status = "existing"
				message = "复用已有 event stream"
		except Exception as exc:
			skipped_symbols.append({"symbol": symbol, "reason": f"eventstream 生成失败: {exc}"})
			continue

		prepared_symbols.append(
			PreparedSymbol(
				symbol=symbol,
				market=market,
				eventstream_path=eventstream_path,
				alpha_points=len(alpha_records_by_symbol[symbol]),
				status=status,
				message=message,
			)
		)

	_write_manifest_files(repo_root, prepared_symbols, skipped_symbols)

	return PreparedUniverseBundle(
		trade_date=trade_date,
		alpha_file=alpha_file,
		repo_dir=repo_root,
		prepared_symbols=prepared_symbols,
		alpha_records_by_symbol={item.symbol: alpha_records_by_symbol[item.symbol] for item in prepared_symbols},
		skipped_symbols=skipped_symbols,
	)


def main() -> None:
	"""支持把数据层单独当成命令行工具执行，便于分层调试。"""
	args = _build_parser().parse_args()
	bundle = prepare_universe_data(
		trade_date=args.date,
		alpha_path=args.alpha,
		repo_dir=args.repo_dir,
		requested_symbols=_normalize_requested_symbols(args.symbols),
		max_symbols=args.max_symbols,
		force_regenerate=args.force_regenerate,
	)

	print("=" * 72)
	print("数据准备完成")
	print(f"date              : {bundle.trade_date}")
	print(f"alpha_file        : {bundle.alpha_file}")
	print(f"repo_dir          : {bundle.repo_dir}")
	print(f"prepared_symbols  : {len(bundle.prepared_symbols)}")
	print(f"skipped_symbols   : {len(bundle.skipped_symbols)}")
	print("=" * 72)


if __name__ == "__main__":
	main()