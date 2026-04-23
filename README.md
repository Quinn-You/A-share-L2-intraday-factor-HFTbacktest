# Trade System Pipeline README

## 1. 项目概述

这套 pipeline 把单股票回测扩展成按交易日批量运行的三层流程：

1. 数据层：准备 alpha 和 event stream。
2. 策略层：逐股票运行回测。
3. 报表层：汇总指标并生成图表与 HTML。

当前实现已经从架构上独立于 alpha_bt.py。即使删除 alpha_bt.py，以下核心文件仍可单独构成完整流程：

- main.py
- pipeline_core.py
- pipeline_data_layer.py
- pipeline_strategy_layer.py
- pipeline_report_layer.py
- SSE_eventstream.py
- SZSE_eventstream.py

## 2. 流程入口与执行顺序

统一入口是 main.py，执行顺序固定：

1. 调用数据层 prepare_universe_data(...)。
2. 调用策略层 run_universe_strategy(...)。
3. 调用报表层 generate_report(...)。
4. 在终端打印最终摘要。

## 3. 各层职责

### 3.1 数据层（pipeline_data_layer.py）

职责：

1. 解析交易日 alpha 文件。
2. 读取 alpha 并按 symbol 分组。
3. 根据代码前缀识别市场（SSE/SZSE）。
4. 生成或复用 event stream。
5. 写出 manifest（prepared/skipped）。

关键点：

- 默认复用已有 npz。
- 仅在传入 --force-regenerate 或文件不存在时重建。

### 3.2 策略层（pipeline_strategy_layer.py）

职责：

1. 消费数据层 bundle。
2. 逐股票调用 pipeline_core.py 的单票回测逻辑。
3. 产出逐票交易明细、全量交易明细、股票级汇总、失败列表。
4. 单股票失败不阻塞全批次。

### 3.3 报表层（pipeline_report_layer.py）

职责：

1. 计算全市场汇总指标。
2. 生成横截面图和分布图。
3. 生成每只股票当日净仓位图。
4. 输出 overall_summary.json、symbol_chart_manifest.csv 和 report.html。

说明：

- 为避免跨层重复，报表层不再重复落盘策略层已有的 all_trades.csv、symbol_summary.csv、strategy_failures.csv。

## 4. 命令行运行方式

### 4.1 跑完整流程

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/main.py --date 2024-08-30
```

### 4.2 指定 alpha 路径

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/main.py \
  --date 2024-08-30 \
  --alpha /home/haoranyou/trade_system/output/test_alpha_input
```

### 4.3 指定股票子集

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/main.py \
  --date 2024-08-30 \
  --alpha /home/haoranyou/trade_system/output/test_alpha_input \
  --symbols 000158,000333,600519
```

### 4.4 限制处理前 N 只

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/main.py \
  --date 2024-08-30 \
  --max-symbols 10
```

### 4.5 强制重建 event stream

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/main.py \
  --date 2024-08-30 \
  --force-regenerate
```

## 5. 输出目录（当前版本）

默认输出根目录：

```text
trade_system/output/pipeline_repo/{date}/
```

目录结构：

```text
output/pipeline_repo/2024-08-30/
├── data_layer/
│   ├── eventstream/
│   │   └── 2024-08-30/
│   │       ├── sse/
│   │       └── szse/
│   └── manifest/
│       ├── prepared_symbols.csv
│       └── skipped_symbols.csv
├── strategy_layer/
│   ├── trades/
│   │   ├── 000158_2024-08-30.csv
│   │   ├── 000333_2024-08-30.csv
│   │   └── ...
│   ├── all_trades.csv
│   ├── symbol_summary.csv
│   └── strategy_failures.csv
└── report_layer/
    ├── overall_summary.json
    ├── symbol_chart_manifest.csv
    ├── cumulative_pnl.png
    ├── rank_total_pnl.png
    ├── rank_sharpe_ratio.png
    ├── pnl_distribution.png
    ├── report.html
    └── symbol_charts/
        ├── 000158/
        │   └── net_position.png
        └── ...
```

## 6. 指标与图表（当前版本）

### 6.1 overall_summary.json 主要字段

- n_symbols_completed
- n_symbols_failed
- n_trades
- total_pnl
- avg_trade_pnl
- sharpe_ratio
- best_symbols_by_pnl（Top 10）
- worst_symbols_by_pnl（Bottom 10）
- best_symbols_by_sharpe（Top 10）
- worst_symbols_by_sharpe（Bottom 10）
- n_symbol_net_position_charts

字段口径：

- avg_trade_pnl：单笔交易 pnl 的均值。
- sharpe_ratio：基于单笔真实收益率序列计算，真实收益率定义为 pnl / entry_notional。

### 6.2 报表层图表

- cumulative_pnl.png
- rank_total_pnl.png（Top 10 / Bottom 10）
- rank_sharpe_ratio.png（Top 10 / Bottom 10）
- pnl_distribution.png
- symbol_charts/<symbol>/net_position.png

## 7. 分层调试

### 7.1 只跑数据层

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/pipeline_data_layer.py --date 2024-08-30
```

### 7.2 跑到策略层（不生成报表）

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/pipeline_strategy_layer.py --date 2024-08-30
```

### 7.3 走完整链路

```bash
/home/haoranyou/miniconda3/bin/python /home/haoranyou/trade_system/main.py --date 2024-08-30
```

## 8. 命令行并行提速（不改代码）

这一节用于先提速，再决定要不要改代码。

### 8.1 思路

1. event stream 先并行预生成。
2. 把股票列表切分成多个 shard。
3. 多进程并行启动 main.py，每个 shard 用不同 --repo-dir。
4. 所有 shard 跑完后再查看各 shard 的 report_layer/overall_summary.json。

### 8.2 为什么每个 shard 要单独 repo-dir

同一 repo-dir 下存在固定文件名（如 strategy_layer/all_trades.csv）。
多个进程写同一路径会互相覆盖或打架。

### 8.3 并行运行模板

```bash
PY="/home/haoranyou/miniconda3/bin/python"
ROOT="/home/haoranyou/trade_system"
DATE="2024-08-30"
ALPHA="/home/haoranyou/trade_system/output/test_alpha_input"
CODES="/home/haoranyou/trade_system/check/unique_codes.csv"
P=8

mapfile -t SYMS < <(tail -n +2 "$CODES" | tr -d '\r' | awk 'NF')
N=${#SYMS[@]}
CHUNK=$(( (N + P - 1) / P ))

mkdir -p "$ROOT/output/pipeline_repo/$DATE/shards"

for i in $(seq 0 $((P-1))); do
  start=$(( i * CHUNK ))
  if [[ $start -ge $N ]]; then
    continue
  fi
  part=( "${SYMS[@]:$start:$CHUNK}" )
  symbols_csv=$(IFS=,; echo "${part[*]}")
  repo="$ROOT/output/pipeline_repo/$DATE/shards/part_$i"

  (
    cd "$ROOT" && \
    "$PY" main.py \
      --date "$DATE" \
      --alpha "$ALPHA" \
      --symbols "$symbols_csv" \
      --repo-dir "$repo"
  ) > "$repo.run.log" 2>&1 &
done

wait
echo "All shards done"
```

### 8.4 并行度建议

1. 先从 P=4 开始。
2. 再试 P=8。
3. 若磁盘 I/O 抖动明显，回退并行度。
4. 不要加 --force-regenerate，优先复用已有 eventstream。

## 9. 常见问题

### 9.1 event stream 会不会重复生成

默认不会。已有 npz 会复用。

仅在以下情况重建：

1. 目标 npz 不存在。
2. 显式传入 --force-regenerate。

### 9.2 中文图表字体警告

若环境缺少中文字体，matplotlib 可能打印 glyph 警告。
这不影响回测结果和报表落盘，只影响图中文字体显示。

## 10. 设计原则

1. 总入口唯一：main.py。
2. 分层清晰：数据层、策略层、报表层。
3. 内核独立：核心回测能力在 pipeline_core.py。
4. 单层可调试：每层都可独立执行。
5. 批次鲁棒：单股票失败不阻断整批。
