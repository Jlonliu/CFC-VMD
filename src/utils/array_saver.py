# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)

"""
array_saver.py - 二维列表/数组保存为多种格式文件

支持格式: csv, xls, xlsx, txt

用法示例:
    from array_saver import save_array

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    save_array(data, "output.csv")                    # 保存为 CSV（按列存储，每个子列表为一列）
    save_array(data, "output.csv", by="row")         # 按行存储（默认按列）
    save_array(data, "output.xlsx")                  # 保存为 XLSX
    save_array(data, "output.txt", delimiter="\t")   # 保存为 TSV
    save_array(data, "output.xls")                   # 保存为旧版 XLS

    # 支持二维 numpy 数组
    import numpy as np
    arr = np.random.rand(5, 5)
    save_array(arr, "random.csv")

    # 支持自定义表头（by="col" 时表头为列名，by="row" 时表头为行名）
    save_array(data, "output.csv", by="col", header=["ch1", "ch2", "ch3"])

    # 支持自定义精度
    save_array(arr, "output.csv", fmt_float="%.4f")
"""

import csv
import os
import numpy as np


def save_array(
    data,
    filepath,
    *,
    by="col",
    delimiter=",",
    header=None,
    encoding="utf-8",
    fmt_float="%.6f",
    fmt="auto",
):
    """
    将二维列表或 numpy 数组保存为 CSV/XLS/XLSX/TXT 文件。

    Parameters
    ----------
    data : list[list] | np.ndarray
        二维数据，形状为 (M, N)。
    filepath : str
        输出文件路径，扩展名决定格式。
    by : str
        存储方向:
        - "col"（默认）: data 的每个子列表（每行）存为文件的一列。
          即 data[row][col] → 文件第 col 列、第 row 行。
          适合信号处理：每个子列表是一条信号，CSV 每列一条信号。
        - "row": data 的每个子列表存为文件的一行（常规矩阵存储）。
    delimiter : str
        CSV/TXT 分隔符（默认 ","）。xls/xlsx 忽略此参数。
    header : list[str] | None
        表头（可选）。
        - by="col" 时：每列的列名，长度须与 data 行数一致。
        - by="row" 时：每列的列名，长度须与 data 列数一致。
    encoding : str
        文件编码，默认 utf-8。
    fmt_float : str
        浮点数格式字符串，默认 "%.6f"。
    fmt : str
        显式指定格式 ("csv"/"xls"/"xlsx"/"txt")，默认 "auto"（由扩展名推断）。
    """
    # ---------- 输入预处理 ----------
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(f"不支持 {data.ndim} 维数组，仅支持二维")

    # 格式化单元格
    raw = [[_fmt_cell(data[i][j], fmt_float) for j in range(len(data[i]))]
           for i in range(len(data))]

    # by="col" 时转置：data 的行 → 文件的列
    if by == "col":
        ncols = len(raw)          # 原始行数 → 文件列数
        # 取所有子列表的最大长度作为行数，不足的位置填空字符串
        nrows = max(len(row) for row in raw) if raw else 0
        out = [[raw[c][r] if r < len(raw[c]) else ""
                for c in range(ncols)]
               for r in range(nrows)]
    else:
        out = raw

    # ---------- 格式推断 ----------
    if fmt == "auto":
        ext = os.path.splitext(filepath)[1].lower()
        fmt_map = {".csv": "csv", ".txt": "txt", ".xls": "xls", ".xlsx": "xlsx"}
        fmt = fmt_map.get(ext)
        if fmt is None:
            raise ValueError(
                f"无法从扩展名 '{ext}' 推断格式，"
                "请用 fmt 参数显式指定 (csv/xls/xlsx/txt)"
            )

    # ---------- 写入 ----------
    if fmt in ("csv", "txt"):
        _save_csv(filepath, out, delimiter, header, encoding)
    elif fmt == "xls":
        _save_xls(filepath, out, header)
    elif fmt == "xlsx":
        _save_xlsx(filepath, out, header)
    else:
        raise ValueError(f"不支持的格式: {fmt}")

    print(f"[array_saver] 已保存 → {filepath}  ({len(out)} 行 x {len(out[0])} 列, by={by}, 格式: {fmt})")


# ---------- 内部辅助 ----------

def _fmt_cell(val, fmt_float):
    """格式化单个单元格值。"""
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(fmt_float % val)
    return val


def _save_csv(filepath, data, delimiter, header, encoding):
    """保存为 CSV / TXT。"""
    with open(filepath, "w", newline="", encoding=encoding) as f:
        writer = csv.writer(f, delimiter=delimiter)
        if header:
            writer.writerow(header)
        writer.writerows(data)


def _save_xls(filepath, data, header):
    """保存为 XLS（旧版 Excel 格式，依赖 xlwt）。"""
    try:
        import xlwt
    except ImportError:
        raise ImportError("保存 .xls 需要 xlwt，请运行: pip install xlwt")

    wb = xlwt.Workbook(encoding="utf-8")
    ws = wb.add_sheet("Sheet1")
    row_offset = 0
    if header:
        for j, h in enumerate(header):
            ws.write(0, j, h)
        row_offset = 1
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            ws.write(i + row_offset, j, val)
    wb.save(filepath)


def _save_xlsx(filepath, data, header):
    """保存为 XLSX（依赖 openpyxl）。"""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("保存 .xlsx 需要 openpyxl，请运行: pip install openpyxl")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    if header:
        ws.append(header)
    for row in data:
        ws.append(row)
    wb.save(filepath)


def read_csv_columns(file_path, encoding='utf-8', has_header=False):
    """
    读取 CSV 文件，返回一个列表，其中每个元素是文件一列的数据列表。

    参数：
        file_path (str): CSV 文件路径
        encoding (str): 文件编码，默认 'utf-8'
        has_header (bool): 是否包含表头行。若为 True，第一行将作为表头被包含在每列列表中

    返回：
        list: 列列表，例如 [[col1_row1, col1_row2, ...], [col2_row1, ...], ...]
    """
    with open(file_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        # 读取所有行（每行是一个列表）
        rows = list(reader)

    if not rows:
        return []  # 空文件

    # 转置：将行列表转换为列列表
    # zip(*rows) 将每行的对应元素组合成元组，再用 list() 转换为列表
    columns = [list(col) for col in zip(*rows)]

    # 如果不需要表头，可以移除每列的第一个元素
    if not has_header and len(columns) > 0:
        columns = [col[1:] for col in columns]

    return columns


# ---------- 命令行入口 ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="将二维列表/数组保存为 csv, xls, xlsx, txt"
    )
    parser.add_argument("data_expr", help="数据表达式，如 [[1,2],[3,4]]")
    parser.add_argument("filepath", help="输出文件路径")
    parser.add_argument("--by", default="col", choices=["col", "row"],
                        help="存储方向: col=每行数据为一列 (默认), row=常规行存储")
    parser.add_argument("--delimiter", default=",", help="CSV/TXT 分隔符 (默认 ',')")
    parser.add_argument("--header", nargs="*", default=None, help="表头列")
    parser.add_argument("--fmt", default="auto", help="格式: csv/xls/xlsx/txt/auto")
    parser.add_argument("--encoding", default="utf-8", help="文件编码 (默认 utf-8)")
    args = parser.parse_args()

    import ast
    try:
        parsed = ast.literal_eval(args.data_expr)
    except (SyntaxError, ValueError):
        print(f"[错误] 无法解析数据表达式: {args.data_expr}")
        print("请使用合法 Python 字面量，例如: [[1,2,3],[4,5,6]]")
        exit(1)

    if isinstance(parsed, (int, float)):
        parsed = [[parsed]]
    elif isinstance(parsed, (list, tuple)) and len(parsed) > 0 and not isinstance(parsed[0], (list, tuple)):
        parsed = [[v for v in parsed]]

    save_array(
        parsed,
        args.filepath,
        by=args.by,
        delimiter=args.delimiter,
        header=args.header,
        encoding=args.encoding,
        fmt=args.fmt,
    )
