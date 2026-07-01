"""pytest 配置：将临时目录设到项目内，避免沙箱对系统 Temp 的写入限制。"""
import os
from pathlib import Path

def pytest_configure(config):
    # 允许通过 --basetemp 覆盖；若未指定则用项目内 .pytest_tmp
    if not config.option.basetemp:
        basetemp = Path(__file__).parent / ".pytest_tmp"
        basetemp.mkdir(exist_ok=True)
        config.option.basetemp = str(basetemp)
