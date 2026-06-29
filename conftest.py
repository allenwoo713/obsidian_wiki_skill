"""pytest 配置：将临时目录设到项目内，避免沙箱对系统 Temp 的写入限制。"""
from pathlib import Path

def pytest_configure(config):
    basetemp = Path(__file__).parent / ".pytest_tmp"
    basetemp.mkdir(exist_ok=True)
    config.option.basetemp = str(basetemp)
