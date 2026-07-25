---
title: Columbus Front Radar CFR-100 诊断示例
sources:
  - raw/columbus_front_cfr100_diagnosis.docx
---

# Columbus Front Radar CFR-100 诊断示例

## 常见故障

| 错误码 | 说明 |
|---|---|
| 0xE101 | 发射通道故障：T1 通道无回波，检查雷达天线与馈线。 |
| 0xE104 | 温度越界：雷达板温超出 -40~85°C 工作范围。 |
| 0xE110 | CAN FD 通信超时：连续 100ms 未收到总线心跳。 |

## 排查步骤

1. 读取故障码确认类别（温度 / 通道 / 通信）。
2. 检查供电与 CAN FD 链路。
3. 若为 0xE104 / 0xF104 温度类，确认环境温度在 -40~85°C 内。
