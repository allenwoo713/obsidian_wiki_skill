---
title: Columbus Corner Radar CCR-100 诊断示例
sources:
  - raw/columbus_corner_ccr100_diagnosis.docx
---

# Columbus Corner Radar CCR-100 诊断示例

## 常见故障

| 错误码 | 说明 |
|---|---|
| 0xE101 | 发射通道故障：与 CFR-100 同源，T1 通道无回波。 |
| 0xE201 | 盲区目标：角雷达近场出现静止杂波，已滤除。 |
| 0xE110 | CAN FD 通信超时：与前端雷达一致的总线心跳丢失。 |

## 排查步骤

1. 读取故障码确认类别（温度 / 通道 / 通信）。
2. 检查供电与 CAN FD 链路。
3. 若为 0xE104 / 0xF104 温度类，确认环境温度在 -40~85°C 内。
