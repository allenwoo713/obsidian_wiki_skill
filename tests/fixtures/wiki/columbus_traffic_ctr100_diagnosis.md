---
title: Columbus Traffic Radar CTR-100 诊断示例
sources:
  - raw/columbus_traffic_ctr100_diagnosis.docx
---

# Columbus Traffic Radar CTR-100 诊断示例

## 常见故障

| 错误码 | 说明 |
|---|---|
| 0xE301 | 测速异常：雷达与线圈测速偏差超过 5 km/h。 |
| 0xE104 | 温度越界：与 Columbus 系列一致的板温保护。 |
| 0xE310 | Ethernet 链路断开：交通雷达使用以太网回传。 |

## 排查步骤

1. 读取故障码确认类别（温度 / 通道 / 通信）。
2. 检查供电与 Ethernet 链路。
3. 若为 0xE104 / 0xF104 温度类，确认环境温度在 -40~85°C 内。
