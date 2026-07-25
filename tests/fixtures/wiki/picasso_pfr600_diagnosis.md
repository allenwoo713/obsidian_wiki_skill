---
title: Picasso 6T8R Front Radar PFR-600 诊断示例
sources:
  - raw/picasso_pfr600_diagnosis.docx
---

# Picasso 6T8R Front Radar PFR-600 诊断示例

## 常见故障

| 错误码 | 说明 |
|---|---|
| 0xF101 | 波束失配：6T8R 阵列校准参数加载失败。 |
| 0xF104 | 温度越界：与 Columbus 系列一致的板温保护。 |
| 0xF110 | Automotive Ethernet 通信超时：PFR-600 使用车载以太网。 |

## 排查步骤

1. 读取故障码确认类别（温度 / 通道 / 通信）。
2. 检查供电与 Automotive Ethernet 链路。
3. 若为 0xE104 / 0xF104 温度类，确认环境温度在 -40~85°C 内。
