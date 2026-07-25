---
title: Picasso 6T8R Front Radar PFR-600 UDP 接口
sources:
  - raw/picasso_pfr600_udp.docx
---

# Picasso 6T8R Front Radar PFR-600 UDP 接口

设备通过 Automotive Ethernet 对外通信，UDP 报文头包含 magicWord 字段。

## 错误码

| 错误码 | 说明 |
|---|---|
| 0xF101 | 波束失配：6T8R 阵列校准参数加载失败。 |
| 0xF104 | 温度越界：与 Columbus 系列一致的板温保护。 |
| 0xF110 | Automotive Ethernet 通信超时：PFR-600 使用车载以太网。 |

## magicWord

magicWord 固定为 `0xACME`，若收到 `0x0102` 表示 magicWord 校验失败，需检查字节序与固件版本一致性。
