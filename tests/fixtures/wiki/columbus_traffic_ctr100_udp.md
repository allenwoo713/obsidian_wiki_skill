---
title: Columbus Traffic Radar CTR-100 UDP 接口
sources:
  - raw/columbus_traffic_ctr100_udp.docx
---

# Columbus Traffic Radar CTR-100 UDP 接口

设备通过 Ethernet 对外通信，UDP 报文头包含 magicWord 字段。

## 错误码

| 错误码 | 说明 |
|---|---|
| 0xE301 | 测速异常：雷达与线圈测速偏差超过 5 km/h。 |
| 0xE104 | 温度越界：与 Columbus 系列一致的板温保护。 |
| 0xE310 | Ethernet 链路断开：交通雷达使用以太网回传。 |

## magicWord

magicWord 固定为 `0xACME`，若收到 `0x0102` 表示 magicWord 校验失败，需检查字节序与固件版本一致性。
