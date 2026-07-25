---
title: Columbus Front Radar CFR-100 UDP 接口
sources:
  - raw/columbus_front_cfr100_udp.docx
---

# Columbus Front Radar CFR-100 UDP 接口

设备通过 CAN FD 对外通信，UDP 报文头包含 magicWord 字段。

## 错误码

| 错误码 | 说明 |
|---|---|
| 0xE101 | 发射通道故障：T1 通道无回波，检查雷达天线与馈线。 |
| 0xE104 | 温度越界：雷达板温超出 -40~85°C 工作范围。 |
| 0xE110 | CAN FD 通信超时：连续 100ms 未收到总线心跳。 |

## magicWord

magicWord 固定为 `0xACME`，若收到 `0x0102` 表示 magicWord 校验失败，需检查字节序与固件版本一致性。
