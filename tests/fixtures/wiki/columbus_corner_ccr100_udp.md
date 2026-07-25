---
title: Columbus Corner Radar CCR-100 UDP 接口
sources:
  - raw/columbus_corner_ccr100_udp.docx
---

# Columbus Corner Radar CCR-100 UDP 接口

设备通过 CAN FD 对外通信，UDP 报文头包含 magicWord 字段。

## 错误码

| 错误码 | 说明 |
|---|---|
| 0xE101 | 发射通道故障：与 CFR-100 同源，T1 通道无回波。 |
| 0xE201 | 盲区目标：角雷达近场出现静止杂波，已滤除。 |
| 0xE110 | CAN FD 通信超时：与前端雷达一致的总线心跳丢失。 |

## magicWord

magicWord 固定为 `0xACME`，若收到 `0x0102` 表示 magicWord 校验失败，需检查字节序与固件版本一致性。
