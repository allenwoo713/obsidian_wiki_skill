---
title: Acme VisionCam X400 UDP 接口
sources:
  - raw/visioncam_x400_udp.docx
---

# Acme VisionCam X400 UDP 接口

设备通过 10GigE 对外通信，UDP 报文头包含 magicWord 字段。

## 错误码

| 错误码 | 说明 |
|---|---|
| 0x0102 | magicWord 校验失败：与 X200 同代协议，UDP 报文头字段不匹配。 |
| 0x0201 | 带宽不足：10GigE 链路协商失败，回退至 1GigE 并降分辨率。 |
| E1002 | 温度过高：传感器温度超过 70°C 上限。 |

## magicWord

magicWord 固定为 `0xACME`，若收到 `0x0102` 表示 magicWord 校验失败，需检查字节序与固件版本一致性。
