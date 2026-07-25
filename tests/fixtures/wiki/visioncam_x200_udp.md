---
title: Acme VisionCam X200 UDP 接口
sources:
  - raw/visioncam_x200_udp.docx
---

# Acme VisionCam X200 UDP 接口

设备通过 GigE Vision 对外通信，UDP 报文头包含 magicWord 字段。

## 错误码

| 错误码 | 说明 |
|---|---|
| 0x0102 | magicWord 校验失败：UDP 报文头 magicWord 字段不匹配，常见原因为字节序错误或固件版本不一致。 |
| 0x0105 | 图像超时：连续 3 帧未收到图像数据，检查 GigE Vision 链路与供电。 |
| E1001 | 温度过高：传感器温度超过 60°C 上限，触发降帧保护。 |

## magicWord

magicWord 固定为 `0xACME`，若收到 `0x0102` 表示 magicWord 校验失败，需检查字节序与固件版本一致性。
