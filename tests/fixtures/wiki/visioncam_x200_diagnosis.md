---
title: Acme VisionCam X200 诊断示例
sources:
  - raw/visioncam_x200_diagnosis.docx
---

# Acme VisionCam X200 诊断示例

## 常见故障

| 错误码 | 说明 |
|---|---|
| 0x0102 | magicWord 校验失败：UDP 报文头 magicWord 字段不匹配，常见原因为字节序错误或固件版本不一致。 |
| 0x0105 | 图像超时：连续 3 帧未收到图像数据，检查 GigE Vision 链路与供电。 |
| E1001 | 温度过高：传感器温度超过 60°C 上限，触发降帧保护。 |

## 排查步骤

1. 读取故障码确认类别（温度 / 通道 / 通信）。
2. 检查供电与 GigE Vision 链路。
3. 若为 0xE104 / 0xF104 温度类，确认环境温度在 -20~60°C 内。
