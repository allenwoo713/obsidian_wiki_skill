---
title: Acme VisionCam X400 诊断示例
sources:
  - raw/visioncam_x400_diagnosis.docx
---

# Acme VisionCam X400 诊断示例

## 常见故障

| 错误码 | 说明 |
|---|---|
| 0x0102 | magicWord 校验失败：与 X200 同代协议，UDP 报文头字段不匹配。 |
| 0x0201 | 带宽不足：10GigE 链路协商失败，回退至 1GigE 并降分辨率。 |
| E1002 | 温度过高：传感器温度超过 70°C 上限。 |

## 排查步骤

1. 读取故障码确认类别（温度 / 通道 / 通信）。
2. 检查供电与 10GigE 链路。
3. 若为 0xE104 / 0xF104 温度类，确认环境温度在 -30~70°C 内。
