---
title: Columbus Front Radar CFR-100 安装规范
sources:
  - raw/columbus_front_cfr100_install.docx
---

# Columbus Front Radar CFR-100 安装规范

## 安装步骤

1. 确认供电满足规格书要求，VisionCam 系列使用 PoE 或独立 12V 电源，雷达系列使用整车 12V 电源。
2. 使用屏蔽双绞线连接通信接口：相机为 GigE Vision / 10GigE，Columbus 雷达为 CAN FD，Picasso 为 Automotive Ethernet。
3. 固定安装位置，确保视场角（FOV）覆盖目标区域，避免遮挡与强反射面。
4. 上电后通过配套工具检查链路心跳与温度，确认无 0xE104 / 0xE110 类故障码。
5. 运行自检脚本，验证探测距离与刷新率符合规格书标称值。

## 供电与接口

设备接口为 CAN FD，安装时确保视场角 ±60° 内无遮挡。
