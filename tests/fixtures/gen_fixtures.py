"""生成 obsidian_wiki_skill 的可公开评测 fixture 与 queries.jsonl（issue #9）。

全部内容为**虚构/脱敏**的工业产品知识库（Acme VisionCam 工业相机、
Columbus / Picasso 系列毫米波雷达，品牌 Fusionride），不包含任何真实私有资料。

运行：
    python tests/fixtures/gen_fixtures.py
产物：
    tests/fixtures/wiki/        —— 确定性 markdown 评测 wiki
    tests/fixtures/raw/         —— 少量样本源文件（.md / .txt）
    eval/queries.jsonl          —— >=100 条带金标查询
    eval/_gold_index.json       —— 内部：页面->slug 映射（评测用，非必需）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIKI_DIR = HERE / "wiki"
RAW_DIR = HERE / "raw"
EVAL_DIR = HERE.parent.parent / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 虚构产品定义（脱敏）
# ---------------------------------------------------------------------------
PRODUCTS = {
    "visioncam_x200": {
        "name": "Acme VisionCam X200",
        "slug": "visioncam_x200",
        "type": "工业相机",
        "series": "VisionCam",
        "specs": {
            "分辨率": "1920×1080",
            "帧率": "60 fps",
            "视场角": "62°",
            "工作温度": "-20~60°C",
            "功耗": "8 W",
            "接口": "GigE Vision",
            "传感器": "1/2.9\" CMOS",
            "快门": "全局快门",
        },
        "errors": [
            ("0x0102", "magicWord 校验失败：UDP 报文头 magicWord 字段不匹配，常见原因为字节序错误或固件版本不一致。"),
            ("0x0105", "图像超时：连续 3 帧未收到图像数据，检查 GigE Vision 链路与供电。"),
            ("E1001", "温度过高：传感器温度超过 60°C 上限，触发降帧保护。"),
        ],
    },
    "visioncam_x400": {
        "name": "Acme VisionCam X400",
        "slug": "visioncam_x400",
        "type": "工业相机",
        "series": "VisionCam",
        "specs": {
            "分辨率": "2560×1440",
            "帧率": "30 fps",
            "视场角": "45°",
            "工作温度": "-30~70°C",
            "功耗": "12 W",
            "接口": "10GigE",
            "传感器": "1/1.8\" CMOS",
            "快门": "全局快门",
        },
        "errors": [
            ("0x0102", "magicWord 校验失败：与 X200 同代协议，UDP 报文头字段不匹配。"),
            ("0x0201", "带宽不足：10GigE 链路协商失败，回退至 1GigE 并降分辨率。"),
            ("E1002", "温度过高：传感器温度超过 70°C 上限。"),
        ],
    },
    "columbus_front_cfr100": {
        "name": "Columbus Front Radar CFR-100",
        "slug": "columbus_front_cfr100",
        "type": "前向毫米波雷达",
        "series": "Columbus",
        "specs": {
            "频段": "77 GHz",
            "探测距离": "250 m",
            "视场角": "±60°",
            "工作温度": "-40~85°C",
            "功耗": "5 W",
            "接口": "CAN FD",
            "刷新率": "20 Hz",
            "阵列": "3T4R",
        },
        "errors": [
            ("0xE101", "发射通道故障：T1 通道无回波，检查雷达天线与馈线。"),
            ("0xE104", "温度越界：雷达板温超出 -40~85°C 工作范围。"),
            ("0xE110", "CAN FD 通信超时：连续 100ms 未收到总线心跳。"),
        ],
    },
    "columbus_corner_ccr100": {
        "name": "Columbus Corner Radar CCR-100",
        "slug": "columbus_corner_ccr100",
        "type": "角雷达",
        "series": "Columbus",
        "specs": {
            "频段": "77 GHz",
            "探测距离": "150 m",
            "视场角": "±75°",
            "工作温度": "-40~85°C",
            "功耗": "4 W",
            "接口": "CAN FD",
            "刷新率": "20 Hz",
            "阵列": "2T4R",
        },
        "errors": [
            ("0xE101", "发射通道故障：与 CFR-100 同源，T1 通道无回波。"),
            ("0xE201", "盲区目标：角雷达近场出现静止杂波，已滤除。"),
            ("0xE110", "CAN FD 通信超时：与前端雷达一致的总线心跳丢失。"),
        ],
    },
    "columbus_traffic_ctr100": {
        "name": "Columbus Traffic Radar CTR-100",
        "slug": "columbus_traffic_ctr100",
        "type": "交通雷达",
        "series": "Columbus",
        "specs": {
            "频段": "79 GHz",
            "探测距离": "300 m",
            "视场角": "±15°",
            "工作温度": "-40~85°C",
            "功耗": "6 W",
            "接口": "Ethernet",
            "刷新率": "10 Hz",
            "阵列": "4T8R",
        },
        "errors": [
            ("0xE301", "测速异常：雷达与线圈测速偏差超过 5 km/h。"),
            ("0xE104", "温度越界：与 Columbus 系列一致的板温保护。"),
            ("0xE310", "Ethernet 链路断开：交通雷达使用以太网回传。"),
        ],
    },
    "picasso_pfr600": {
        "name": "Picasso 6T8R Front Radar PFR-600",
        "slug": "picasso_pfr600",
        "type": "前向毫米波雷达",
        "series": "Picasso",
        "specs": {
            "频段": "76-79 GHz",
            "探测距离": "280 m",
            "视场角": "±70°",
            "工作温度": "-40~85°C",
            "功耗": "7 W",
            "接口": "Automotive Ethernet",
            "刷新率": "25 Hz",
            "阵列": "6T8R",
        },
        "errors": [
            ("0xF101", "波束失配：6T8R 阵列校准参数加载失败。"),
            ("0xF104", "温度越界：与 Columbus 系列一致的板温保护。"),
            ("0xF110", "Automotive Ethernet 通信超时：PFR-600 使用车载以太网。"),
        ],
    },
}

# 每个产品的通用文档结构（安装 / 校准步骤）
INSTALL_STEPS = [
    "确认供电满足规格书要求，VisionCam 系列使用 PoE 或独立 12V 电源，雷达系列使用整车 12V 电源。",
    "使用屏蔽双绞线连接通信接口：相机为 GigE Vision / 10GigE，Columbus 雷达为 CAN FD，Picasso 为 Automotive Ethernet。",
    "固定安装位置，确保视场角（FOV）覆盖目标区域，避免遮挡与强反射面。",
    "上电后通过配套工具检查链路心跳与温度，确认无 0xE104 / 0xE110 类故障码。",
    "运行自检脚本，验证探测距离与刷新率符合规格书标称值。",
]
CALIB_STEPS = [
    "将设备置于温箱中，在 25°C 标称温度下进行零偏校准。",
    "采集远场点目标回波，拟合距离与角度标定曲线。",
    "对阵列各通道做幅相一致性补偿，雷达系列需加载阵列校准参数。",
    "写入标定参数并断电保存，重启后校验 0xE101 类发射通道故障不再出现。",
    "记录校准报告，包含视场角、探测距离与刷新率实测值。",
]

OVERVIEW = {
    "VisionCam": "VisionCam 系列是 Acme 的工业相机产品线，包含 X200（1920×1080 / 60fps / GigE Vision）与 X400（2560×1440 / 30fps / 10GigE）两款全局快门 CMOS 相机，面向机器视觉与质检场景。",
    "Columbus": "Columbus 系列是 Fusionride 的毫米波雷达产品线，包含前向雷达 CFR-100（77GHz / 250m / CAN FD）、角雷达 CCR-100（77GHz / 150m / CAN FD）与交通雷达 CTR-100（79GHz / 300m / Ethernet），覆盖 L2+ 自动驾驶感知。",
    "Picasso": "Picasso 6T8R 系列是 Fusionride 面向高阶自动驾驶的前向雷达产品线，当前型号 PFR-600（76-79GHz / 280m / 6T8R / Automotive Ethernet），客户包括 Doordash 等干线物流场景。",
}


def slug_of(product_slug: str, doc: str) -> str:
    return f"{product_slug}_{doc}.md"


def write_page(path: Path, title: str, body: str, sources: list[str]):
    front = "---\n"
    front += f"title: {title}\n"
    front += "sources:\n" + "".join(f"  - {s}\n" for s in sources)
    front += "---\n\n"
    path.write_text(front + body, encoding="utf-8")


def build_wiki():
    if WIKI_DIR.exists():
        shutil.rmtree(WIKI_DIR)
    WIKI_DIR.mkdir(parents=True)
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True)

    for prod in PRODUCTS.values():
        s = prod["slug"]
        name = prod["name"]
        # 1) datasheet
        spec_rows = "\n".join(f"| {k} | {v} |" for k, v in prod["specs"].items())
        ds_body = (
            f"# {name} 数据手册\n\n"
            f"{name} 是 {prod['series']} 系列的{prod['type']}。\n\n"
            f"## 关键规格\n\n"
            f"| 参数 | 数值 |\n|---|---|\n{spec_rows}\n\n"
            f"## 概述\n\n{OVERVIEW[prod['series']]}\n\n"
            f"## 接口说明\n\n本设备接口为 **{prod['specs']['接口']}**，"
            f"工作温度范围 {prod['specs']['工作温度']}，功耗 {prod['specs']['功耗']}。\n"
        )
        write_page(WIKI_DIR / slug_of(s, "datasheet"), f"{name} 数据手册",
                   ds_body, [f"raw/{s}_datasheet.docx"])
        # 2) install
        steps = "\n".join(f"{i+1}. {t}" for i, t in enumerate(INSTALL_STEPS))
        inst_body = (
            f"# {name} 安装规范\n\n"
            f"## 安装步骤\n\n{steps}\n\n"
            f"## 供电与接口\n\n设备接口为 {prod['specs']['接口']}，"
            f"安装时确保视场角 {prod['specs']['视场角']} 内无遮挡。\n"
        )
        write_page(WIKI_DIR / slug_of(s, "install"), f"{name} 安装规范",
                   inst_body, [f"raw/{s}_install.docx"])
        # 3) calibration
        csteps = "\n".join(f"{i+1}. {t}" for i, t in enumerate(CALIB_STEPS))
        array_line = ""
        if "阵列" in prod["specs"]:
            array_line = (
                f"## 阵列参数\n\n本设备阵列配置为 {prod['specs']['阵列']}，"
                f"校准需加载对应阵列校准参数，避免 0xE101 / 0xF101 类发射通道故障。\n\n"
            )
        cal_body = (
            f"# {name} 校准规范\n\n"
            f"## 校准流程\n\n{csteps}\n\n"
            f"{array_line}"
        )
        write_page(WIKI_DIR / slug_of(s, "calibration"), f"{name} 校准规范",
                   cal_body, [f"raw/{s}_calibration.docx"])
        # 4) udp interface
        err_rows = "\n".join(f"| {c} | {d} |" for c, d in prod["errors"])
        udp_body = (
            f"# {name} UDP 接口\n\n"
            f"设备通过 {prod['specs']['接口']} 对外通信，UDP 报文头包含 magicWord 字段。\n\n"
            f"## 错误码\n\n| 错误码 | 说明 |\n|---|---|\n{err_rows}\n\n"
            f"## magicWord\n\nmagicWord 固定为 `0xACME`，若收到 `0x0102` 表示 magicWord 校验失败，"
            f"需检查字节序与固件版本一致性。\n"
        )
        write_page(WIKI_DIR / slug_of(s, "udp"), f"{name} UDP 接口",
                   udp_body, [f"raw/{s}_udp.docx"])
        # 5) diagnosis
        diag_body = (
            f"# {name} 诊断示例\n\n"
            f"## 常见故障\n\n| 错误码 | 说明 |\n|---|---|\n{err_rows}\n\n"
            f"## 排查步骤\n\n1. 读取故障码确认类别（温度 / 通道 / 通信）。\n"
            f"2. 检查供电与 {prod['specs']['接口']} 链路。\n"
            f"3. 若为 0xE104 / 0xF104 温度类，确认环境温度在 {prod['specs']['工作温度']} 内。\n"
        )
        write_page(WIKI_DIR / slug_of(s, "diagnosis"), f"{name} 诊断示例",
                   diag_body, [f"raw/{s}_diagnosis.docx"])

    # 系列 overview（global 查询用）
    for series, ov in OVERVIEW.items():
        members = [p["name"] for p in PRODUCTS.values() if p["series"] == series]
        ov_body = (
            f"# {series} 系列概述\n\n{ov}\n\n"
            f"## 成员产品\n\n" + "\n".join(f"- {m}" for m in members) + "\n"
        )
        write_page(WIKI_DIR / f"{series.lower()}_series_overview.md",
                   f"{series} 系列概述", ov_body, ["raw/series_overview.docx"])

    # raw 样本源文件（解析测试用，纯文本脱敏）
    (RAW_DIR / "sample_source.txt").write_text(
        "Acme VisionCam X200 分辨率 1920×1080，接口 GigE Vision，工作温度 -20~60°C。\n"
        "Columbus Front Radar CFR-100 频段 77GHz，探测距离 250m，接口 CAN FD。\n",
        encoding="utf-8")
    (RAW_DIR / "readme.md").write_text(
        "# 样本源文件\n用于解析器测试，内容为虚构脱敏数据。\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# queries.jsonl 生成（金标由结构化数据推导，保证正确）
# ---------------------------------------------------------------------------
def build_queries():
    q = []

    def add(query, intent, pages, sections, facts):
        q.append({
            "query": query,
            "intent": intent,
            "relevant_pages": pages,
            "relevant_sections": sections,
            "required_facts": facts,
        })

    # ---- 25 精确参数 / 型号 / 错误码 ----
    exact_specs = [
        ("visioncam_x200", "分辨率", "1920×1080"),
        ("visioncam_x200", "帧率", "60 fps"),
        ("visioncam_x200", "接口", "GigE Vision"),
        ("visioncam_x400", "分辨率", "2560×1440"),
        ("visioncam_x400", "帧率", "30 fps"),
        ("visioncam_x400", "接口", "10GigE"),
        ("columbus_front_cfr100", "探测距离", "250 m"),
        ("columbus_front_cfr100", "频段", "77 GHz"),
        ("columbus_front_cfr100", "接口", "CAN FD"),
        ("columbus_corner_ccr100", "探测距离", "150 m"),
        ("columbus_corner_ccr100", "视场角", "±75°"),
        ("columbus_traffic_ctr100", "探测距离", "300 m"),
        ("columbus_traffic_ctr100", "频段", "79 GHz"),
        ("picasso_pfr600", "探测距离", "280 m"),
        ("picasso_pfr600", "阵列", "6T8R"),
        ("picasso_pfr600", "接口", "Automotive Ethernet"),
        ("columbus_front_cfr100", "工作温度", "-40~85°C"),
        ("columbus_corner_ccr100", "刷新率", "20 Hz"),
        ("columbus_traffic_ctr100", "刷新率", "10 Hz"),
        ("picasso_pfr600", "刷新率", "25 Hz"),
        ("visioncam_x200", "视场角", "62°"),
        ("visioncam_x400", "视场角", "45°"),
        ("columbus_front_cfr100", "视场角", "±60°"),
        ("picasso_pfr600", "视场角", "±70°"),
        ("columbus_traner_placeholder", "阵列", "4T8R"),  # 替换为真实
    ]
    # 修正最后一条
    exact_specs[-1] = ("columbus_traffic_ctr100", "阵列", "4T8R")

    for prod_slug, param, val in exact_specs:
        prod = PRODUCTS[prod_slug]
        add(f"{prod['name']} 的{param}是多少",
            "lookup",
            [slug_of(prod_slug, "datasheet")],
            [f"关键规格"],
            [val])

    # ---- 错误码（额外补充，替换上面占位之外的精确查询）----
    # 上面已含参数；下面专加错误码类（不计入 25，但丰富 lookup）
    err_queries = [
        ("visioncam_x200", "0x0102", "magicWord"),
        ("columbus_front_cfr100", "0xE101", "发射通道故障"),
        ("columbus_corner_ccr100", "0xE101", "发射通道故障"),
        ("picasso_pfr600", "0xF101", "波束失配"),
        ("columbus_traffic_ctr100", "0xE310", "Ethernet 链路断开"),
    ]
    for prod_slug, code, fact in err_queries:
        prod = PRODUCTS[prod_slug]
        add(f"{prod['name']} 报 {code} 错误是什么意思",
            "lookup",
            [slug_of(prod_slug, "udp"), slug_of(prod_slug, "diagnosis")],
            ["错误码", "常见故障"],
            [fact, code])

    # ---- 20 中文语义 ----
    DS = lambda s: slug_of(s, "datasheet")
    CAL = lambda s: slug_of(s, "calibration")
    INST = lambda s: slug_of(s, "install")
    UDP = lambda s: slug_of(s, "udp")
    DIAG = lambda s: slug_of(s, "diagnosis")
    semantic = [
        ("工业相机选型时为什么要关注视场角", [DS("visioncam_x200"), DS("visioncam_x400")], ["视场角"]),
        ("毫米波雷达在自动驾驶中起什么作用", ["columbus_series_overview.md", DS("columbus_front_cfr100")], ["概述"]),
        ("为什么雷达需要校准", [CAL("columbus_front_cfr100"), CAL("columbus_corner_ccr100")], ["校准流程"]),
        ("全局快门对机器视觉有何意义", [DS("visioncam_x200"), DS("visioncam_x400")], ["全局快门"]),
        ("CAN FD 相比传统 CAN 的优势", [DS("columbus_front_cfr100"), DS("columbus_corner_ccr100")], ["CAN FD"]),
        ("77GHz 与 79GHz 雷达频段差异", [DS("columbus_front_cfr100"), DS("columbus_traffic_ctr100")], ["77 GHz", "79 GHz"]),
        ("为什么交通雷达用以太网回传", [DS("columbus_traffic_ctr100")], ["Ethernet"]),
        ("车载以太网在雷达中的应用", [DS("picasso_pfr600")], ["Automotive Ethernet"]),
        ("探测距离受哪些因素影响", [DS("columbus_front_cfr100"), DS("columbus_traffic_ctr100")], ["探测距离"]),
        ("刷新率对感知系统的重要性", [DS("columbus_front_cfr100"), DS("columbus_traffic_ctr100")], ["刷新率"]),
        ("阵列配置 6T8R 意味着什么", [CAL("picasso_pfr600"), DS("picasso_pfr600")], ["6T8R"]),
        ("温度保护对户外雷达的必要性", [DS("columbus_front_cfr100")], ["工作温度"]),
        ("PoE 供电对工业相机的便利", [DS("visioncam_x200"), INST("visioncam_x200")], ["功耗"]),
        ("角雷达与前向雷达的分工", [DS("columbus_front_cfr100"), DS("columbus_corner_ccr100")], ["视场角"]),
        ("如何判断雷达通道故障", [DIAG("columbus_front_cfr100"), DIAG("columbus_corner_ccr100")], ["发射通道故障"]),
        ("分辨率与帧率的权衡", [DS("visioncam_x200"), DS("visioncam_x400")], ["分辨率", "帧率"]),
        ("盲区目标为何在角雷达出现", [DS("columbus_corner_ccr100"), DIAG("columbus_corner_ccr100")], ["盲区"]),
        ("标定参数丢失会导致什么", [CAL("columbus_front_cfr100"), CAL("columbus_corner_ccr100")], ["校准流程"]),
        ("为什么 UDP 报文需要 magicWord", [UDP("visioncam_x200")], ["magicWord"]),
        ("干线物流对前向雷达的要求", [DS("picasso_pfr600")], ["探测距离"]),
    ]
    for query, pages, facts in semantic:
        add(query, "lookup", pages, ["关键规格"], facts)

    # ---- 15 中英混合 ----
    mixed = [
        ("VisionCam X200 的 GigE Vision 接口如何配置", [DS("visioncam_x200"), UDP("visioncam_x200")], ["GigE Vision"]),
        ("CFR-100 CAN FD 通信协议怎么设置", [DS("columbus_front_cfr100"), UDP("columbus_front_cfr100")], ["CAN FD"]),
        ("Picasso PFR-600 Automotive Ethernet 配置步骤", [DS("picasso_pfr600"), INST("picasso_pfr600")], ["Automotive Ethernet"]),
        ("Columbus Traffic Radar CTR-100 Ethernet 回传延迟", [DS("columbus_traffic_ctr100"), UDP("columbus_traffic_ctr100")], ["Ethernet"]),
        ("VisionCam X400 10GigE 带宽不足如何处理", [DS("visioncam_x400"), UDP("visioncam_x400")], ["10GigE"]),
        ("CFR-100 77GHz 雷达 FOV 是多少", [DS("columbus_front_cfr100")], ["77 GHz", "视场角"]),
        ("CCR-100 corner radar 安装注意事项", [DS("columbus_corner_ccr100"), INST("columbus_corner_ccr100")], ["视场角"]),
        ("PFR-600 6T8R array calibration 流程", [CAL("picasso_pfr600"), DS("picasso_pfr600")], ["6T8R"]),
        ("VisionCam global shutter 优势", [DS("visioncam_x200"), DS("visioncam_x400")], ["全局快门"]),
        ("Radar detection range 250m 对应哪款", [DS("columbus_front_cfr100")], ["250 m"]),
        ("CTR-100 traffic radar 79GHz 频段特点", [DS("columbus_traffic_ctr100")], ["79 GHz"]),
        ("X200 UDP magicWord 0x0102 报错", [UDP("visioncam_x200")], ["0x0102", "magicWord"]),
        ("CFR-100 CAN FD heartbeat timeout 0xE110", [DS("columbus_front_cfr100"), UDP("columbus_front_cfr100")], ["0xE110"]),
        ("Picasso PFR-600 refresh rate 25Hz 含义", [DS("picasso_pfr600")], ["25 Hz"]),
        ("VisionCam X200 frame rate 60fps 应用", [DS("visioncam_x200")], ["60 fps"]),
    ]
    for query, pages, facts in mixed:
        add(query, "lookup", pages, ["关键规格"], facts)

    # ---- 15 流程 / 步骤 ----
    procedure = [
        ("VisionCam X200 安装步骤", [INST("visioncam_x200"), DS("visioncam_x200")], ["安装步骤"], ["安装规范"]),
        ("CFR-100 雷达校准流程", [CAL("columbus_front_cfr100"), DS("columbus_front_cfr100")], ["校准流程"], ["校准规范"]),
        ("Picasso PFR-600 校准步骤", [CAL("picasso_pfr600"), DS("picasso_pfr600")], ["校准流程"], ["校准规范"]),
        ("Columbus Traffic Radar CTR-100 安装规范", [INST("columbus_traffic_ctr100"), DS("columbus_traffic_ctr100")], ["安装步骤"], ["安装规范"]),
        ("VisionCam X400 如何安装", [INST("visioncam_x400"), DS("visioncam_x400")], ["安装步骤"], ["安装规范"]),
        ("CCR-100 角雷达校准方法", [CAL("columbus_corner_ccr100"), DS("columbus_corner_ccr100")], ["校准流程"], ["校准规范"]),
        ("如何对毫米波雷达做零偏校准", [CAL("columbus_front_cfr100"), CAL("columbus_corner_ccr100")], ["校准流程"], ["校准规范"]),
        ("设备上线前自检步骤", [INST("visioncam_x200"), INST("visioncam_x400")], ["安装步骤"], ["安装规范"]),
        ("阵列校准参数加载流程", [CAL("picasso_pfr600"), CAL("columbus_front_cfr100")], ["阵列参数"], ["校准规范"]),
        ("VisionCam X200 上电检查", [INST("visioncam_x200"), DS("visioncam_x200")], ["安装步骤"], ["安装规范"]),
        ("CFR-100 链路心跳检查方法", [INST("columbus_front_cfr100"), DS("columbus_front_cfr100")], ["安装步骤"], ["安装规范"]),
        ("PFR-600 车载以太网连通性验证", [DS("picasso_pfr600"), INST("picasso_pfr600")], ["接口说明"], ["数据手册"]),
        ("雷达温度保护触发后如何处理", [DS("columbus_front_cfr100")], ["工作温度"], ["数据手册"]),
        ("相机 GigE Vision 链路排查", [DS("visioncam_x200"), UDP("visioncam_x200")], ["接口说明"], ["数据手册"]),
        ("校准规范中报告应记录哪些内容", [CAL("columbus_front_cfr100"), CAL("columbus_corner_ccr100")], ["校准流程"], ["校准规范"]),
    ]
    for query, pages, secs, facts in procedure:
        add(query, "procedure", pages, secs, facts)

    # ---- 10 对比 ----
    comparison = [
        ("VisionCam X200 和 X400 分辨率对比",
         ["visioncam_x200", "visioncam_x400"], ["关键规格"], ["1920×1080", "2560×1440"]),
        ("Columbus Front Radar 与 Corner Radar 探测距离差异",
         ["columbus_front_cfr100", "columbus_corner_ccr100"], ["关键规格"], ["250 m", "150 m"]),
        ("CFR-100 与 CTR-100 接口有何不同",
         ["columbus_front_cfr100", "columbus_traffic_ctr100"], ["接口说明"], ["CAN FD", "Ethernet"]),
        ("Picasso PFR-600 相比 Columbus CFR-100 阵列优势",
         ["picasso_pfr600", "columbus_front_cfr100"], ["阵列参数"], ["6T8R", "3T4R"]),
        ("X200 与 X400 帧率谁更高",
         ["visioncam_x200", "visioncam_x400"], ["关键规格"], ["60 fps", "30 fps"]),
        ("77GHz 与 79GHz 雷达适用场景",
         ["columbus_front_cfr100", "columbus_traffic_ctr100"], ["关键规格"], ["77 GHz", "79 GHz"]),
        ("前向雷达与角雷达视场角对比",
         ["columbus_front_cfr100", "columbus_corner_ccr100"], ["关键规格"], ["±60°", "±75°"]),
        ("CAN FD 雷达与 Automotive Ethernet 雷达区别",
         ["columbus_front_cfr100", "picasso_pfr600"], ["接口说明"], ["CAN FD", "Automotive Ethernet"]),
        ("Columbus 与 Picasso 系列工作温度是否一致",
         ["columbus_front_cfr100", "picasso_pfr600"], ["关键规格"], ["-40~85°C"]),
        ("交通雷达与前向雷达刷新率差异",
         ["columbus_traffic_ctr100", "columbus_front_cfr100"], ["关键规格"], ["10 Hz", "20 Hz"]),
    ]
    for query, keys, secs, facts in comparison:
        pages = [slug_of(k, "datasheet") for k in keys]
        add(query, "comparison", pages, secs, facts)

    # ---- 10 图谱关系（多页相关）----
    graph_rel = [
        ("支持 CAN FD 接口的雷达有哪些",
         ["columbus_front_cfr100", "columbus_corner_ccr100"], ["接口说明"]),
        ("哪些产品工作温度达到 -40~85°C",
         ["columbus_front_cfr100", "columbus_corner_ccr100", "columbus_traffic_ctr100", "picasso_pfr600"],
         ["关键规格"]),
        ("使用以太网回传的雷达",
         ["columbus_traffic_ctr100", "picasso_pfr600"], ["接口说明"]),
        ("Columbus 系列包含哪些雷达",
         ["columbus_front_cfr100", "columbus_corner_ccr100", "columbus_traffic_ctr100"], ["系列概述"]),
        ("哪些相机使用 GigE 接口",
         ["visioncam_x200"], ["接口说明"]),
        ("探测距离超过 250 米的雷达",
         ["columbus_traffic_ctr100", "picasso_pfr600"], ["关键规格"]),
        ("视场角大于 ±70° 的设备",
         ["columbus_corner_ccr100", "picasso_pfr600"], ["关键规格"]),
        ("哪些雷达采用 6T8R 或 4T8R 阵列",
         ["picasso_pfr600", "columbus_traffic_ctr100"], ["阵列参数"]),
        ("77GHz 频段的雷达产品",
         ["columbus_front_cfr100", "columbus_corner_ccr100"], ["关键规格"]),
        ("面向干线物流的雷达型号",
         ["picasso_pfr600"], ["概述"]),
    ]
    for query, keys, secs in graph_rel:
        pages = [slug_of(k, "datasheet") for k in keys]
        add(query, "relation", pages, secs, [PRODUCTS[k]["name"] for k in keys])

    # ---- 5 全局主题 ----
    global_q = [
        ("Columbus 系列整体概述",
         ["columbus_series_overview.md", DS("columbus_front_cfr100"), DS("columbus_corner_ccr100"), DS("columbus_traffic_ctr100")],
         ["系列概述"], ["Columbus"]),
        ("整个 Picasso 6T8R 系列有哪些产品",
         ["picasso_series_overview.md", DS("picasso_pfr600")],
         ["系列概述"], ["Picasso"]),
        ("VisionCam 工业相机产品线总览",
         ["visioncam_series_overview.md", DS("visioncam_x200"), DS("visioncam_x400")],
         ["系列概述"], ["VisionCam"]),
        ("Fusionride 所有雷达型号盘点",
         [DS("columbus_front_cfr100"), DS("columbus_corner_ccr100"), DS("columbus_traffic_ctr100"), DS("picasso_pfr600")],
         ["数据手册"], ["Columbus"]),
        ("各系列工作温度范围统一吗",
         [DS("columbus_front_cfr100"), DS("columbus_corner_ccr100"), DS("columbus_traffic_ctr100"), DS("picasso_pfr600"), DS("visioncam_x200")],
         ["关键规格"], ["工作温度"]),
    ]
    for query, pages, secs, facts in global_q:
        add(query, "global", pages, secs, facts)

    # 写出
    out = EVAL_DIR / "queries.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for item in q:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"生成 {len(q)} 条查询 -> {out}")
    # 分布统计
    from collections import Counter
    c = Counter(x["intent"] for x in q)
    print("意图分布:", dict(c))


if __name__ == "__main__":
    build_wiki()
    build_queries()
