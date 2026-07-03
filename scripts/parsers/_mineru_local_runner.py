"""Standalone runner invoked by the MinerU venv Python to call do_parse."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 5:
        print(
            "Usage: python _mineru_local_runner.py <input_pdf_path> <output_dir> <backend> <language>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    backend = sys.argv[3]
    language = sys.argv[4]

    os.environ["MINERU_MODEL_SOURCE"] = "local"
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pdf_bytes = Path(input_pdf_path).read_bytes()
    stem = Path(input_pdf_path).stem

    from mineru.cli.common import do_parse

    try:
        do_parse(
            output_dir=output_dir,
            pdf_file_names=[stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[language],
            backend=backend,
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            start_page_id=0,
            end_page_id=None,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
