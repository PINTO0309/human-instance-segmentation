#!/usr/bin/env python3
"""
COCOフォーマットのアノテーションから、横x縦=640x480の画像ファイル名を出力するスクリプト。

デフォルトのアノテーションパス:
  data/annotations/instances_val2017_person_only_no_crowd.json

使い方:
  python scripts/print_coco_640x480_images.py
  python scripts/print_coco_640x480_images.py -a path/to/annotations.json
  python scripts/print_coco_640x480_images.py --resize 320,240 --images-root data/val2017

標準出力に条件を満たす `file_name` を1行ずつ出力します。`--resize` を指定すると
画像をリサイズして `scripts/images`（既定）に保存します。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DEFAULT_ANN_PATH = Path("data/annotations/instances_val2017_person_only_no_crowd.json")
DEFAULT_OUTPUT_DIR = Path("scripts/images")


def load_images(ann_path: Path) -> Iterable[dict]:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValueError("Invalid COCO annotations: 'images' is not a list")
    return images


def filter_640x480(images: Iterable[dict]) -> List[str]:
    result: List[str] = []
    for img in images:
        # COCO images entries typically contain: id, width, height, file_name, ...
        width = img.get("width")
        height = img.get("height")
        fname = img.get("file_name")
        if width == 640 and height == 480 and isinstance(fname, str):
            result.append(fname)
    # 並びを安定化
    result.sort()
    return result


def parse_size(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    raw = text.strip().lower().replace(" ", "")
    sep = "," if "," in raw else ("x" if "x" in raw else None)
    if not sep:
        raise argparse.ArgumentTypeError("サイズは 'W,H' または 'WxH' で指定してください")
    try:
        w_str, h_str = raw.split(sep)
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception as e:
        raise argparse.ArgumentTypeError("サイズは 'W,H' または 'WxH' の正の整数で指定してください") from e


def try_import_pil():
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception:
        return None


def resolve_image_path(images_root: Optional[Path], file_name: str) -> Path:
    # 絶対パスが与えられている場合
    p = Path(file_name)
    if p.is_absolute() and p.exists():
        return p
    candidates: List[Path] = []
    if images_root is not None:
        candidates.append(images_root / file_name)
    # よくある配置の推測: data/val2017, data/images/val2017, data
    candidates.extend([
        Path("data/val2017") / file_name,
        Path("data/images/val2017") / file_name,
        Path("data") / file_name,
    ])
    for c in candidates:
        if c.exists():
            return c
    # 見つからない場合は最後の候補を返す（後で存在チェック）
    return candidates[0] if candidates else p


def resize_and_save(src: Path, dst: Path, size: Tuple[int, int], ImageMod) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with ImageMod.open(src) as im:
        resized = im.resize(size, resample=ImageMod.LANCZOS)
        # 元拡張子を維持（Pillowは拡張子から形式を推定）
        resized.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="COCOアノテーションから横640×縦480の画像file_nameを列挙"
    )
    parser.add_argument(
        "-a",
        "--annotations",
        type=Path,
        default=DEFAULT_ANN_PATH,
        help=f"アノテーションJSONへのパス (デフォルト: {DEFAULT_ANN_PATH})",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="リサイズサイズを 'W,H' または 'WxH' で指定（例: 320,240）",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="元画像のルートディレクトリ（未指定時は代表的な配置を自動推測）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"リサイズ画像の出力先 (デフォルト: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    ann_path: Path = args.annotations
    if not ann_path.is_file():
        raise FileNotFoundError(f"アノテーションファイルが見つかりません: {ann_path}")

    images = load_images(ann_path)
    matches = filter_640x480(images)
    for name in matches:
        print(name)

    # オプションでリサイズ＆保存
    if args.resize:
        size = parse_size(args.resize)
        assert size is not None
        ImageMod = try_import_pil()
        if ImageMod is None:
            print(
                "[WARN] Pillow (PIL) が見つかりません。 --resize を使うには 'pip install pillow' を実行してください。",
                file=sys.stderr,
            )
            return
        out_dir: Path = args.output_dir
        images_root: Optional[Path] = args.images_root
        count_ok, count_missing, count_error = 0, 0, 0
        for fname in matches:
            src = resolve_image_path(images_root, fname)
            if not src.exists():
                print(f"[WARN] 画像ファイルが見つかりません: {src}", file=sys.stderr)
                count_missing += 1
                continue
            dst = out_dir / Path(fname).name
            try:
                resize_and_save(src, dst, size, ImageMod)
                count_ok += 1
            except Exception as e:
                print(f"[ERROR] リサイズ失敗: {src} -> {dst}: {e}", file=sys.stderr)
                count_error += 1
        print(
            f"[INFO] リサイズ完了: 成功 {count_ok}, 不明 {count_missing}, 失敗 {count_error}; 出力先: {out_dir}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
