import argparse
import json
from pathlib import Path

CLASSES = ["sphere", "cube", "cylinder"]


VOXEL_SIZE = 0.01,
RADIUS_NORMAL = 2 * VOXEL_SIZE
MAX_NN_NORMAL = 30
RADIUS_FEATURE = 5 * VOXEL_SIZE
MAX_NN_FEATURE = 100


def main():
    ap = argparse.ArgumentParser(description="Создать структуру датасета для 3D классификации.")
    ap.add_argument("--root", type=Path, default=Path("dataset_root"),
                    help="Корневая директория датасета")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Список сплитов (по умолчанию: train val test)")
    args = ap.parse_args()

    root = args.root
    meta = root / "meta"

    for split in args.splits:
        for cls in CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)

    # meta/
    meta.mkdir(parents=True, exist_ok=True)
    with open(meta / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({c: i for i, c in enumerate(CLASSES)}, f, ensure_ascii=False, indent=2)

    with open(meta / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump({s: {c: [] for c in CLASSES} for s in args.splits}, f, ensure_ascii=False, indent=2)

    with open(meta / "preprocessing.yaml", "w", encoding="utf-8") as f:
        f.write(
            f"voxel_size: {VOXEL_SIZE}\n"
            f"radius_normal: {RADIUS_NORMAL}\n"
            f"max_nn_normal: {MAX_NN_NORMAL}\n"
            f"radius_feature: {RADIUS_FEATURE}\n"
            f"max_nn_feature: {MAX_NN_FEATURE}\n"
        )

    print(f"[OK] Структура создана в: {root.resolve()}")

if __name__ == "__main__":
    main()
