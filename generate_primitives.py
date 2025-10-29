import argparse
import json
import random
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

CLASSES = ["sphere", "cube", "cylinder", "cone", "torus"]

def next_index_for_class(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p.stem for p in out_dir.glob(f"{prefix}_*.ply")])
    if not existing:
        return 1
    last = max(int(name.split("_")[-1]) for name in existing)
    return last + 1

def mesh_sphere(radius: float, resolution: int = 40):
    return o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)

def mesh_cube(edge: float):
    return o3d.geometry.TriangleMesh.create_box(width=edge, height=edge, depth=edge)

def mesh_cylinder(radius: float, height: float, resolution: int = 64, split: int = 4):
    return o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height,
                                                     resolution=resolution, split=split)

def mesh_cone(radius: float, height: float, resolution: int = 64, split: int = 4):
    return o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height,
                                                 resolution=resolution, split=split)

def mesh_torus(radius: float, tube_radius: float, radial_resolution: int = 48, tubular_resolution: int = 32):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=radius,
        tube_radius=tube_radius,
        radial_resolution=radial_resolution,
        tubular_resolution=tubular_resolution
    )
    return mesh

def sample_dense_point_cloud(mesh: o3d.geometry.TriangleMesh,
                             points_per_unit_area: float,
                             method: str = "poisson",
                             init_factor: float = 5.0):
    mesh.compute_vertex_normals()
    area = mesh.get_surface_area()
    n_points = max(100, int(round(area * points_per_unit_area)))

    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points,
                                              init_factor=init_factor)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return pcd

def update_manifest(manifest_path: Path, split: str, cls_name: str, rel_path: str):
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in manifest:
        manifest[split] = {}
    if cls_name not in manifest[split]:
        manifest[split][cls_name] = []
    if rel_path not in manifest[split][cls_name]:
        manifest[split][cls_name].append(rel_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def generate_split(args, split_name, per_class_count):
    """Генерирует данные для конкретного сплита"""
    root = args.root
    split_dir = root / split_name
    manifest_path = root / "meta" / "split_manifest.json"

    for cls_name in CLASSES:
        (split_dir / cls_name).mkdir(parents=True, exist_ok=True)

    prefix_map = {
        "sphere": "s", 
        "cube": "c", 
        "cylinder": "y",
        "cone": "o",
        "torus": "t"
    }

    for cls_name in CLASSES:
        out_dir = split_dir / cls_name
        prefix = prefix_map[cls_name]
        idx = next_index_for_class(out_dir, prefix)

        print(f"Генерация {per_class_count} объектов класса {cls_name} в сплит {split_name}")
        for _ in tqdm(range(per_class_count), desc=f"{cls_name:>8}", unit="obj"):
            if cls_name == "sphere":
                r = random.uniform(args.sphere_min_radius, args.sphere_max_radius)
                mesh = mesh_sphere(radius=r)
            elif cls_name == "cube":
                edge = random.uniform(args.cube_min_edge, args.cube_max_edge)
                mesh = mesh_cube(edge=edge)
            elif cls_name == "cylinder":
                r = random.uniform(args.cyl_min_radius, args.cyl_max_radius)
                h = random.uniform(args.cyl_min_height, args.cyl_max_height)
                mesh = mesh_cylinder(radius=r, height=h)
            elif cls_name == "cone":
                r = random.uniform(args.cone_min_radius, args.cone_max_radius)
                h = random.uniform(args.cone_min_height, args.cone_max_height)
                mesh = mesh_cone(radius=r, height=h)
            elif cls_name == "torus":
                radius = random.uniform(args.torus_min_radius, args.torus_max_radius)
                tube_radius = random.uniform(args.torus_min_tube_radius, args.torus_max_tube_radius)
                mesh = mesh_torus(radius=radius, tube_radius=tube_radius)

            pcd = sample_dense_point_cloud(
                mesh,
                points_per_unit_area=args.points_per_unit_area,
                method=args.sampling_method,
                init_factor=args.poisson_init_factor,
            )

            out_path = out_dir / f"{prefix}_{idx:04d}.ply"
            o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=True)

            rel_path = str(out_path.relative_to(root)).replace("\\", "/")
            update_manifest(manifest_path, split_name, cls_name, rel_path)
            idx += 1

    print(f"Сгенерировано по {per_class_count} объектов для каждого класса в {split_dir}")

def main():
    ap = argparse.ArgumentParser(description="Генерация СЫРЫХ облаков точек примитивов (сфера/куб/цилиндр/конус/тор).")
    ap.add_argument("--root", type=Path, default=Path("dataset"),
                    help="Корень датасета (должен содержать meta/, train/val/test/)")
    
    # Изменяем аргумент split на multiple choice
    ap.add_argument("--split", type=str, nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"],
                    help="Для каких сплитов генерировать данные")
    
    # Добавляем отдельные параметры для каждого сплита
    ap.add_argument("--train-per-class", type=int, default=50, 
                    help="Сколько объектов генерировать для каждого класса в train")
    ap.add_argument("--val-per-class", type=int, default=15, 
                    help="Сколько объектов генерировать для каждого класса в val")
    ap.add_argument("--test-per-class", type=int, default=15, 
                    help="Сколько объектов генерировать для каждого класса в test")
    
    ap.add_argument("--seed", type=int, default=42, help="Фиксировать seed")

    # Диапазоны для сферы
    ap.add_argument("--sphere-min-radius", type=float, default=0.3)
    ap.add_argument("--sphere-max-radius", type=float, default=0.9)

    # Диапазоны для куба
    ap.add_argument("--cube-min-edge", type=float, default=0.4)
    ap.add_argument("--cube-max-edge", type=float, default=1.2)

    # Диапазоны для цилиндра
    ap.add_argument("--cyl-min-radius", type=float, default=0.25)
    ap.add_argument("--cyl-max-radius", type=float, default=0.7)
    ap.add_argument("--cyl-min-height", type=float, default=0.5)
    ap.add_argument("--cyl-max-height", type=float, default=1.5)

    # Диапазоны для конуса
    ap.add_argument("--cone-min-radius", type=float, default=0.2)
    ap.add_argument("--cone-max-radius", type=float, default=0.6)
    ap.add_argument("--cone-min-height", type=float, default=0.5)
    ap.add_argument("--cone-max-height", type=float, default=1.4)

    # Диапазоны для тора
    ap.add_argument("--torus-min-radius", type=float, default=0.4)
    ap.add_argument("--torus-max-radius", type=float, default=0.8)
    ap.add_argument("--torus-min-tube-radius", type=float, default=0.1)
    ap.add_argument("--torus-max-tube-radius", type=float, default=0.3)

    # Плотность точек
    ap.add_argument("--points-per-unit-area", type=float, default=3000.0,
                    help="Точек на единицу площади поверхности")

    # Метод выборки
    ap.add_argument("--sampling-method", type=str, default="poisson", choices=["poisson", "uniform"],
                    help="Способ дискретизации поверхности")
    ap.add_argument("--poisson-init-factor", type=float, default=5.0,
                    help="init_factor для poisson-дискретизации")

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Создаем структуру директорий
    (args.root / "meta").mkdir(parents=True, exist_ok=True)

    # Соответствие сплитов и количества объектов
    split_counts = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class
    }

    # Генерируем данные для каждого запрошенного сплита
    for split_name in args.split:
        print(f"\n=== Генерация сплита {split_name.upper()} ===")
        generate_split(args, split_name, split_counts[split_name])

    print(f"\nГенерация завершена для сплитов: {', '.join(args.split)}")

if __name__ == "__main__":
    main()