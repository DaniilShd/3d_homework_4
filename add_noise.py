# quick_noise.py - упрощенная версия
import numpy as np
import open3d as o3d
from pathlib import Path

def add_noise_to_dataset(input_dir="dataset", output_dir="dataset_noisy", 
                        sigma_rel=0.01, p_out=0.05, bbox_expand=1.2):
    """
    Функция для добавления шума ко всему датасету
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Находим все PLY файлы
    ply_files = list(input_path.rglob("*.ply"))
    
    print(f"Найдено {len(ply_files)} файлов для обработки")
    
    for ply_file in ply_files:
        # Определяем выходной путь
        relative_path = ply_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        # Создаем директорию если нужно
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Загружаем и обрабатываем
        pcd = o3d.io.read_point_cloud(str(ply_file))
        points = np.asarray(pcd.points)
        
        # Гауссовский шум
        bbox_size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        noise = np.random.normal(0, sigma_rel * bbox_size, points.shape)
        noisy_points = points + noise
        
        # Выбросы
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        expanded_size = (bbox_max - bbox_min) * bbox_expand
        
        n_outliers = int(len(points) * p_out)
        outliers = np.random.uniform(
            low=bbox_center - expanded_size/2,
            high=bbox_center + expanded_size/2,
            size=(n_outliers, 3)
        )
        
        final_points = np.vstack([noisy_points, outliers])
        
        # Сохраняем
        noisy_pcd = o3d.geometry.PointCloud()
        noisy_pcd.points = o3d.utility.Vector3dVector(final_points)
        o3d.io.write_point_cloud(str(output_file), noisy_pcd)
        
        print(f"Обработан: {relative_path}")

if __name__ == "__main__":
    add_noise_to_dataset()