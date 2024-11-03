from PIL import Image
import rasterio
import os
import sys

def check_tif_info(image_path):
    """检查 TIF 图像的基本信息"""
    
    print(f"\n检查图像: {os.path.basename(image_path)}")
    print("-" * 50)
    
    # 使用 PIL 检查基本信息
    try:
        with Image.open(image_path) as img:
            print("PIL 信息:")
            print(f"图像模式: {img.mode}")
            print(f"图像大小: {img.size}")
            print(f"图像格式: {img.format}")
    except Exception as e:
        print(f"PIL 读取错误: {str(e)}")
    
    print("-" * 30)
    
    # 使用 rasterio 检查详细信息
    try:
        with rasterio.open(image_path) as src:
            print("Rasterio 信息:")
            print(f"波段数: {src.count}")
            print(f"数据类型: {src.dtypes[0]}")
            print(f"是否有地理信息: {src.crs is not None}")
            
            # 读取第一个波段的一些值来检查数值范围
            data = src.read(1)
            print(f"像素值范围: {data.min()} 到 {data.max()}")
    except Exception as e:
        print(f"Rasterio 读取错误: {str(e)}")
    
    print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("使用方法: python check_tif.py <tif_file_path> [tif_file_path2 ...]")
        return
    
    # 检查所有输入的图像
    for image_path in sys.argv[1:]:
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在 - {image_path}")
            continue
        
        check_tif_info(image_path)

if __name__ == "__main__":
    main()
