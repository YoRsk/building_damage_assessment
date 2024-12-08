import numpy as np
import rasterio

def shift_image(input_path, output_path, fill_value=0):
    """
    将栅格数据向左偏移一个像素，保持地理参考信息
    
    参数:
        input_path: str, 输入栅格文件路径
        output_path: str, 输出栅格文件路径
        fill_value: int 或 float, 用于填充的值
    """
    # 打开源文件
    with rasterio.open(input_path) as src:
        # 读取数据
        data = src.read()
        
        # 创建结果数组
        result = data.copy()
        
        # 对每个波段进行处理
        for band in range(data.shape[0]):
            # 只向左移动像素
            result[band, :, :-1] = data[band, :, 1:]
            # 填充最右边一列
            result[band, :, -1] = fill_value
        
        # 创建输出文件，复制原始文件的元数据
        kwargs = src.meta.copy()
        
        # 写入新文件
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(result)

def main():
    input_path = './20210915_073716_84_2440_3B_Visual_clip.tif'
    output_path = input_path.replace('.tif', '_shifted.tif')
    
    try:
        shift_image(input_path, output_path, fill_value=0)
        print(f"处理完成，结果已保存至 {output_path}")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()