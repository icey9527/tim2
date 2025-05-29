import os
import argparse
from PIL import Image
import struct

def process_tm2_file(input_path, output_dir):
    with open(input_path, 'rb') as f:
        data = f.read()
    
    # 验证文件大小
    if len(data) < 0x40:
        print(f"文件过小: {input_path}")
        return
    
    # 检查像素大小（应为8位）

    
    # 读取宽度和高度（小端序）
    width = struct.unpack('<H', data[0x24:0x26])[0]
    height = struct.unpack('<H', data[0x26:0x28])[0]
    
    # 计算像素数据大小
    pixel_data_size = struct.unpack('<I', data[0x18:0x1C])[0]
    palette_start = 0x40 + pixel_data_size
    
    # 提取像素数据
    pixels = data[0x40:0x40 + pixel_data_size]
    palette_data = data[0x40 + pixel_data_size:]
    cont = struct.unpack('<H', data[0x1E:0x20])[0]
    # 解析调色板
        # 原始调色板处理（完全保持原有逻辑）
    original_palette = []
    for i in range(cont):

        pos = i * 4
        r = palette_data[pos]
        g = palette_data[pos + 1]
        b = palette_data[pos + 2]
        a = palette_data[pos + 3]
        a = min(a * 2 - 1, 255)
        
        original_palette.append((r, g, b, a))
    
    # 调色板重排逻辑（完全保持原有算法）
    if cont == 256:
        palette = []
        for major_group_start in range(0, 256, 32):  # 每32色一个大组
            major_group = original_palette[major_group_start:major_group_start+32]
            subgroup1 = major_group[0:8]    # 第1小组
            subgroup2 = major_group[8:16]   # 第2小组
            subgroup3 = major_group[16:24]  # 第3小组
            subgroup4 = major_group[24:32]  # 第4小组
            reordered_major_group = subgroup1 + subgroup3 + subgroup2 + subgroup4
            palette.extend(reordered_major_group)
    else:
        palette = original_palette
    
    # 创建RGBA图像
    img = Image.new('RGBA', (width, height))
    img_data = []

    if cont == 256:
        for i in range(pixel_data_size):
            if i == width * height:
                break
            color_idx = pixels[i]
            img_data.append(palette[color_idx])
    elif cont == 16:
        for i in range(pixel_data_size):
            if i == width * height // 2:
                break            
            byte_value = pixels[i]
            img_data.append(palette[(byte_value & 0xF)])
            img_data.append(palette[(byte_value >> 4) & 0xF])
            
    elif cont == 0:
        for i in range(0, pixel_data_size, 4):
            img_data.append((pixels[i], pixels[i+1], pixels[i+2], min(pixels[i+3] * 2 - 1, 255)))
    
    
    img.putdata(img_data)
    
    # 保存PNG文件
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
    img.save(output_path, 'PNG')
    print(f"已转换: {input_path} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description='转换TM2纹理文件为PNG格式')
    parser.add_argument('input_dir', help='包含TM2文件的输入目录')
    parser.add_argument('output_dir', help='PNG文件的输出目录')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有tm2文件
    count = 0
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith('.tm2'):
            input_path = os.path.join(args.input_dir, filename)
            try:
                process_tm2_file(input_path, args.output_dir)
                count += 1
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    print(f"\n转换完成! 共处理 {count} 个文件")

if __name__ == '__main__':
    main()