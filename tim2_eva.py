# tim2_batch.py
import struct
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse

class TIM2Parser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, 'rb') as f:
            self.data = bytearray(f.read())
        self._parse_header()
        self._find_picture_headers()

    def _parse_header(self):
        self.magic = self.data[0:4].decode('ascii', errors='ignore')
        if self.magic != 'TIM2':
            raise ValueError(f'{self.filepath}: not a TIM2 file')
        self.version = struct.unpack('<H', self.data[0x04:0x06])[0]
        self.picture_count = struct.unpack('<H', self.data[0x06:0x08])[0]
        self.total_width = struct.unpack('<H', self.data[0x08:0x0A])[0]
        self.total_height = struct.unpack('<H', self.data[0x0A:0x0C])[0]

    def _find_picture_headers(self):
        # 图片块从 0x80 开始，按 header["total_size"] 顺序累加
        self.picture_headers = []
        offset = 0x80
        for _ in range(self.picture_count):
            if offset + 0x28 > len(self.data):
                break
            header = self._parse_picture_header_at(offset)
            self.picture_headers.append((offset, header))
            offset += header['total_size']

    def _parse_picture_header_at(self, offset: int):
        h = {}
        # TIM2 picture header (常见布局)
        h['total_size']   = struct.unpack('<I', self.data[offset+0x00:offset+0x04])[0]
        h['palette_size'] = struct.unpack('<I', self.data[offset+0x04:offset+0x08])[0]
        h['image_size_b'] = struct.unpack('<I', self.data[offset+0x08:offset+0x0C])[0]
        h['header_size']  = struct.unpack('<H', self.data[offset+0x0C:offset+0x0E])[0] or 0x80
        h['color_count']  = struct.unpack('<H', self.data[offset+0x0E:offset+0x10])[0]
        h['format']       = self.data[offset+0x10]
        h['mipmap_count'] = self.data[offset+0x11]
        h['clut_format']  = self.data[offset+0x12]
        h['bpp']          = self.data[offset+0x13]
        h['width']        = struct.unpack('<H', self.data[offset+0x14:offset+0x16])[0]
        h['height']       = struct.unpack('<H', self.data[offset+0x16:offset+0x18])[0]
        # 可选 GS 寄存器（不用也行）
        # h['gs_tex0']   = struct.unpack('<Q', self.data[offset+0x18:offset+0x20])[0]
        # h['gs_tex1']   = struct.unpack('<Q', self.data[offset+0x20:offset+0x28])[0]
        h['image_offset']   = offset + h['header_size']
        h['palette_offset'] = h['image_offset'] + h['image_size_b']
        return h

    def extract_image(self, index: int):
        """仅实现 32 位 RGBA (format 0x00/0x03)，并做 alpha 调整"""
        if index >= len(self.picture_headers):
            return None
        _, h = self.picture_headers[index]
        fmt = h['format']
        w, hgt = h['width'], h['height']
        data_off = h['image_offset']

        # 仅处理 32bpp（常见场景）
        if fmt not in (0x00, 0x03):
            print(f'  Skip picture {index}: unsupported format 0x{fmt:02X}')
            return None

        pixel_bytes = w * hgt * 4
        if data_off + pixel_bytes > len(self.data):
            print(f'  Data overflow for picture {index}')
            return None

        buf = self.data[data_off:data_off + pixel_bytes]
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((hgt, w, 4))

        # 不再做通道调换（保持 RGBA）
        # Alpha 增强：a = min(a*2 - 1, 255)，并裁下限到 0 以防负值
        a = arr[:, :, 3].astype(np.int32)
        a = np.minimum(a * 2 - 1, 255)
        a = np.clip(a, 0, 255).astype(np.uint8)
        arr[:, :, 3] = a

        return Image.fromarray(arr, mode='RGBA')

    def extract_all_images(self):
        imgs = []
        for i in range(len(self.picture_headers)):
            img = self.extract_image(i)
            imgs.append(img)
        return imgs

    def _infer_column_height(self, images):
        """在 total_height==0 时，智能推断一个列高，使得能将序列按列高分组"""
        heights = [im.height for im in images if im is not None]
        if not heights:
            return 0

        # 典型高度优先（PS2常见拼法）
        typical = [448, 480, 512, 384, 320, 256]
        # 再加上一些“前缀和”候选（即前n块相加得到的高度）
        prefix = []
        s = 0
        for h in heights[:min(6, len(heights))]:
            s += h
            prefix.append(s)

        # 合并候选，去重保持顺序
        cand = []
        seen = set()
        for v in typical + prefix:
            if v > 0 and v not in seen:
                seen.add(v)
                cand.append(v)

        def can_partition(H):
            acc = 0
            for h in heights:
                acc += h
                if acc == H:
                    acc = 0
                elif acc > H:
                    return False
            return acc == 0

        for H in cand:
            if can_partition(H):
                return H

        # 实在不行就把所有块当成一列
        return sum(heights)

    def compose_columns(self, images):
        """
        列优先拼接（回到你的原始算法）：
        - 总高度非 0：用总高度为列高
        - 总高度为 0：自动推断列高
        - 每列从上到下堆，列宽取该列中最大块宽
        - 总宽度非 0：用总宽度；否则为各列宽之和
        """
        # 1) 确定列高
        if self.total_height > 0:
            col_target_h = self.total_height
        else:
            col_target_h = self._infer_column_height(images)

        # 2) 第一遍走位，计算每个块的位置和各列宽度
        positions = []
        col_widths = []
        x = 0
        col_y = 0
        col_w = 0

        for im in images:
            if im is None:
                continue
            # 如果放不下当前列，先结算本列，换列
            if col_y > 0 and col_y + im.height > col_target_h:
                col_widths.append(col_w)
                x += col_w
                col_y = 0
                col_w = 0
            # 放到当前列
            positions.append((x, col_y))
            col_y += im.height
            col_w = max(col_w, im.width)

            # 如果正好填满一列，则结算并开新列
            if col_y == col_target_h:
                col_widths.append(col_w)
                x += col_w
                col_y = 0
                col_w = 0

        # 收尾：最后一列如果有内容，结算之
        if col_y > 0 or (col_w > 0 and (not col_widths)):
            col_widths.append(col_w)
            x += col_w

        computed_w = sum(col_widths) if col_widths else 0
        # 3) 确定画布大小
        if self.total_width > 0 and self.total_height > 0:
            canvas_w = self.total_width
            canvas_h = self.total_height
        else:
            canvas_w = computed_w if computed_w > 0 else max(im.width for im in images if im)
            canvas_h = col_target_h

        # 4) 画布并粘贴
        canvas = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        for im, (px, py) in zip(images, positions):
            if im is None:
                continue
            canvas.paste(im, (px, py))

        # 一些提示（不影响结果）
        if self.total_width > 0 and canvas_w != self.total_width:
            print(f'  Warn: composed width {canvas_w} != total_width {self.total_width}')
        if self.total_height > 0 and canvas_h != self.total_height:
            print(f'  Warn: composed height {canvas_h} != total_height {self.total_height}')

        return canvas


def process_tex_file(in_path: Path, out_dir: Path, save_parts: bool = False):
    try:
        parser = TIM2Parser(str(in_path))
        imgs = parser.extract_all_images()

        # 保存分块（可选）
        if save_parts:
            parts_dir = out_dir / (in_path.stem + '_parts')
            parts_dir.mkdir(parents=True, exist_ok=True)
            for idx, im in enumerate(imgs):
                if im is not None:
                    im.save(parts_dir / f'picture_{idx:02d}.png')

        # 合成整图
        combined = parser.compose_columns(imgs)
        combined_path = out_dir / f'{in_path.stem}.png'
        combined.save(combined_path)
        print(f'[OK] {in_path.name} -> {combined_path}')
    except Exception as e:
        print(f'[ERR] {in_path}: {e}')


def main():
    ap = argparse.ArgumentParser(description='Batch convert PS2 TIM2 .TEX to PNG')
    ap.add_argument('-i', '--input', required=True, help='input .tex file or directory')
    ap.add_argument('-o', '--output', required=True, help='output directory')
    ap.add_argument('--parts', action='store_true', help='also save split parts for each tex')
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        if in_path.suffix.lower() == '.tex':
            process_tex_file(in_path, out_dir, save_parts=args.parts)
        else:
            print(f'{in_path} is not a .tex file')
    else:
        # 递归批量
        files = list(in_path.rglob('*.tex'))
        if not files:
            print('No .tex files found')
            return
        for fp in files:
            # 保持子目录结构
            rel = fp.parent.relative_to(in_path)
            dst = out_dir / rel
            dst.mkdir(parents=True, exist_ok=True)
            process_tex_file(fp, dst, save_parts=args.parts)


if __name__ == '__main__':
    main()