import numpy as np

class RGBSpace:
    def __init__(self, N):
        """
        初始化RGB空间，将RGB空间分割成N^3块。
        :param N: 分割块的数量
        """
        self.MAX_VAL = 255  # RGB每个通道的最大值
        self.N = N  # 分割块的数量
        self.block_size = self.MAX_VAL // N  # 每个小块的边长

    def get_block_coords(self, i, j, k):
        """
        获取块(i, j, k)的坐标范围。
        :param i: 块的x坐标
        :param j: 块的y坐标
        :param k: 块的z坐标
        :return: 块的坐标范围(x_start, x_end), (y_start, y_end), (z_start, z_end)
        """
        x_start = (i / self.N) * self.MAX_VAL
        x_end = ((i + 1) / self.N) * self.MAX_VAL
        y_start = (j / self.N) * self.MAX_VAL
        y_end = ((j + 1) / self.N) * self.MAX_VAL
        z_start = (k / self.N) * self.MAX_VAL
        z_end = ((k + 1) / self.N) * self.MAX_VAL
        return (x_start, x_end), (y_start, y_end), (z_start, z_end)

    def map_rgb_to_block(self, R, G, B):
        """
        根据RGB值映射到空间的块(i, j, k)。
        :param R: 红色通道的值
        :param G: 绿色通道的值
        :param B: 蓝色通道的值
        :return: 块坐标(i, j, k)和该块的坐标范围
        """
        # 计算块的索引，确保在[0, N-1]范围内
        i = int(R / self.MAX_VAL * self.N)
        j = int(G / self.MAX_VAL * self.N)
        k = int(B / self.MAX_VAL * self.N)

        # 获取该块的坐标范围
        (x_start, x_end), (y_start, y_end), (z_start, z_end) = self.get_block_coords(i, j, k)
        return (i, j, k), (x_start, x_end), (y_start, y_end), (z_start, z_end)

    def map_block_to_rgb(self, i, j, k):
        """
        根据块(i, j, k)的坐标获取该块的RGB颜色范围中心值。
        :param i: 块的x坐标
        :param j: 块的y坐标
        :param k: 块的z坐标
        :return: 对应块的RGB值
        """
        # 获取块的坐标范围的中点
        x_start, x_end = self.get_block_coords(i, j, k)[0]
        y_start, y_end = self.get_block_coords(i, j, k)[1]
        z_start, z_end = self.get_block_coords(i, j, k)[2]
        
        # 返回RGB值的中点
        R = (x_start + x_end) / 2
        G = (y_start + y_end) / 2
        B = (z_start + z_end) / 2
        return int(R), int(G), int(B)

    def get_block_coordinates_and_rgb(self, R, G, B):
        # 获取块坐标和块的坐标范围
        block_coords, x_range, y_range, z_range = self.map_rgb_to_block(R, G, B)
        
        # 打印调试信息
        print(f"RGB({R}, {G}, {B}) 所在块坐标: {block_coords}")
        print(f"该块的坐标范围: X:{x_range} Y:{y_range} Z:{z_range}")
        
        # 返回块坐标和坐标范围
        return block_coords, (x_range, y_range, z_range)

    def get_block_center_rgb(self, i, j, k):
        """
        获取给定块坐标(i, j, k)的中心RGB值。
        :param i: 块的x坐标
        :param j: 块的y坐标
        :param k: 块的z坐标
        :return: 对应块中心的RGB值
        """
        return self.map_block_to_rgb(i, j, k)
    
    def get_coordinates_in_block(self, R, G, B):
        # 获取块的坐标和坐标范围
        block_coords, x_range, y_range, z_range = self.map_rgb_to_block(R, G, B)

        # 从范围中提取起始和结束值
        x_start, x_end = x_range
        y_start, y_end = y_range
        z_start, z_end = z_range

        # 计算该 RGB 值在块内的相对坐标
        x_relative = (R - x_start) / (x_end - x_start)
        y_relative = (G - y_start) / (y_end - y_start)
        z_relative = (B - z_start) / (z_end - z_start)

        # 打印调试信息
        print(f"RGB({R}, {G}, {B}) 在所属块中的相对坐标: X:{x_relative:.2f}, Y:{y_relative:.2f}, Z:{z_relative:.2f}")
        
        return x_relative, y_relative, z_relative

# 使用示例
if __name__ == "__main__":
    rgb_space = RGBSpace(N=50)
    
    # RGB值为(200, 50, 150)的映射
    R, G, B = 200, 50, 150
    block_coords, block_range = rgb_space.get_block_coordinates_and_rgb(R, G, B)
    print(f"RGB({R}, {G}, {B}) 所在块坐标: {block_coords}")
    print(f"该块的坐标范围: X:{block_range[0]} Y:{block_range[1]} Z:{block_range[2]}")
    
    # 获取RGB值为块(5, 3, 7)的中心RGB值
    center_rgb = rgb_space.get_block_center_rgb(5, 3, 7)
    print(f"块(5, 3, 7)的中心RGB值为: {center_rgb}")
    
    # 获取RGB值在所属块中的相对坐标
    x_relative, y_relative, z_relative = rgb_space.get_coordinates_in_block(R, G, B)
    print(f"RGB({R}, {G}, {B})在所属块中的相对坐标: X:{x_relative:.2f}, Y:{y_relative:.2f}, Z:{z_relative:.2f}")
