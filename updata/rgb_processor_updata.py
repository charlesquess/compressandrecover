from PIL import Image
from rgb_space import RGBSpace  

class ImageProcessor:
    def __init__(self, image_path, N):
        """
        初始化图像处理器。
        :param image_path: 图像文件路径
        :param N: RGB 空间的分割数量
        """
        self.image_path = image_path
        self.N = N
        self.rgb_space = RGBSpace(N)  # 创建 RGBSpace 对象
        self.image = Image.open(image_path)  # 打开图像
        self.width, self.height = self.image.size  # 获取图像的宽度和高度
        self.pixel_coordinates = []  # 存储处理后的像素信息

    def process_pixels(self):
        """
        处理每个像素，将 RGB 值映射到三维空间坐标。
        """
        for y in range(self.height):
            for x in range(self.width):
                # 获取当前像素的 RGB 值
                R, G, B = self.image.getpixel((x, y))

                # 计算该像素的相对坐标
                x_relative, y_relative, z_relative = self.rgb_space.get_coordinates_in_block(R, G, B)

                # 存储像素的位置信息、RGB值和相对坐标
                self.pixel_coordinates.append({
                    'pixel': (x, y),
                    'rgb': (R, G, B),
                    'relative_coordinates': (x_relative, y_relative, z_relative)
                })

    def get_pixel_data(self):
        """
        获取处理后的所有像素数据。
        :return: 所有像素的数据，包括 RGB 值和相对坐标
        """
        return self.pixel_coordinates

    def display_sample_data(self, sample_count=5):
        """
        显示前几个像素的相关数据。
        :param sample_count: 需要显示的样本数量
        """
        for data in self.pixel_coordinates[:sample_count]:
            print(f"像素位置: {data['pixel']}, RGB值: {data['rgb']}, 相对坐标: {data['relative_coordinates']}")
