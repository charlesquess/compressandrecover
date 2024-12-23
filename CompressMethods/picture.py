from PIL import Image
import random

# 设置图片尺寸为 4K
width, height = 3840, 2160

# 创建一个新图像
img = Image.new("RGB", (width, height))

# 获取图像的像素对象
pixels = img.load()

# 填充每个像素的RGB值为随机值
for i in range(width):
    for j in range(height):
        r = random.randint(0, 255)  # 红色通道
        g = random.randint(0, 255)  # 绿色通道
        b = random.randint(0, 255)  # 蓝色通道
        pixels[i, j] = (r, g, b)

# 保存生成的图像
img.save("random_4k_image.png")
