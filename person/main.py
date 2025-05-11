import cv2

# 读取图像列表，假设图像位于 'data' 目录下
images = [cv2.imread(f"data/boat{i}.jpg") for i in [1, 2]]  # 替换为您的图片文件名

# 创建 Stitcher 对象
stitcher = cv2.Stitcher_create()

# 执行图像拼接
status, panorama = stitcher.stitch(images)

# 检查拼接是否成功
if status == cv2.Stitcher_OK:
    # 拼接成功，保存结果到 'results' 目录下的 'results.jpg'
    cv2.imshow('Panorama', panorama)
    cv2.imwrite("results/result.jpg", panorama)
else:
    print("图像拼接失败！")
