import cv2
import numpy as np

# 读取两张图像
image1 = cv2.imread('data/boat1.jpg')
image2 = cv2.imread('data/boat2.jpg')

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点并计算描述符
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 使用FLANN匹配器进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 应用比率测试筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 使用RANSAC算法估计单应性矩阵
if len(good_matches) > 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # 获取图像尺寸
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 计算变换后image1的四个角点
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)

    # 计算拼接画布的大小
    all_corners = np.concatenate((corners1_transformed,
                                  np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]],
                                           dtype=np.float32).reshape(-1, 1, 2)))

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 计算平移变换矩阵
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    # 应用平移变换
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    panorama = cv2.warpPerspective(image1, H_translation.dot(H), (panorama_width, panorama_height))

    # 将image2放入全景图中
    panorama[translation_dist[1]:translation_dist[1] + h2,
    translation_dist[0]:translation_dist[0] + w2] = image2

    # 改进的重叠区域融合
    # 创建一个mask来确定重叠区域
    warped_image1 = cv2.warpPerspective(image1, H_translation.dot(H), (panorama_width, panorama_height))
    mask1 = np.zeros((panorama_height, panorama_width), np.uint8)
    cv2.fillConvexPoly(mask1, np.int32(corners1_transformed + translation_dist), 255)

    mask2 = np.zeros((panorama_height, panorama_width), np.uint8)
    cv2.rectangle(mask2, (translation_dist[0], translation_dist[1]),
                  (translation_dist[0] + w2, translation_dist[1] + h2), 255, -1)

    overlap_mask = cv2.bitwise_and(mask1, mask2)

    # 使用加权平均融合重叠区域
    if np.any(overlap_mask):
        overlap_area = np.where(overlap_mask)
        panorama[overlap_area] = cv2.addWeighted(
            warped_image1[overlap_area], 0.5,
            panorama[overlap_area], 0.5, 0)

    # 裁剪黑色边界
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        panorama = panorama[y:y + h, x:x + w]

    # 显示和保存结果
    cv2.imshow('Improved Panorama', panorama)
    cv2.imwrite('results/improved_panorama.jpg', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches are found - {}/{}".format(len(good_matches), 4))