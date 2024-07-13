import time

import cv2
import numpy as np


def find_connected_components(mask):
    """
    找到掩码中的所有连通块。

    参数:
    - mask: 二值掩模图像，其中非零值表示感兴趣的区域。

    返回:
    - num_components: 连通块的数量。
    - components: 包含每个连通块的像素索引的数组。
    """
    # 将掩码转换为整数类型，因为connectedComponents要求掩码为整数类型
    mask = mask.astype(np.uint8)

    # 应用connectedComponents函数
    num_components, components = cv2.connectedComponents(mask)

    return num_components, components


def filter_small_components_by_bound(components, area_lower_bound):
    # 创建一个用于存储每个连通块像素数目的数组
    areas = np.bincount(components.flatten())

    # 找到所有面积大于或等于area_lower_bound的连通块的标签
    valid_labels = np.where(areas[1:] >= area_lower_bound)[0] + 1  # +1 因为数组从0开始，但标签从1开始

    # 使用有效连通块的标签创建一个布尔索引数组
    valid_mask = (components > 0)  # 背景标签为0，所以从1开始
    valid_mask[components != 0] = np.isin(components[components != 0], valid_labels)

    # 使用 valid_mask 找出所有有效连通块的标签
    valid_components = np.unique(components[valid_mask])
    # 返回布尔索引数组，它标记了所有有效连通块的像素
    return valid_components.size, valid_components


def find_extreme_points(image, component_mask):
    # 提取连通块的像素坐标
    points = np.argwhere(component_mask)

    # 找到最下最左的点
    min_row_index = np.argmin(points[:, 0])  # 找到行坐标中的最小值的索引
    bottom_left = (points[min_row_index, 0], points[min_row_index, 1])

    # 找到最下最右的点
    max_col_index = np.argmax(points[:, 1])  # 找到列坐标中的最大值的索引
    bottom_right = (points[max_col_index, 0], points[max_col_index, 1])

    # 计算几何中心
    if points.size > 0:
        center_of_mass = (int(np.mean(points[:, 0])), int(np.mean(points[:, 1])))
    else:
        return image  # 如果连通块为空，则直接返回原始图像

    # 在图像上标记点
    cv2.circle(image, bottom_left, 5, (0, 255, 0), -1)  # 绿色：最下最左
    cv2.circle(image, bottom_right, 5, (255, 0, 0), -1)  # 蓝色：最下最右
    cv2.circle(image, center_of_mass, 5, (0, 0, 255), -1)  # 红色：几何中心

    return bottom_left, bottom_right, center_of_mass

def get_red(image):
    # 将图像从BGR颜色空间转换到HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色在HSV空间的阈值
    # 注意：HSV的范围通常是[0, 180] for HUE, [0, 255] for SATURATION and VALUE
    # 下面的阈值是一个示例，您可能需要根据图像调整它们

    # 定义更宽泛的红色在HSV空间的阈值
    # 这里我们将色调的范围扩展到几乎整个红色区域，并降低饱和度的阈值来包含更浅的红色
    lower_red1 = np.array([0, 50, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 70])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩模
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 使用掩模将红色部分变为绿色
    # 将原图像转换为BGR颜色空间
    green = np.zeros_like(image, dtype=np.uint8)
    green[:, :] = [0, 255, 0]  # 绿色在BGR中的表示

    # image[mask != 0] = green[mask != 0]

    components_cnt, components = find_connected_components(mask)
    valid_components_cnt, valid_components = filter_small_components_by_bound(components, area_lower_bound=50)

    # 打印连通块的数量
    print(f"Number of connected components: {components_cnt}")
    print(f"Number of valid connected components: {valid_components_cnt}")

    # 可选：可视化连通块
    # 将连通块编号绘制到原图中
    for i in valid_components:  # 遍历所有有效连通块的标签
        color = (255 * (i % 3), 255 * ((i + 1) % 3),  255 * ((i + 2) % 3))  # BGR颜色：绿色
        # 创建一个布尔索引数组，对应当前连通块的像素
        component_mask = (components == i)
        # 将当前连通块的像素点填充为随机颜色
        image[component_mask] = color

        # 获取偏移量
        find_extreme_points(image, component_mask)
    return image

# 打开摄像头并实时处理视频帧
cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = get_red(frame)
        # 显示检测结果
        cv2.imshow("Track Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error in main loop: {e}")
        break

cap.release()
cv2.destroyAllWindows()
