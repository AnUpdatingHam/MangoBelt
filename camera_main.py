import time

import cv2
import numpy as np

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

FRAME_RATE = 30

AREA_LOWER_BOUND = IMAGE_SIZE / 30

OBSTACLE_CENTRAL_LINES_THRESHOLD = 0.3
OBSTACLE_AREA_THRESHOLD = 0.45

OFFSET_SLIGHT_THRESHOLD = 0.25
OFFSET_MEDIUM_THRESHOLD = 0.4
OFFSET_EXTREME_THRESHOLD = 0.6

AUDIO_COOL_DOWN = 3.0 * FRAME_RATE

AUDIO_LEVEL_NORMAL = 0
AUDIO_LEVEL_WARNING = 2

__cur_audio_cooldown = 0.0
__cur_audio_level = 0

audios_dict = {
    "KEEP_FORWARD": 0,
    "OBSTACLE_FORWARD": 1,
    "SLIGHT_LEFT": 2,
    "SLIGHT_RIGHT": 3,
    "MEDIUM_LEFT": 4,
    "MEDIUM_RIGHT": 5,
    "EXTREME_LEFT": 6,
    "EXTREME_RIGHT": 7,
}

def play_audio(audio, audio_level):
    global __cur_audio_level  # 声明__cur_audio_level为全局变量
    global __cur_audio_cooldown  # 声明__cur_audio_cooldown为全局变量
    if audio_level > __cur_audio_level:
        print("play audio!")
        __cur_audio_level = audio_level
        __cur_audio_cooldown = AUDIO_COOL_DOWN
        return
    if __cur_audio_cooldown > 0:
        return
    print("play audio!")
    __cur_audio_level = audio_level
    __cur_audio_cooldown = AUDIO_COOL_DOWN


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


def filter_small_components_by_bound(components):
    # 创建一个用于存储每个连通块像素数目的数组
    areas = np.bincount(components.flatten())

    # 找到所有面积大于或等于area_lower_bound的连通块的标签
    valid_labels = np.where(areas[1:] >= AREA_LOWER_BOUND)[0] + 1  # +1 因为数组从0开始，但标签从1开始

    # 使用有效连通块的标签创建一个布尔索引数组
    valid_mask = (components > 0)  # 背景标签为0，所以从1开始
    valid_mask[components != 0] = np.isin(components[components != 0], valid_labels)

    # 使用 valid_mask 找出所有有效连通块的标签
    valid_components = np.unique(components[valid_mask])
    # 返回布尔索引数组，它标记了所有有效连通块的像素
    return valid_components.size, valid_components


def judge_direction_by_key_points(left_x, right_x, center_x):
    base_len = right_x - left_x
    offset = left_x + right_x - 2 * center_x
    return offset / base_len


def find_extreme_points(image, component_mask):
    # 步骤1：找到所有包含1的行的索引
    rows_with_ones = np.any(component_mask, axis=1)

    # 使用np.where找到所有包含True的行索引，然后取最后一个
    indices_with_ones = np.where(rows_with_ones)[0]
    if indices_with_ones.size > 0:
        bottom_row_index = indices_with_ones[-1]  # 获取最后一个索引
    else:
        return image  # 如果没有找到True，则直接返回原始图像

    # 步骤3：提取最底下的这一行，上提20个像素
    if bottom_row_index < 40:
        print("最后一行index不大于20")
        return 0, 0, 0
    bottom_row = component_mask[bottom_row_index - 20, :]

    # 找到最左边和最右边的1的索引
    leftmost_index = np.argmax(bottom_row)
    rightmost_index = len(bottom_row) - np.argmax(np.flipud(bottom_row))

    points = np.argwhere(component_mask)
    center_of_mass = (int(np.mean(points[:, 1])), int(np.mean(points[:, 0])))

    # 在图像上标记点
    cv2.circle(image, (leftmost_index, bottom_row_index), 5, (0, 255, 0), -1)  # 绿色：最下最左
    cv2.circle(image, (rightmost_index, bottom_row_index), 5, (255, 0, 0), -1)  # 蓝色：最下最右
    cv2.circle(image, center_of_mass, 5, (0, 0, 255), -1)  # 红色：几何中心

    # 绘制连线
    cv2.line(image, (leftmost_index, bottom_row_index), center_of_mass, (0, 0, 255), 2)  # 从最下最左到几何中心
    cv2.line(image, (rightmost_index, bottom_row_index), center_of_mass, (0, 0, 255), 2)  # 从最下最右到几何中心

    return leftmost_index, rightmost_index, center_of_mass[0]


def judge_obstacle_by_area(mask):
    bool_mask = mask.astype(bool)
    red_pixel_count = np.sum(bool_mask)
    return (red_pixel_count / IMAGE_SIZE) < OBSTACLE_AREA_THRESHOLD


def judge_obstacle_by_central_lines(mask):
    # 确定图像的宽度并计算中心线的 x 坐标
    midline_x = IMAGE_WIDTH // 2

    # 初始化计数器和y坐标
    false_count = 0
    countable_height = int(IMAGE_HEIGHT / 2)

    for y in range(countable_height, IMAGE_HEIGHT):
        if not mask[y, midline_x]:  # 如果 mask 的值为 False
            false_count += 1
    false_ratio = false_count / countable_height

    cv2.line(image, (midline_x, IMAGE_HEIGHT), (midline_x, countable_height), (0, 125, 125), 2)
    # print(f"false ratio: {false_ratio}")
    return false_ratio > OBSTACLE_CENTRAL_LINES_THRESHOLD


def get_red(image):
    # 将图像从BGR颜色空间转换到HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
    if judge_obstacle_by_area(mask):
        cv2.putText(image, "Obstacle by area", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        play_audio(audios_dict["OBSTACLE_FORWARD"], AUDIO_LEVEL_WARNING)
    if judge_obstacle_by_central_lines(mask):
        cv2.putText(image, "Obstacle by central", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        play_audio(audios_dict["OBSTACLE_FORWARD"], AUDIO_LEVEL_WARNING)

    components_cnt, components = find_connected_components(mask)
    valid_components_cnt, valid_components = filter_small_components_by_bound(components)

    left_offset_index = 0.0

    # 将连通块编号绘制到原图中
    for i in valid_components:  # 遍历所有有效连通块的标签
        color = (255 * (i % 3), 255 * ((i + 1) % 3),  255 * ((i + 2) % 3))  # BGR颜色：绿色
        # 创建一个布尔索引数组，对应当前连通块的像素
        component_mask = (components == i)
        # 将当前连通块的像素点填充为随机颜色
        image[component_mask] = color
        # 获取偏移量
        left_x, right_x, center_x = find_extreme_points(image, component_mask)
        left_offset_index += judge_direction_by_key_points(left_x, right_x, center_x)

    if len(valid_components) != 0:
        left_offset_index /= len(valid_components)
        if abs(left_offset_index) < OFFSET_SLIGHT_THRESHOLD:
            cv2.putText(image, "straight forward", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            play_audio(audios_dict["KEEP_FORWARD"], AUDIO_LEVEL_NORMAL)
        else:
            prefix = "SLIGHT"
            if abs(left_offset_index) > OFFSET_EXTREME_THRESHOLD:
                prefix = "EXTREME"
            elif abs(left_offset_index) > OFFSET_MEDIUM_THRESHOLD:
                prefix = "MEDIUM"

            suffix = "LEFT"
            if left_offset_index < 0:
                suffix = "RIGHT"

            cv2.putText(image, f"{prefix}_{suffix}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            play_audio(audios_dict[f"{prefix}_{suffix}"], AUDIO_LEVEL_NORMAL)

    return valid_components_cnt, valid_components

# 打开摄像头并实时处理视频帧
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

while True:
    try:
        ret, image = cap.read()
        __cur_audio_cooldown -= 1
        if not ret:
            print("Failed to grab frame")
            break

        valid_components_cnt, valid_components = get_red(image)

        # 显示检测结果
        cv2.imshow("Track Detection", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error in main loop: {e}")
        break

cap.release()
cv2.destroyAllWindows()
