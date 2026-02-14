# ==============================================
# 环境依赖：pip install opencv-python numpy
# 整体功能：
# 1. 批量遍历根目录下的子文件夹，处理灰度掩码JPG图，去噪+外轮廓检测+过滤小噪声
# 2. 对每个轮廓：先算整体质心→以其为中心饼状扇形等分360°→生成【1-10全量】个质心
# 3. 可视化拆分：10个点数分别绘制到10张独立图像，每张图仅含「绿色边界框+对应数量饼状质心」
# 4. 批量保存10张可视化结果，JSONL格式存储所有点数质心+边界框数据
# 5. 每个子文件夹仅处理**第一张检测到有效轮廓**的掩码图
# ==============================================
import cv2
import numpy as np
import os
import json

# ===================== 可配置常量（核心修改：扩充1-10全量点数） =====================
POINT_COUNTS = [1,2,3,4,5,6,7,8,9,10]  # 扩充为1-10全量点数，核心修改点1
MIN_CONTOUR_AREA = 5  # 轮廓面积过滤阈值，过小轮廓视为噪声
MORPH_KERNEL_SIZE = (5, 5)  # 形态学开运算去噪核大小
# 不同点数质心的绘制配置：(颜色BGR, 半径, 字体大小)，补充4/6/7/8/9配置，核心修改点2
# 规律：点数↑ → 半径↓ + 字体↓；颜色差异化，无重复，保证可视化区分度
DRAW_CONFIG = {
    1: ((0, 0, 255), 6, 0.5),    # 1个点：红色、最大半径、最大字体
    2: ((255, 0, 0), 5, 0.45),   # 2个点：蓝色、次大半径、次大字体
    3: ((0, 255, 255), 5, 0.45), # 3个点：黄色、次大半径、次大字体
    4: ((255, 0, 255), 4, 0.4),  # 4个点：玫红、中半径、中等字体（新增）
    5: ((255, 255, 0), 4, 0.4),  # 5个点：青色、中半径、中等字体
    6: ((0, 128, 0), 3, 0.35),   # 6个点：深绿、小半径、微字体（新增）
    7: ((255, 165, 0), 3, 0.35), # 7个点：橙色、小半径、微字体（新增）
    8: ((0, 128, 128), 2, 0.3),  # 8个点：墨蓝、更小半径、最小字体（新增）
    9: ((128, 128, 0), 2, 0.3),  # 9个点：橄榄黄、更小半径、最小字体（新增）
    10: ((128, 0, 128), 2, 0.3)  # 10个点：紫色、最小半径、最小字体
}
# ====================================================================================

def calc_pie_even_centroids(contour, mask_shape):
    """
    【核心饼状划分函数】对单个轮廓，以**整体质心为中心饼状扇形等分**生成指定数量的质心点
    核心原理：
        1. 计算轮廓整体质心（饼状划分的极点/中心）
        2. 将所有前景像素转换为「以整体质心为原点」的极坐标，提取极角（0-360°）
        3. 将360°均分为N个扇形区间（饼状区域），按极角将像素分到对应区间
        4. 计算每个扇形区间内像素的质心，即为饼状区域的质心
    :param contour: np.array，单个轮廓的点集（cv2.findContours返回的轮廓）
    :param mask_shape: tuple，原掩码图像的尺寸 (h, w)，用于创建轮廓独立掩码
    :return: dict，key=点数（1-10），value=该点数下的饼状质心列表[(cx1,cy1), ...]
    """
    # 1. 创建轮廓独立掩码，仅填充当前轮廓（避免与其他轮廓混淆）
    single_contour_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(single_contour_mask, [contour], -1, 255, cv2.FILLED)

    # 2. 提取轮廓所有前景像素坐标（x,y），总像素数=轮廓面积
    y_coords, x_coords = np.where(single_contour_mask == 255)
    total_pix = len(x_coords)
    if total_pix == 0:  # 空轮廓，返回空字典
        return {n: [] for n in POINT_COUNTS}
    pix_points = np.array(list(zip(x_coords, y_coords)))  # (N,2) 像素坐标数组

    # 3. 计算【轮廓整体质心】（饼状划分的中心/极点），用轮廓矩实现（标准方法）
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx0 = int(M["m10"] / M["m00"])  # 整体质心x
        cy0 = int(M["m01"] / M["m00"])  # 整体质心y
    else:
        cx0, cy0 = np.mean(x_coords).astype(int), np.mean(y_coords).astype(int)
    center = np.array([cx0, cy0])  # 饼状中心坐标

    # 4. 计算所有像素点**相对于整体质心**的极角（核心步骤：极坐标转换）
    dx = pix_points[:, 0] - center[0]  # 每个像素x方向偏移量
    dy = pix_points[:, 1] - center[1]  # 每个像素y方向偏移量
    # np.arctan2(dy, dx)：返回四象限极角（弧度，范围-π~π），符合OpenCV图像坐标系
    angles_rad = np.arctan2(dy, dx)
    # 弧度转角度 + 统一转换为**0-360°正角度**（负角度转正，避免区间偏移）
    angles_deg = np.degrees(angles_rad)
    angles_deg = np.where(angles_deg < 0, angles_deg + 360, angles_deg)

    # 5. 遍历每个需要生成的点数（1-10），按角度均分360°为饼状区域，计算各区域质心
    centroid_dict = {}
    for n in POINT_COUNTS:
        if n == 1:  # 特殊情况：1个点=整体质心，无需划分
            centroid_dict[n] = [(cx0, cy0)]
            continue
        # 计算每个饼状扇形的**角度步长**（360°均分为n份，自动适配1-10所有n）
        angle_step = 360.0 / n
        # 生成每个扇形区域的**角度区间** [start_deg, end_deg)，自动适配任意n
        angle_intervals = [(i * angle_step, (i + 1) * angle_step) for i in range(n)]
        centroids = []
        # 遍历每个角度区间（每个区间=一个饼状扇形区域）
        for start_deg, end_deg in angle_intervals:
            # 筛选出当前角度区间内的所有像素点，处理浮点误差避免漏分
            if start_deg < 360 - 1e-6:  # 常规区间：[start, end)
                mask = (angles_deg >= start_deg) & (angles_deg < end_deg)
            else:  # 最后一个区间：包含360°（等于0°）的像素
                mask = (angles_deg >= start_deg) | (angles_deg < end_deg)
            region_pix = pix_points[mask]
            # 计算当前饼状区域的质心
            if len(region_pix) > 0:
                # 有像素：取像素坐标的平均值（取整），为区域几何中心
                cx = int(np.mean(region_pix[:, 0]))
                cy = int(np.mean(region_pix[:, 1]))
            else:
                # 无像素（空区间）：兜底用**整体质心**，保证点数数量一致
                cx, cy = cx0, cy0
            centroids.append((cx, cy))
        centroid_dict[n] = centroids

    return centroid_dict

def get_boxes_and_multi_centroids(mask_image):
    """
    【核心函数】从灰度掩码图提取轮廓，返回边界框+饼状划分的1-10点质心字典
    :param mask_image: np.array，单通道灰度掩码图（黑底白前景）
    :return: boxes: 边界框列表[(xmin,ymin,xmax,ymax), ...]
             multi_centroids: 质心字典，key=1-10，value=对应饼状质心列表
    """
    # 形态学开运算去噪（无修改）
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)
    
    # 外轮廓检测（无修改）
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], {n: [] for n in POINT_COUNTS}

    boxes = []
    multi_centroids = {n: [] for n in POINT_COUNTS}

    # 遍历轮廓，过滤小噪声并计算边界框+饼状质心（自动适配1-10点）
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        # 计算外接边界框（无修改）
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
        # 计算当前轮廓的饼状质心并追加（自动适配1-10点）
        contour_centroids = calc_pie_even_centroids(contour, mask_image.shape)
        for n in POINT_COUNTS:
            multi_centroids[n].extend(contour_centroids[n])

    return boxes, multi_centroids

def find_and_save_first_valid_mask(directory, output_directory):
    """
    遍历文件夹找到第一张有效掩码图，按1-10点数拆分绘制**10张独立图像**并保存
    每张图仅含：绿色边界框 + 对应数量的饼状质心（带序号标注）
    :param directory: 待处理文件夹路径
    :param output_directory: 可视化结果保存路径
    :return: file_name/None: 第一张有效掩码文件名
             boxes/[]: 边界框列表
             multi_centroids/{}: 1-10点饼状质心字典
    """
    files = sorted(os.listdir(directory))
    for file in files:
        if file.endswith('.jpg'):
            file_path = os.path.join(directory, file)
            # 读取彩色原图（绘制用）和灰度掩码（轮廓检测用）
            image_origin = cv2.imread(file_path)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or image_origin is None:
                continue

            # 获取边界框和1-10点的饼状质心（自动适配，无修改）
            boxes, multi_centroids = get_boxes_and_multi_centroids(mask)
            if not boxes:  # 无有效轮廓，跳过
                continue

            # 循环1-10每个点数，绘制并保存**10张独立图像**（自动遍历POINT_COUNTS，无修改）
            for n in POINT_COUNTS:
                # 深拷贝原始原图作为独立画布，避免绘制叠加
                image_draw = image_origin.copy()
                # 1. 绘制所有轮廓的绿色边界框（每张图都保留，保证轮廓完整性）
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    box_points = np.array([[x_min, y_min], [x_max, y_min],
                                           [x_max, y_max], [x_min, y_max]], dtype=np.int32)
                    cv2.polylines(image_draw, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)

                # 2. 绘制当前点数的饼状质心+序号标注（自动适配1-10点的绘制配置）
                color, radius, font_scale = DRAW_CONFIG[n]
                centroids = multi_centroids[n]
                for idx, (cx, cy) in enumerate(centroids, 1):
                    cv2.circle(image_draw, (cx, cy), radius, color, -1)  # 绘制实心质心圆
                    label = f"{n}-{idx}"  # 标注：点数-序号（如4-2、9-5）
                    cv2.putText(image_draw, label, (cx + 8, cy + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

                # 3. 构造当前点数的独立保存路径，后缀加_count{n}区分（自动适配1-10）
                output_file_name = f"processed_{file}_count{n}.jpg"
                output_path = os.path.join(output_directory, output_file_name)
                os.makedirs(output_directory, exist_ok=True)
                cv2.imwrite(output_path, image_draw)
                print(f"【保存{n}个饼状点】{output_path}")

            # 所有10张图像保存完成后，返回结果
            return file, boxes, multi_centroids

    # 无有效掩码文件，返回空结果
    return None, [], {n: [] for n in POINT_COUNTS}

def main(input_dir, output_dir, jsonl_file):
    """
    批量处理主函数：遍历根目录子文件夹，处理掩码并保存JSONL结果（无修改）
    JSONL自动保存1-10所有点数的饼状质心，数据结构与之前一致
    :param input_dir: 根输入目录（内含多个子文件夹）
    :param output_dir: 根输出目录（自动创建子文件夹，存放10张/个的可视化图）
    :param jsonl_file: JSONL结果文件路径
    :return: None
    """
    for sub_folder_name in os.listdir(input_dir):
        sub_input_path = os.path.join(input_dir, sub_folder_name)
        if not os.path.isdir(sub_input_path):  # 跳过根目录下的非文件夹文件
            continue
        sub_output_path = os.path.join(output_dir, sub_folder_name)

        # 处理当前子文件夹，拆分保存10张可视化图，获取1-10点饼状质心结果
        first_file, boxes, multi_centroids = find_and_save_first_valid_mask(sub_input_path, sub_output_path)

        # 构造JSONL数据（自动适配1-10点，结构无修改）
        result_data = {
            "output_directory": sub_folder_name,
            "file": first_file,
            "boxes": boxes,
            "points": {str(n): multi_centroids[n] for n in POINT_COUNTS}  # 点数转字符串适配JSON
        }

        # 追加写入JSONL文件（utf-8编码，避免中文/特殊字符报错）
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')

        # 控制台打印处理结果，实时查看（自动适配1-10点）
        if first_file:
            print(f"\n【处理完成】子文件夹：{sub_folder_name}")
            print(f"  有效掩码文件：{first_file} | 有效轮廓数：{len(boxes)}")
            print(f"  已生成：{POINT_COUNTS}个饼状点，共10张可视化图像\n")
        else:
            print(f"\n【处理失败】子文件夹：{sub_folder_name} → 无有效掩码轮廓\n")
        print("-" * 100)  # 分隔线，优化控制台阅读体验

if __name__ == "__main__":
    # ===================== 请根据自己的需求修改以下3个路径 =====================
    ROOT_INPUT_DIR = '/home/user512-003/songcw/data/naochuxue/test/label_jpg'  # 掩码根输入目录（绝对路径）
    ROOT_OUTPUT_DIR = 'mask2'  # 可视化结果根输出目录（相对/绝对路径均可）
    JSONL_RESULT_FILE = 'output_pie_centroids.jsonl'  # 建议改名区分1-10全量结果
    # ===========================================================================

    # 【可选】若需要覆盖原有JSONL数据，取消以下2行注释（默认追加模式）
    # if os.path.exists(JSONL_RESULT_FILE):
    #     os.remove(JSONL_RESULT_FILE)

    # 执行批量处理
    main(ROOT_INPUT_DIR, ROOT_OUTPUT_DIR, JSONL_RESULT_FILE)
    print(f"\n【全部处理完成】")
    print(f"1. 可视化图像：每个子文件夹下生成10张，对应{POINT_COUNTS}个饼状点")
    print(f"2. 数据文件：所有饼状质心已保存至 → {JSONL_RESULT_FILE}")