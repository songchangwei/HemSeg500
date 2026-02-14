# ==============================================
# 环境依赖：pip install opencv-python numpy
# 整体功能：
# 1. 批量遍历根目录下的子文件夹，处理其中的灰度掩码JPG图
# 2. 对掩码图去噪、检测外轮廓，过滤小噪声轮廓
# 3. 计算每个有效轮廓的外接边界框（xmin,ymin,xmax,ymax）和质心坐标
# 4. 在彩色原图上绘制绿色边界框+红色实心质心，保存可视化结果
# 5. 将处理结果（文件夹名/有效文件名/边界框/质心）按JSONL格式追加保存
# 6. 每个子文件夹仅处理**第一张检测到有效轮廓**的掩码图
# ==============================================
import cv2          # 计算机视觉库：图像读取/绘制/轮廓检测/形态学操作
import numpy as np  # 数值计算库：处理图像数组（OpenCV图像本质是np.array）
import os           # 系统库：文件/目录的路径拼接/创建/遍历
import json         # 数据序列化库：Python字典转JSON字符串，用于JSONL写入

def get_boxes_and_points_from_mask(mask_image):
    """
    【核心算法】从灰度掩码图中提取有效轮廓的边界框和质心坐标
    :param mask_image: np.array，单通道灰度掩码图（黑底白前景，前景为待检测目标）
    :return: boxes: list，有效轮廓的边界框列表，每个元素为(xmin, ymin, xmax, ymax)
             points: list，有效轮廓的质心坐标列表，每个元素为(cx, cy)
             无有效轮廓时均返回None
    """
    # 定义形态学核：5x5全1数组，用于开运算去噪（核大小可根据噪声调整：噪声大则调大，如7x7）
    kernel = np.ones((5, 5), np.uint8)
    # 形态学开运算：先腐蚀后膨胀，消除掩码中比核小的孤立白色噪声点（不改变前景主体形状）
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)
    
    # 轮廓检测：仅检测最外层轮廓+压缩轮廓点（节省内存）
    # cv2.RETR_EXTERNAL：只提取外轮廓，忽略轮廓内部的子轮廓（适合掩码前景检测）
    # cv2.CHAIN_APPROX_SIMPLE：压缩轮廓点（如矩形仅保留4个顶点），而非存储所有像素点
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 无检测到任何轮廓，直接返回None
    if not contours:
        return None, None

    # 初始化结果列表：存储所有有效轮廓的边界框和质心
    boxes = []
    points = []

    # 遍历每个检测到的轮廓，过滤无效小轮廓并计算参数
    for contour in contours:
        # 过滤小轮廓：轮廓面积<5视为噪声（阈值可调整：目标小则调小，如3；目标大则调大，如10）
        if cv2.contourArea(contour) < 5:
            continue

        # 计算轮廓的外接矩形：返回(x, y, w, h)，x/y=矩形左上角坐标，w/h=宽/高
        x, y, w, h = cv2.boundingRect(contour)
        # 转换为目标检测通用的(xmin, ymin, xmax, ymax)格式，更易后续处理（如裁剪/坐标判断）
        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        boxes.append((x_min, y_min, x_max, y_max))  # 加入边界框列表

        # 计算轮廓质心：通过轮廓矩实现（计算机视觉计算几何中心的标准方法）
        M = cv2.moments(contour)  # 计算轮廓的所有矩（零阶/一阶/二阶矩等）
        # 零阶矩m00=轮廓面积，判断是否为0，避免除零错误
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # 质心x坐标 = 一阶矩m10 / 零阶矩m00
            cy = int(M["m01"] / M["m00"])  # 质心y坐标 = 一阶矩m01 / 零阶矩m00
        else:
            # 若m00=0（无效轮廓），则用外接矩形中心作为质心（兜底方案）
            cx, cy = x + w // 2, y + h // 2
        points.append((cx, cy))  # 加入质心列表

    return boxes, points

def find_and_save_first_valid_mask(directory, output_directory):
    """
    遍历指定文件夹，找到**第一张有效掩码图**，可视化绘制并保存结果，返回处理信息
    :param directory: str，待处理文件夹的完整路径（内含JPG掩码图）
    :param output_directory: str，可视化结果的保存路径
    :return: file_name: str，第一张有效掩码的文件名；None表示无有效文件
             boxes: list，有效轮廓的边界框列表；None表示无有效轮廓
             points: list，有效轮廓的质心坐标列表；None表示无有效轮廓
    """
    # 获取文件夹下所有文件并按文件名排序：保证处理顺序固定，避免系统随机遍历
    files = sorted(os.listdir(directory))
    
    # 遍历所有文件，筛选JPG后缀的掩码图
    for file in files:
        if file.endswith('.jpg'):
            # 拼接文件完整路径：os.path.join跨平台兼容（Windows/Linux路径分隔符自动适配）
            file_path = os.path.join(directory, file)
            # 两次读取同一文件：分别作为彩色原图（绘制用）和灰度掩码（轮廓检测用）
            # OpenCV默认读取彩色图为BGR格式（与PIL的RGB格式通道顺序相反）
            image = cv2.imread(file_path)  # 读取彩色原图（用于绘制边界框和质心）
            # 强制读取为单通道灰度图（掩码处理的标准方式，忽略颜色信息）
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # 掩码读取失败（如文件损坏），直接跳过当前文件
            if mask is None:
                continue

            # 调用核心算法，提取边界框和质心
            boxes, points = get_boxes_and_points_from_mask(mask)
            # 检测到有效轮廓（非空），开始可视化绘制
            if boxes and points:
                # 遍历每个轮廓的边界框和质心，逐个绘制
                for box, point in zip(boxes, points):
                    x_min, y_min, x_max, y_max = box
                    # 构造边界框的4个顶点：cv2.polylines需要np.array格式的点集
                    box_points = np.array([
                        [x_min, y_min],  # 左上角
                        [x_max, y_min],  # 右上角
                        [x_max, y_max],  # 右下角
                        [x_min, y_max]   # 左下角
                    ], dtype=np.int32)  # 显式指定int32，避免OpenCV类型警告
                    # 绘制绿色闭合边界框：isClosed=True=闭合图形，color=(0,255,0)=BGR格式绿色，thickness=2=线宽
                    cv2.polylines(image, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)
                    # 解包质心坐标
                    cx, cy = point
                    # 绘制红色实心质心圆：radius=5=半径，color=(0,0,255)=BGR格式红色，thickness=-1=实心
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                
                # 构造可视化结果的保存路径：文件名加前缀processed_，便于区分原文件
                output_path = os.path.join(output_directory, f"processed_{file}")
                print(f"【保存可视化结果】{output_path}")
                # 创建输出目录：os.makedirs支持多级目录创建，exist_ok=True=目录已存在时不报错（优化原代码的os.mkdir）
                os.makedirs(output_directory, exist_ok=True)
                # 保存绘制后的彩色原图
                cv2.imwrite(output_path, image)
                
                # 找到第一张有效文件，立即返回结果（不处理后续文件）
                return file, boxes, points

    # 遍历完所有文件均无有效掩码，返回None
    return None, None, None

def main(input_dir, output_dir, jsonl_file):
    """
    【批量处理主函数】遍历根输入目录的所有子文件夹，批量处理掩码图并保存JSONL结果
    :param input_dir: str，根输入目录的完整路径（内含多个子文件夹，每个子文件夹是一个样本集）
    :param output_dir: str，根输出目录的路径（会自动创建与输入子文件夹同名的子输出目录）
    :param jsonl_file: str，JSONL结果文件的路径（每行一个JSON对象，存储处理结果）
    :return: None
    """
    # 遍历根输入目录下的所有子文件夹（input_file_dir为子文件夹名称）
    for input_file_dir in os.listdir(input_dir):
        # 拼接子文件夹的完整输入路径：根输入目录 + 子文件夹名
        sub_input_path = os.path.join(input_dir, input_file_dir)
        # 拼接子文件夹的完整输出路径：根输出目录 + 子文件夹名（与输入目录一一对应，便于溯源）
        sub_output_path = os.path.join(output_dir, input_file_dir)
    
        # 处理当前子文件夹，找到第一张有效掩码图并保存可视化结果
        first_valid_file, boxes, points = find_and_save_first_valid_mask(sub_input_path, sub_output_path)
        
        # 构造JSONL单行数据：字典格式，字段含义清晰，便于后续解析
        result_data = {
            "sub_folder_name": input_file_dir,  # 子文件夹名称（样本集标识）
            "first_valid_mask_file": first_valid_file,  # 第一张有效掩码的文件名
            "bounding_boxes": boxes,  # 边界框列表，无有效轮廓则为None
            "centroid_points": points  # 质心坐标列表，无有效轮廓则为None
        }
        # 追加写入JSONL文件：'a'=追加模式（避免覆盖已有数据），每行一个JSON字符串
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            # json.dumps将Python字典转为JSON字符串，+ '\n'保证每行一个JSON对象（JSONL标准格式）
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
        
        # 控制台打印处理结果，方便实时查看
        if first_valid_file:
            print(f"\n【处理成功】子文件夹：{input_file_dir}")
            print(f"  第一张有效掩码文件：{first_valid_file}")
            print(f"  检测到有效轮廓数：{len(boxes)}")
            print(f"  边界框列表：{boxes}")
            print(f"  质心坐标列表：{points}\n")
        else:
            print(f"\n【处理失败】子文件夹：{input_file_dir} → 未检测到有效掩码轮廓\n")

# 主执行入口：只有直接运行该脚本时，以下代码才会执行（导入为模块时不执行）
if __name__ == "__main__":
    # ===================== 请根据自己的需求修改以下路径 =====================
    # 根输入目录：存放掩码图的根目录（内含多个子文件夹，每个子文件夹是一个样本集），建议用绝对路径
    ROOT_INPUT_DIR = '/home/user512-001/songcw/data/naochuxue/test/label_jpg'
    # 根输出目录：可视化结果的保存根目录（会自动创建子文件夹），相对路径/绝对路径均可
    ROOT_OUTPUT_DIR = 'mask'
    # JSONL结果文件：处理结果的保存路径，相对路径/绝对路径均可
    JSONL_RESULT_FILE = 'output_data.jsonl'
    # ======================================================================

    # 调用批量处理主函数
    main(ROOT_INPUT_DIR, ROOT_OUTPUT_DIR, JSONL_RESULT_FILE)
    print("【全部处理完成】结果已保存至JSONL文件：", JSONL_RESULT_FILE)