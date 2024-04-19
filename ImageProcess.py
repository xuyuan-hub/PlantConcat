import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import zipfile

OFFSET = 150
Y1 = [186, 569, 955, 1336, 1715, 2095.5, 2479.5, 2862.5]
X1 = [250, 618.625, 987.5, 1376.5, 1855, 2246, 2637, 3020, 3513.5, 3882, 4244, 4598]
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
NUMLIST = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


def convert_annotation(xml_path: str):
    """
    从xml文件中提取bounding box的信息
    :param img_path:path to input img
    :param xml_path:path to input xml
    :return:[(bndbox1),(bndbox2),...]
    """
    boxes = []
    with open(xml_path, 'r') as f:
        root = ET.fromstring(f.read())
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        box = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
               float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        boxes.append(box)
    sorted_list = sorted(boxes, key=lambda x: x[0])
    return sorted_list


def remove_background(image_path):
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)
    _, thresh = cv2.threshold(g, 170, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(img, mask)
    return result


def custom_mask(hsv_image):
    # 分别计算三个通道的和
    channel_sum = np.sum(hsv_image, axis=2)
    # 条件1：三通道加起来的值小于750，大于100
    condition1 = (channel_sum > 100) & (channel_sum < 750)
    # 计算蓝绿通道的和
    blue_green_sum = hsv_image[:, :, 0] + hsv_image[:, :, 1]
    # 条件2：蓝绿通道加起来小于520，大于80
    condition2 = (blue_green_sum > 100) & (blue_green_sum < 520)
    # 结合条件1和条件2
    combined_mask = condition1 & condition2
    return combined_mask.astype(np.uint8) * 255


def remove_with_color(image_path: str):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    upper_green = np.array([255, 255, 255])
    lower_green = np.array([10, 50, 10])
    mask1 = cv2.inRange(hsv_image, lower_green, upper_green)
    mask2 = custom_mask(hsv_image)
    mask = cv2.bitwise_and(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)

    min_area_threshold = 150  # 设置面积阈值
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area_threshold:
            # 在结果图像上填充轮廓
            epsilon = 0.0012 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(result, [approx], -1, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)
    return result


def plantExtract(img_path: str, result_path: str, xml_path: str):
    """
    从图片中提取植物照片
    :param img_pth:图片路径
    :param result_path:结果路径
    :return:
    """
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    boxes = convert_annotation(xml_path=xml_path)
    img = Image.open(img_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    basename = os.path.basename(img_path)

    day = basename.split('-')[-1].split('.')[0]
    font = ImageFont.truetype('simsunb.ttf', 30)
    text_position = (10, 10)
    text_color = (255, 0, 0)
    img_basename = "-".join(basename.split('.')[0].split('-')[0:-1])
    for box in boxes:
        x_index = min(range(len(X1)), key=lambda i: abs(X1[i] - box[0]))
        y_index = min(range(len(Y1)), key=lambda i: abs(Y1[i] - box[1]))
        region = img.crop(box)
        draw = ImageDraw.Draw(region)
        draw.text(text_position, day, fill=text_color, font=font)
        try:
            region.save(os.path.join(result_path, f"{img_basename}{ALPHABET[y_index]}{NUMLIST[x_index]}.jpg"))
        except:
            print(region)


def extract_and_sort(s):
    if s != 'concat':
        return int(s.split('-')[-1][:-1])
    else:
        return -1


def launch(xml_path: str, imgs_path: str, results_path: str):
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    for item in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, item)
        img_name = ".".join(item.split('.')[:-1])
        project_result_path = os.path.join(results_path, img_name)
        if not os.path.isdir(project_result_path):
            os.mkdir(project_result_path)
        cropped_results_path = os.path.join(project_result_path, 'cropped')
        plantExtract(img_path=img_path, result_path=cropped_results_path, xml_path=xml_path)

        # remove background
        # rm_bg_results_path = os.path.join(project_result_path,'rm_bg')
        # if not os.path.isdir(rm_bg_results_path):
        #     os.mkdir(rm_bg_results_path)
        # for img in os.listdir(cropped_results_path):
        #     img_path = os.path.join(cropped_results_path,img)
        #     result_image = remove_with_color(image_path=img_path)
        #     result_image_path = os.path.join(rm_bg_results_path,img)
        #     cv2.imwrite(result_image_path,result_image)

    # concatenate plant
    concate_results_path = os.path.join(results_path, 'concat')
    if not os.path.isdir(concate_results_path):
        os.mkdir(concate_results_path)
    img_dict = {}

    sorted_list = sorted(os.listdir(results_path), key=extract_and_sort)
    for item in sorted_list:
        if os.path.join(results_path, item) == concate_results_path:
            continue

        # contcate rm_bg
        # for img in os.listdir(os.path.join(results_path,item,'rm_bg')):
        #     img_name = ".".join(os.path.basename(img).split('.')[:-1])
        #     if img_name not in img_dict:
        #         img_dict[img_name] = []
        #     img_dict[img_name].append(os.path.join(results_path,item,'rm_bg',img))

        # concate cropped
        for img in os.listdir(os.path.join(results_path, item, 'cropped')):
            img_name = ".".join(os.path.basename(img).split('.')[:-1])
            if img_name not in img_dict:
                img_dict[img_name] = []
            img_dict[img_name].append(os.path.join(results_path, item, 'cropped', img))

    for img_key in img_dict:
        images = [cv2.imread(img_path) for img_path in img_dict[img_key]]
        all_successful = all(img.size != 0 for img in images)
        if not all_successful:
            print("有图像读取失败！")
        else:
            # 水平拼接所有图像
            concatenated_img = cv2.hconcat(images)
            cv2.imwrite(os.path.join(concate_results_path, img_key + '.jpg'), concatenated_img)


def _zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))


def ziplaunch(xml_path: str, zip_path: str, results_path: str):
    """

    """
    parent_path = os.path.dirname(zip_path)
    target_file = zipfile.ZipFile(zip_path)
    target_file.extractall(parent_path)
    target_file.close()
    os.remove(zip_path)
    launch(xml_path=xml_path, imgs_path=parent_path, results_path=results_path)
    result_files = os.path.join(results_path, 'concat')
    zip_name = 'result.zip'
    zip_path = os.path.join(results_path, zip_name)
    _zip_folder(result_files, zip_path)


def launchwithxml(zip_path: str, results_path: str):
    """

    """
    parent_path = os.path.dirname(zip_path)
    target_file = zipfile.ZipFile(zip_path)
    target_file.extractall(parent_path)
    target_file.close()
    os.remove(zip_path)
    launch(xml_path=xml_path, imgs_path=parent_path, results_path=results_path)
    result_files = os.path.join(results_path, 'concat')
    zip_name = 'result.zip'
    zip_path = os.path.join(results_path, zip_name)
    _zip_folder(result_files, zip_path)

    # separate plant
    ziplaunch(xml_path, imgs_path, results_path)


if __name__ == "__main__":
    xml_path = '/home/yhh/DigitalLab/BackEnd/ImageProcess/Tools/BG202388-42D.xml'
    imgs_path = '/home/yhh/DigitalLab/BackEnd/ImageProcess/Tools/BG202388-42D.zip'
    results_path = '/home/yhh/DigitalLab/BackEnd/ImageProcess/Tools'

    # separate plant
    ziplaunch(xml_path, imgs_path, results_path)