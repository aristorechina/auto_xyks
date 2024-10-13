import os
from time import sleep
from time import time
from PIL import Image
import cv2
import numpy as np
import pyautogui
import os
import sys

#获取当前文件所在目录
path = os.path.dirname(os.path.realpath(sys.argv[0]))
new_path = "/".join(path.split("\\"))
 
def take_screenshot(path):
    """从设备截取屏幕并保存到指定路径。"""
    os.system(f'adb shell screencap -p > {path}')
 
    # 读取截取的屏幕截图并替换行结束符
    with open(path, 'rb') as f:
        return f.read().replace(b'\r\n', b'\n')
 
def process_image(image_path, crop_area):
    """打开图片，裁剪并返回裁剪后的图片。"""
    with Image.open(image_path) as img:
        return img.crop(crop_area)

def load_images_from_folder(folder):
    images = {}
    filenames = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png']
    
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        images[filename] = img
    
    return images

def compare_images_with_sift(image1, image2):
    """使用 SIFT 特征匹配计算两张图片的相似度"""
    # 转换为灰度图
    gray_img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2GRAY)

    # 初始化 SIFT 检测器
    sift = cv2.SIFT_create(nfeatures=1000)  # 限制特征点数量
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # 匹配特征点
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 只保留前一定数量的最佳匹配点
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]

    # 计算相似度
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    return similarity

def get_number(file):
    # 加载所有图片
    images = load_images_from_folder(new_path)

    # 加载屏幕截图
    try:
        screenshot = Image.open(file)
    except Exception as e:
        print(f"无法打开文件 {file}: {e}")
        exit()

    # 比较并找到最相似的图片
    max_similarity = -1
    most_similar_image = ''

    # 遍历所有图片
    for filename, img in images.items():
        similarity = compare_images_with_sift(screenshot, img)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image = os.path.splitext(filename)[0]

    return most_similar_image

def compare_numbers(x, y):
    """比较两个数字并相应地执行滑动操作。"""
    try:
        x_int, y_int = int(x), int(y)
        if x_int > y_int:
            print(f"{x} > {y}")
            start_time = time()
            greater_than()
            end_time = time()
            elapsed_time = end_time - start_time
            print(f"运行时间: {elapsed_time:.6f} 秒")

        else:
            print(f"{x} < {y}")
            less_than()

    except ValueError:
        print("数字格式无效。")
 
def main():
    screenshot_path = 'screenshot.png'
 
    # 截取屏幕并保存
    screenshot = take_screenshot(screenshot_path)
    with open(screenshot_path, 'wb') as f:
        f.write(screenshot)
 
    # 定义裁剪区域（左，上，右，下）分别是两个数字在图片中的区域坐标
    position = lambda x,y : (x,y,x+120,y+150)

    crop_areas = [
        position(440, 680),
        position(880,680)
    ]
 
    cropped_images = []
    for i, crop_area in enumerate(crop_areas, start=1):
        cropped_image = process_image(screenshot_path, crop_area)
        cropped_image_path = f"{new_path}/screenshot{i}.png"
        cropped_image.save(cropped_image_path)
        cropped_images.append(cropped_image_path)

    text0 = get_number(cropped_images[0])
    text1 = get_number(cropped_images[1])
 
    # 比较提取的数字
    compare_numbers(text0, text1)


def greater_than():
    os.system("adb shell input swipe 450 1800 850 1900 0")
    os.system("adb shell input swipe 850 1900 450 2000 0")
def less_than():
    os.system("adb shell input swipe 850 1800 450 1900 0")
    os.system("adb shell input swipe 450 1900 850 2000 0")

def get_xy(img_model_path):
    pyautogui.screenshot().save(f"{new_path}/PC_screenshot.png")
    img = cv2.imread(f"{new_path}/PC_screenshot.png")

    while True:
        location=pyautogui.locateCenterOnScreen(img_model_path,confidence=0.9)
        if location is not None:
            break
        print("未找到匹配图片,0.1秒后重试")
        sleep(0.1)

    img_terminal = cv2.imread(img_model_path)

    height, width, channel = img_terminal.shape
    result = cv2.matchTemplate(img, img_terminal, cv2.TM_SQDIFF_NORMED)
    upper_left = cv2.minMaxLoc(result)[2]
    lower_right = (upper_left[0] + width, upper_left[1] + height)
    avg = (int((upper_left[0] + lower_right[0]) / 2), int((upper_left[1] + lower_right[1]) / 2))
    return avg

def routine(img_model_path, name):
    avg = get_xy(img_model_path)
    print(f"{name}")
    pyautogui.click(avg[0], avg[1], button='left')

if __name__ == '__main__':
    os.system(f'adb connect 127.0.0.1:7555')
    while True:

        while True:
            try:
                get_xy(f"{new_path}/GO.png")
                break
            except Exception as e:
                print(f"未开始")

        start_time = time()
        while time() - start_time < 30:
            main()
            sleep(0.4) # 等待0.4 秒

        for i in range(3):
            try:
                routine(f"{new_path}/finish.png", '开心收下')
                break
            except Exception as e:
                print(f"未找到 开心收下 ，3秒后重试...")
                sleep(3)  # 等待3秒后再次尝试
        sleep(0.5)


        for i in range(3):
            try:
                routine(f"{new_path}/continue.png", '继续')
                break
            except Exception as e:
                print(f"未找到 继续 ，3秒后重试...")
                sleep(3)  # 等待3秒后再次尝试
        sleep(0.5)

        while True:
            try:
                routine(f"{new_path}/continuePK.png", '继续PK')
                break
            except Exception as e:
                print(f"未找到 继续PK ，3秒后重试...")
                sleep(3)  # 等待3秒后再次尝试
        sleep(0.5)