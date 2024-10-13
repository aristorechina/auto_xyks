import os
from time import sleep
from time import time
from PIL import Image
import cv2
import numpy as np
import os
import sys

# 获取当前文件所在目录
path = os.path.dirname(os.path.realpath(sys.argv[0]))
new_path = "/".join(path.split("\\"))

# 从设备截取屏幕并保存到指定路径。
def take_screenshot(path):
    os.system(f'adb shell screencap -p > {path}')
 
    with open(path, 'rb') as f:
        return f.read().replace(b'\r\n', b'\n')

# 打开图片，裁剪并返回裁剪后的图片。
def process_image(image_path, crop_area):
    with Image.open(image_path) as img:
        return img.crop(crop_area)

######################################################################
###  视觉部分  ########################################################
######################################################################
# 加载图片
def load_images_from_folder(folder):
    images = {}
    filenames = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png']
    
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        images[filename] = img
    
    return images

# 使用 SIFT 特征匹配计算两张图片的相似度
def compare_images_with_sift(image1, image2):
    gray_img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=1000)  # 限制特征点数量
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]

    similarity = len(good_matches) / max(len(kp1), len(kp2))
    return similarity

# 获取数字
def get_number(file):
    images = load_images_from_folder(new_path)

    try:
        screenshot = Image.open(file)
    except Exception as e:
        print(f"无法打开文件 {file}: {e}")
        exit()

    max_similarity = -1
    most_similar_image = ''

    for filename, img in images.items():
        similarity = compare_images_with_sift(screenshot, img)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image = os.path.splitext(filename)[0]

    return most_similar_image

######################################################################
######################################################################
######################################################################

def greater_than():
    os.system("adb shell input swipe 450 1800 850 1900 0")
    os.system("adb shell input swipe 850 1900 450 2000 0")
def less_than():
    os.system("adb shell input swipe 850 1800 450 1900 0")
    os.system("adb shell input swipe 450 1900 850 2000 0")

# 比较两个数字并相应地执行滑动操作。
def compare_numbers(x, y):
    try:
        x_int, y_int = int(x), int(y)
        if x_int > y_int:
            print(f"{x} > {y}")
            greater_than()


        else:
            print(f"{x} < {y}")
            less_than()

    except ValueError:
        print("数字格式无效。")

# 获取图片中心位置
def get_image_position(screenshot_name,template_name):
    screenshot_image = cv2.imread(f'{new_path}/{screenshot_name}')
    template_image = cv2.imread(f'{new_path}/{template_name}')

    template_height, template_width = template_image.shape[:2]

    match_result = cv2.matchTemplate(screenshot_image, template_image, cv2.TM_CCOEFF_NORMED)

    _, max_value, _, max_location = cv2.minMaxLoc(match_result)

    if max_value > 0.8:
        top_left_x, top_left_y = max_location
        center_x = top_left_x + template_width // 2
        center_y = top_left_y + template_height // 2
        return center_x, center_y
    else:
        return None

# 模拟点击(OpenCV方案)
def routine(img_model_path,image_name="screenshot_.png"):
    screenshot_path = f'{new_path}/{image_name}'
    # 截取屏幕并保存
    screenshot = take_screenshot(screenshot_path)
    with open(screenshot_path, 'wb') as f:
        f.write(screenshot)

    avg = get_image_position(image_name,img_model_path)
    os.system(f'adb shell input tap {avg[0]} {avg[1]}')

def main():
    screenshot_path = f'{new_path}/screenshot.png'

    # 截取屏幕并保存
    screenshot = take_screenshot(screenshot_path)
    with open(screenshot_path, 'wb') as f:
        f.write(screenshot)

    # 定义裁剪区域（左，上，右，下）分别是两个数字在图片中的区域坐标
    clipping_region = lambda x,y : (x,y,x+120,y+150)

    crop_areas = [
        clipping_region(440, 680),
        clipping_region(880,680)
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

if __name__ == '__main__':
    os.system(f'adb connect 127.0.0.1:7555')
    while True:

        # 检测是否开始
        while True:
            try:
                routine("Ready.png")
                break
            except Exception as e:
                print(f"未开始")

        # 做题
        for i in range(25):
            main()
            sleep(0.4) # 等待0.4 秒

        # 结束PK后再次开始
        for i in range(2):
            try:
                routine("finish.png")
                break
            except Exception as e:
                print(f"未找到 开心收下 ，3秒后重试...")
                sleep(3)  # 等待3秒后再次尝试
        sleep(0.5)


        for i in range(2):
            try:
                routine("continue.png")
                break
            except Exception as e:
                print(f"未找到 继续 ，3秒后重试...")
                sleep(3)  # 等待3秒后再次尝试
        sleep(0.5)

        while True:
            try:
                routine("continuePK.png")
                break
            except Exception as e:
                print(f"未找到 继续PK ，3秒后重试...")
                sleep(3)  # 等待3秒后再次尝试



3 < 3
0 < 0
0 < 0
4 > 2
5 > 0
1 < 2
1 > 0
0 < 1
2 < 3
5 > 2
0 < 2
2 < 5
4 < 5
5 > 3
3 < 5
3 > 2
5 > 4
4 > 0
3 > 0
4 > 1
2 > 0
5 > 1
1 < 4
0 < 0