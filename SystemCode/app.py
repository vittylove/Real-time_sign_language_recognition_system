import cv2
from test import detect_image
from PIL import Image, ImageFont, ImageDraw
import wordninja
import mediapipe as mp
import numpy as np
import time
from spellchecker import SpellChecker

class Num:
    nNum = 0
    def inc(self):
        self.nNum += 1
    def zero(self):
        self.nNum = 0
class Text:
    text = []
    pre = ""
    def text_append(self, word):
        self.text.append(word)
    def pre_set(self, word):
        self.pre = word
class ImageSize:
    minX = 0
    maxX = 0
    minY = 0
    maxY = 0
    def set_min_x(self, num):
        self.minX = num if num >= 0 else 0

    def set_max_x(self, max, num):
        self.maxX = num if num <= max else max

    def set_min_y(self, num):
        self.minY = num if num >= 0 else 0

    def set_max_y(self, max, num):
        self.maxY = num if num <= max else max


img_hand = ImageSize()
spell = SpellChecker()
inst = Num()
text = Text()
def gen_frames():  # generate frame by frame from camera
    mp_hands = mp.solutions.hands
    hands_mode = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # use 0 for web camera
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if success:
            img = Image.fromarray(frame[..., ::-1])  # 完成np.array向PIL.Image格式的转换
            # 参数：1、是否检测静态图片，2、手的数量，3、检测阈值，4、跟踪阈值
            image_hight, image_width, _ = frame.shape
            # 将NGR转为RGB图像
            image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 应用mdiapipe
            results = hands_mode.process(image1)
            shape = [(0, 0), (0, 0)]
            new_frame_time = time.time()

            if results.multi_handedness:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 定义截图范围
                    img_hand.set_max_x(image_width,int((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x) * image_width + (image_hight/4)))
                    img_hand.set_max_y(image_hight,int((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y) * image_hight + (image_hight/4)))
                    img_hand.set_min_x(int((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x) * image_width - (image_hight/4)))
                    img_hand.set_min_y(int((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y) * image_hight - (image_hight/4)))
                addText(frame[img_hand.minY:img_hand.maxY, img_hand.minX:img_hand.maxX])
                shape = [(img_hand.minX, img_hand.maxY), (img_hand.maxX, img_hand.minY)]

            text2 = "".join(text.text)
            text_list = []
            text_word = wordninja.split(text2)
            for word in text_word:
                text_list.append(spell.correction(word))
            draw = ImageDraw.Draw(img)
            draw.rectangle(shape, fill=None, outline=(255, 0, 0))
            font = "./data/FiraMono-Mesdium.otf"
            pointsize = 30
            shadowcolor = "black"
            fill_color = "white"
            font = ImageFont.truetype(font, pointsize)
            text_border(draw, 10, image_hight - 30, font, shadowcolor, fill_color, " ".join(text_word))
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            fps = "FPS: " + fps
            text_border(draw, 7, 10, font, shadowcolor, fill_color, fps)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", frame)
        if cv2.waitKey(1):
            pass

def addText(image):
    dst = cv2.resize(image, (416, 416))
    return_label = detect_image(dst)
    if return_label != "":
        if return_label == text.pre:
            if inst.nNum >= 5:
                inst.zero()
                text.text_append(return_label)
            else:
                inst.inc()
        else:
            inst.zero()
            text.pre_set(return_label)


def text_border(draw, x, y, font, shadowcolor, fillcolor, text):
    # thin border
    draw.text((x - 1, y), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y), text, font=font, fill=shadowcolor)
    draw.text((x, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x, y + 1), text, font=font, fill=shadowcolor)

    # thicker border
    draw.text((x - 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x - 1, y + 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y + 1), text, font=font, fill=shadowcolor)

    # now draw the text over it
    draw.text((x, y), text, font=font, fill=fillcolor)
gen_frames()