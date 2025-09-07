import cv2
import mediapipe as mp
import numpy as np
import math
from pynput.keyboard import Controller


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils


keyboard = Controller()


cap = cv2.VideoCapture(0)
cap.set(2, 150)


class Button():
    def __init__(self, pos, text, size=[70, 70]):
        self.pos = pos
        self.size = size
        self.text = text


keys_upper = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "CL"],
              ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "SP"],
              ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "APR"]]
keys_lower = [[k.lower() for k in row] for row in keys_upper]


def create_buttons(key_set):
    return [Button([80 * j + 10, 80 * i + 10], key) for i, row in enumerate(key_set) for j, key in enumerate(row)]

buttons_upper = create_buttons(keys_upper)
buttons_lower = create_buttons(keys_lower)


def draw_all_buttons(img, buttons):
    for button in buttons:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (96, 96, 96), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img


def highlight_button(frame, button, color_bg, color_text):
    x, y = button.pos
    w, h = button.size
    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), color_bg, cv2.FILLED)
    cv2.putText(frame, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, color_text, 4)


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


x_vals = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y_vals = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x_vals, y_vals, 2)


mode = 0
delay = 0
typed_text = ""

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (1000, 580))
    frame = cv2.flip(frame, 1)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    current_buttons = buttons_upper if mode == 0 else buttons_lower
    layout_indicator = "up" if mode == 0 else "down"
    frame = draw_all_buttons(frame, current_buttons)

    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])

    if landmarks:
        try:
            x5, y5 = landmarks[5][1], landmarks[5][2]
            x17, y17 = landmarks[17][1], landmarks[17][2]
            hand_distance = calculate_distance(x5, y5, x17, y17)
            A, B, C = coff
            distance_cm = A * hand_distance ** 2 + B * hand_distance + C

            if 20 < distance_cm < 50:
                x8, y8 = landmarks[8][1], landmarks[8][2]
                x6, y6 = landmarks[6][1], landmarks[6][2]
                x12, y12 = landmarks[12][1], landmarks[12][2]

                cv2.circle(frame, (x8, y8), 20, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x12, y12), 20, (255, 0, 255), cv2.FILLED)

                if y6 > y8:
                    for button in current_buttons:
                        xb, yb = button.pos
                        wb, hb = button.size

                        if xb < x8 < xb + wb and yb < y8 < yb + hb:
                            highlight_button(frame, button, (160, 160, 160), (255, 255, 255))
                            click_distance = calculate_distance(x8, y8, x12, y12)

                            if click_distance < 50 and delay == 0:
                                key = button.text
                                highlight_button(frame, button, (255, 255, 255), (0, 0, 0))

                                if key == "SP":
                                    typed_text += " "
                                    keyboard.press(" ")
                                elif key == "CL":
                                    typed_text = typed_text[:-1]
                                    keyboard.press('\b')
                                elif key == "APR":
                                    mode = 1 if layout_indicator == "up" else 0
                                else:
                                    typed_text += key
                                    keyboard.press(key)

                                delay = 1

        except:
            pass

    if delay != 0:
        delay += 1
        if delay > 10:
            delay = 0


    cv2.rectangle(frame, (20, 250), (850, 400), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, typed_text, (30, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)


    cv2.imshow('Virtual Keyboard', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
