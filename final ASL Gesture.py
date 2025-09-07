#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


HAND_CONNECTIONS = [
    (2, 3), (3, 4),  # Thumb
    (5, 6), (6, 7), (7, 8),  # Index finger
    (9, 10), (10, 11), (11, 12),  # Middle finger
    (13, 14), (14, 15), (15, 16),  # Ring finger
    (17, 18), (18, 19), (19, 20),  # Little finger
    (0, 1), (1, 2), (2, 5), (5, 9),  # Palm
    (9, 13), (13, 17), (17, 0)
]

COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'highlight': (152, 251, 152),
    'right_hand': (0, 255, 0),  # Green for right hand
    'left_hand': (255, 0, 0)    # Red for left hand
}

HISTORY_LENGTH = 16
POINT_RADIUS = {
    'default': 5,
    'fingertips': 8
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.8)  # Increased from 0.7
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    return parser.parse_args()

def _draw_with_outline(image, text, position, font_scale, thickness):
    cv.putText(image, text, position, cv.FONT_HERSHEY_SIMPLEX, 
              font_scale, COLORS['black'], thickness + 2, cv.LINE_AA)
    cv.putText(image, text, position, cv.FONT_HERSHEY_SIMPLEX, 
              font_scale, COLORS['white'], thickness, cv.LINE_AA)

def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,  # Detect both hands
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    point_history = deque(maxlen=HISTORY_LENGTH)
    finger_gesture_history = deque(maxlen=HISTORY_LENGTH)
    mode = 0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
            
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):

                hand_type = handedness.classification[0].label
                hand_score = handedness.classification[0].score


                if hand_score < 0.8:
                    continue

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                logging_csv(number, mode, pre_processed_landmark_list,
                          pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                point_history.append(landmark_list[8] if hand_sign_id == 2 else [0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (HISTORY_LENGTH * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()


                hand_color = COLORS['right_hand'] if hand_type == 'Right' else COLORS['left_hand']
                debug_image = draw_bounding_rect(debug_image, brect, hand_color)
                debug_image = draw_landmarks(debug_image, landmark_list, hand_color)
                debug_image = draw_info_text(
                    debug_image, brect, handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    elif key == 110:
        mode = 0
    elif key == 107:
        mode = 1
    elif key == 104:
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([(min(int(lm.x * image_width), image_width - 1),
                               min(int(lm.y * image_height), image_height - 1))
                              for lm in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [
        [min(int(lm.x * image_width), image_width - 1),
         min(int(lm.y * image_height), image_height - 1)]
        for lm in landmarks.landmark
    ]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    
    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list]

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = temp_point_history[0]

    return list(itertools.chain.from_iterable(
        [(x - base_x) / image_width, (y - base_y) / image_height]
        for x, y in temp_point_history
    ))

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0 or not (0 <= number <= 9):
        return
        
    csv_path = ('model/keypoint_classifier/keypoint.csv' if mode == 1 
               else 'model/point_history_classifier/point_history.csv')
    
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *(landmark_list if mode == 1 else point_history_list)])

def draw_landmarks(image, landmark_point, hand_color=None):
    if not landmark_point:
        return image
        

    line_color = hand_color if hand_color else COLORS['white']
    

    for start, end in HAND_CONNECTIONS:
        cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]),
               COLORS['black'], 6)
        cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]),
               line_color, 2)
    

    for i, landmark in enumerate(landmark_point):
        radius = POINT_RADIUS['fingertips'] if i in [4, 8, 12, 16, 20] else POINT_RADIUS['default']
        cv.circle(image, (landmark[0], landmark[1]), radius, 
                 line_color, -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, 
                 COLORS['black'], 1)
    
    return image

def draw_bounding_rect(image, brect, color=None):
    color = color if color else COLORS['black']
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                 color, 2)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    hand_label = handedness.classification[0].label
    hand_score = handedness.classification[0].score
    

    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 COLORS['black'], -1)
    

    info_text = f"{hand_label} ({hand_score:.2f})"
    if hand_sign_text:
        info_text += f": {hand_sign_text}"
    
    _draw_with_outline(image, info_text, (brect[0] + 5, brect[1] - 4), 0.6, 1)

    if finger_gesture_text:
        _draw_with_outline(image, f"Finger Gesture: {finger_gesture_text}", 
                         (10, 60), 1.0, 2)

    return image

def draw_point_history(image, point_history):
    for index, (x, y) in enumerate(point_history):
        if x != 0 and y != 0:
            cv.circle(image, (x, y), 1 + int(index / 2),
                      COLORS['highlight'], 2)
    return image

def draw_info(image, fps, mode, number):
    _draw_with_outline(image, f"FPS: {fps}", (10, 30), 1.0, 2)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        _draw_with_outline(image, f"MODE: {mode_string[mode - 1]}", (10, 90), 0.6, 1)
        if 0 <= number <= 9:
            _draw_with_outline(image, f"NUM: {number}", (10, 110), 0.6, 1)
    return image

if __name__ == '__main__':
    main()