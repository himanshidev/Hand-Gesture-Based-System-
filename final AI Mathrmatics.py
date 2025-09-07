import cvzone   
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import streamlit as st

# Configure Streamlit layout
st.set_page_config(layout="wide")
col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox('Run', value=False)
    if run:
        FRAME_WINDOW = st.image([])
    else:
        FRAME_WINDOW = None

with col2:
    st.title("Answer")
    output_text_area = st.empty()


genai.configure(api_key="AIzaSyBzRp_E8B_bqcPQ0tRH_2qJF7QEA5xMxac")
model = genai.GenerativeModel('gemini-1.5-flash')


cap = cv2.VideoCapture(0)
cap.set(3, 980)
cap.set(4, 620)


detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas, img):
    fingers, lmList = info
    current_pos = None
    if fingers == [0,1,0,0,0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [0,0,0,0,1]:
        canvas = np.zeros_like(img)
    return current_pos, canvas

def sendToAI(model, canvas):
    pil_image = Image.fromarray(canvas)
    response = model.generate_content(["State the math problem and solve the math problem step wise", pil_image])
    return response.text

def format_text(text):
    words = text.split()
    lines = [' '.join(words[i:i+6]) for i in range(0, len(words), 6)]
    return '\n'.join(lines)


prev_pos = None
canvas = None
output_text = ""
request_sent = False

# Main loop
while run:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas, img)

        if fingers == [0,1,1,0,0] and not request_sent:
            output_text = sendToAI(model, canvas)
            request_sent = True

        elif fingers == [0,0,0,0,1]:  # Clear canvas gesture
            canvas = np.zeros_like(img)
            request_sent = False
            output_text = ""

    image_combine = cv2.addWeighted(img, 0.6, canvas, 0.4, 0)
    FRAME_WINDOW.image(image_combine, channels="BGR")

    if output_text:
        formatted_text = format_text(output_text)
        output_text_area.text(formatted_text)
