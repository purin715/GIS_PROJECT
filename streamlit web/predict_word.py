import tensorflow as tf
import PIL
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pytesseract
from tensorflow import keras 
from keras.models import load_model

##################
## 이미지 업로드 ##
##################

st.title("하겐다즈 팀 : 폰트 분류")
st.header("1. 이미지 업로드")

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

image_file = st.file_uploader(label='이미지를 업로드 해주세요.', type=['png', 'jpg'])

if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    img = load_image(image_file)
    st.image(img)
    with open(os.path.join(image_file.name),"wb") as f: 
        f.write(image_file.getbuffer())         
    st.success("Upload Successful!") 


#########################
## opencv로 이미지 분할 ##
#########################

if image_file: 
    st.header("")
    st.header("2. 폰트 이미지 선택")
    large = cv2.imread(image_file.name)
    rgb = cv2.pyrDown(large)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    number = 0 
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (255,255,255), 0)
            roi = rgb[y:y+h, x:x+w]     # roi 지정
            large2 = roi.copy()           # roi 배열 복제 ---①

            # 새로운 좌표에 roi 추가, 태양 2개 만들기
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (255,255,255), 0) # 2개의 태양 영역에 사각형 표시

            # 이미지 출력
            number = number + 1
            st.sidebar.write('이미지 번호 : ', number)
            # st.image(large2)     # roi 만 따로 출력
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(str(number) + ".jpg", large2)
            st.sidebar.image(str(number) + '.jpg')

    # 원하는 이미지 선택
    option = st.radio('원하는 이미지의 번호를 선택해 주세요.', range(1, number+1))
    if option:
        st.write('You selected:', option)
        st.image(str(option) + '.jpg')
    input_image = str(option) + '.jpg'

###############
## 모델 예측 ##
###############
    st.header("")
    st.header("3. 모델 예측")

    input_images = []
    for i in range(1, number+1):
        input_image = 'alphabet' + str(i) + '.jpg'
        img = Image.open(input_image).convert('RGB')
        resized = img.resize([50,50])
        img_array = np.array(resized)

        input_images.append(img_array)

    xs = np.array(input_images)
    xs_norm = xs / 255.0

    model_1 = tf.keras.models.load_model('font_init_test_9.h5')

    if image_file is not None:
        pred_y = model_1.predict(xs_norm)
        st.write(pred_y)

        for i in range(1, number+1): 
            # target_names =  ['Balthazar-Regular', 'Cambo-Regular', 'Fauna One-Regular', 'Gilda Display-Regular', 'Habibi-Regular', 'Josefin Slab-Regular', 'Kotta One-Regular', 'Mate-Regular']
            # target_names =  ['Cambo-Regular', 'Fauna One-Regular']
            target_names =  ['Adlinnaka-BoldDemo', 'Adolphus']
            st.write('폰트 : ', target_names[np.argmax(pred_y[i-1])])