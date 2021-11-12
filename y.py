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
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            roi = rgb[y:y+h, x:x+w]     # roi 지정
            large2 = roi.copy()           # roi 배열 복제 ---①

            # 새로운 좌표에 roi 추가, 태양 2개 만들기
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2) # 2개의 태양 영역에 사각형 표시

            # 이미지 출력
            number = number + 1
            st.sidebar.write('이미지 번호 : ', number)
            # st.image(large2)     # roi 만 따로 출력
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(str(number) + ".jpg", large2)
            st.sidebar.image(str(number) + '.jpg')
    
    
    # number_list = []
    # for i in range(1, number+1):
    #     a = 'col' + str(i)
    #     number_list.append(a)
    # columns_list=  ', '.join(number_list)
    
    # columns_list = st.columns(number)
    
    # for i in range(len(number_list)):
    #     with number_list[i]:
    #         st.header('이미지 번호 : ', i+1)
    #         st.image(str(i+1) + '.jpg')


    # 원하는 이미지 선택
    st.subheader("2-1. 원하는 이미지 선택")
    option = st.radio('원하는 이미지의 번호를 선택해 주세요.', range(1, number+1))
    if option:
        st.write('You selected:', option)
        st.image(str(option) + '.jpg')
    input_image = str(option) + '.jpg'



###########################
## 이미지 직접 자르기 ##
###########################
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.header("")
    st.subheader("2-2. 원하는 이미지 직접 자르기")
    
    with st.expander("click button"):
        if image_file:           
            img = Image.open(image_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, aspect_ratio=None)
            
            # Manipulate cropped image at will
            st.write("Select Image")
            _ = cropped_img.thumbnail((300,300))
            st.image(cropped_img)


###########################
## 알파벳으로 이미지 분할 ##
###########################
    st.header("")
    st.header("3. 알파벳으로 이미지 분할")

    #import image
    image = cv2.imread(input_image)

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # st.image(gray)
    cv2.waitKey(0)

    #binary
    ret,thresh = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # st.image(thresh)
    cv2.waitKey(0)

    #dilation
    kernel = np.ones((1,1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # cv2_imshow(img_dilation)
    # cv2.waitKey(0)

    edged = cv2.Canny(img_dilation, 30, 150)
    # st.image(edged)
    cv2.waitKey(0)

    #find contours
    ctrs, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    number = 0 
    values = st.slider('Select a range of values', min_value=0, max_value=50, value=15, step=1 )
    st.write('Values:', values)
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        if h > values:
            # Getting ROI
            roi = img_dilation[y:y+h, x:x+w]
            roi_1 = image[y:y+h, x:x+w]
            roi_shape = roi.shape[0]

            # show ROI
            number = number +1
            cv2.rectangle(img_dilation,(x,y),( x + w, y + h ),(0,255,0),2)
            cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)
            cv2.imwrite('alphabet{}.jpg'.format(number), roi)
            st.image(roi_1)
    
    # st.image(img_dilation)
    # st.image(image)
    cv2.waitKey(0)

    # 원하는 이미지 선택
    # st.subheader("3-1. 원하는 이미지 선택")
    # option = st.radio('원하는 이미지의 번호를 선택해 주세요.', range(1, number+1))
    # if option:
    #     st.write('You selected:', option)
    #     st.image('alphabet'+ str(option) + '.jpg')
    

###############
## 모델 예측 ##
###############
    st.header("")
    st.header("4. 모델 예측")

    input_images = []
    for i in range(1, number+1):
        input_image = 'alphabet' + str(i) + '.jpg'
        img = Image.open(input_image).convert('RGB')
        resized = img.resize([50,50])
        img_array = np.array(resized)

        input_images.append(img_array)

    input_images = np.array(input_images)
    print(input_images.shape)

    model_1 = tf.keras.models.load_model('font_init_1.h5')

    if image_file is not None:
        pred_y = model_1.predict(input_images)
        st.write(pred_y)