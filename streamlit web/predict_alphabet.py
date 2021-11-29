import cv2
import pytesseract
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



################
## 이미지 업로드 ##
################

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


#######################
## opencv로 이미지 분할 ##
#######################

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
            roi = rgb[y:y+h, x:x+w]     
            large2 = roi.copy()           

            # 이미지 출력
            number = number + 1
            st.sidebar.write('이미지 번호 : ', number)  
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(str(number) + ".jpg", large2)
            st.sidebar.image(str(number) + '.jpg')


    # 원하는 이미지 선택
    st.subheader("2-1. 원하는 이미지 선택")
    option = st.radio('원하는 이미지의 번호를 선택해 주세요.', range(1, number+1))
    if option:
        st.write('You selected:', option)
        st.image(str(option) + '.jpg')
    input_image = str(option) + '.jpg'


####################
## 이미지 직접 자르기 ##
####################
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.header("")
    st.subheader("2-2. 원하는 이미지 직접 자르기")
   
    if image_file:
        with st.form("my_form"):
            submitted = st.form_submit_button("click")
            if submitted:         
                img = Image.open(image_file)

                # Get a cropped image from the frontend
                cropped_img = st_cropper(img, aspect_ratio=None)
                
                # Manipulate cropped image at will
                st.write("Select Image")
                _ = cropped_img.thumbnail((300,300))
                st.image(cropped_img)
                cropped_img.save('crop.jpg')
                input_image = 'crop.jpg'
                          
            
#######################
## 알파벳으로 이미지 분할 ##
#######################
    st.header("")
    st.header("3. 알파벳으로 이미지 분할")

    # import image
    image = cv2.imread(input_image)

    # grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.waitKey(0)

    # binary
    ret,thresh = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.waitKey(0)

    # dilation
    kernel = np.ones((1,1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    edged = cv2.Canny(img_dilation, 30, 150)
    cv2.waitKey(0)

    # find contours
    ctrs, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
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
            cv2.rectangle(img_dilation,(x,y),( x + w, y + h ),(255,255,255), 0)
            cv2.rectangle(image,(x,y),( x + w, y + h ),(255,255,255), 0)
            cv2.imwrite('alphabet{}.jpg'.format(number), roi)
            st.image(roi_1)
            
    cv2.waitKey(0)
    

#############
## 모델 예측 ##
#############
    st.header("")
    st.header("4. 모델 예측")

    input_images = []
    for i in range(1, number+1):
        input_image = 'alphabet' + str(i) + '.jpg'
        # 흰색 배경 만들기
        x, y = [128, 128]
        theImage = Image.new('RGB', (x, y), (255, 255, 255))
        theImage.save('white.jpg')

        # 폰트와 이미지 합성
        font = cv2.imread(input_image)
        white = cv2.imread('white.jpg')
        h, w, c = font.shape
        white[40:40+h, 40:40+w] = font
        cv2.imwrite('font.jpg', white)
        
        img = Image.open('font.jpg')
        img_array = np.array(img)
        input_images.append(img_array)

    xs = np.array(input_images)
    xs_norm = xs / 255.0

    model_1 = tf.keras.models.load_model('font_test_0.01_4.h5')

    if image_file is not None:
        pred_y = model_1.predict(xs_norm)
        st.write(pred_y)

        pred_list = []
        for i in range(1, number+1): 
            # # 5개
            # target_names = ['Arbutus Slab-Regular', 'B612Mono-Bold', 'Dosis-Bold', 'Rakesly-Lightitalic', 'Vollkorn-BoldItalic']
            # # 10개(미팅전)
            # target_names = ['Arbutus Slab-Regular', 'Archivo_SemiExpanded-ExtraBoldItalic', 'Bitter-LightItalic',
            #                 'Fat Flamingo5-Overlay', 'Mukta Mahee-SemiBold', 'Old Standard TT-Regular',
            #                 'Rakesly-Lightitalic', 'Truculenta_Expanded-SemiBold', 'Uchen-Regular', 'Vollkorn-BoldItalic']
            # # 10개(미팅후)
            target_names = ['(Old-Modern) SenzaBella-Bold', 'Anonymous Pro-Regular', 'Arbutus Slab-Regular', 'B612Mono-Bold', 
                'Butter-Unsalted', 'Dosis-Bold', 'Libre Baskerville-Italic', 'Linden Hill-Regular', 
                'Playfair Display-SemiBoldItalic', 'Rakesly-Regaulr']
            st.write(i, '번째 알파벳의 폰트 : ', target_names[np.argmax(pred_y[i-1])])
            pred_list.append(target_names[np.argmax(pred_y[i-1])])
        
        counts = collections.Counter(pred_list)
        font_name = counts.most_common(1)[0][0]

        # 1. 알파벳별 폰트의 최빈수로 폰트 종류 알려주기
        st.header("")
        st.subheader("4-1. 빈도 수 가장 높은 폰트")
        st.write('▶ 위 단어의 폰트는 {} 입니다.'.format(counts.most_common(1)[0][0]))

        # 2. 폰트별로 확률을 더해서 최대값을 가진 폰트 알려주기
        st.header("")
        st.subheader("4-2. 폰트 별 확률 합")
        S = []
        for i in range(10):
            for j in range(number):
                sum_pred = 0
                sum_pred += pred_y[j][i]
            S.append(sum_pred)
            st.write(i+1, '번째 폰트의 확률 합 : ', np.round(S[i], 4))
        
        max = np.round(max(S), 4)
        n = np.argmax(S)
        st.write('따라서 ', n+1, '번째 폰트의 확률 합이 ', max, '(으)로 가장 큽니다.')
        st.write('▶ 위 단어의 폰트는 {} 입니다.'.format(target_names[n]))

##################
## 폰트 다운로드 ##
##################

    st.header("")
    st.subheader("5. 폰트 다운로드")

    if font_name :      
        font_file = font_name + '.zip'
        zip_path = 'font_download'
        font_path = os.path.join(zip_path, font_file)
        
        with open(font_path, 'rb') as f:
            st.download_button('Download Font!', f, file_name= font_file) 