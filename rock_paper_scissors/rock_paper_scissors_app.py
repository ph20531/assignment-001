import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import time
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

isPlaying = False
kr_names = ["보자기", "주먹", "가위"]
ids = ["paper", "rock", "scissors"]
sprites = ["✋", "✊", "✌"]
sprites_len = len(sprites)
sprites_index = 0

st.title("가위 바위 보")
placeholder = st.empty()

def display():
    global isPlaying
    global sprites_index
    global placeholder
    
    while not isPlaying:
        time.sleep(0.3)
        placeholder.markdown(f"<div style='font-size:128pt;'>{sprites[sprites_index]}</div>", unsafe_allow_html=True)
        
        sprites_index += 1
        if sprites_index >= sprites_len:
            sprites_index = 0  
    else:
        placeholder.markdown(f"<div style='font-size:128pt;''>{sprites[sprites_index]}</div>", unsafe_allow_html=True)
            
def controller():
    uploaded_file = st.file_uploader("파일 업로드", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        shoot(uploaded_file)

def shoot(file):
    global isPlaying
    isPlaying = True
    
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(os.path.join(root_dir, "rock_paper_scissors_model.h5"), compile=False)

    # Load the labels
    class_names = open(os.path.join(root_dir, "rock_paper_scissors_labels.txt"), "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The "length" or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    # image = Image.open(os.path.join(root_dir, "res", file_name)).convert("RGB")
    image = Image.open(file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    mine = sprites[index]
    opponent = sprites[sprites_index]
    
    placeholder_display = st.empty()
    
    with placeholder_display.container():
        
        st.image(image)
        st.caption(f"해당 이미지는 {int(confidence_score*100)}%의 확률로 {kr_names[index]}입니다.")
    
        st.divider()

        st.header('게임 기록')
        
        st.subheader('당신의 선택')
        
        # st.write(mine)
        st.markdown(f"<div style='font-size:128pt;'>{mine}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader('상대방의 선택')
        # st.write(opponent)
        st.markdown(f"<div style='font-size:128pt;'>{opponent}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader('게임 결과')
        if mine == opponent:
            st.info("무승부입니다.")
        elif (mine == sprites[2] and opponent == sprites[0]) or \
            (mine == sprites[1] and opponent == sprites[2]) or \
            (mine == sprites[0] and opponent == sprites[1]):
            st.success("당신이 승리했습니다.")
        else:
            st.error("당신이 패배했습니다.")
            
        if st.button("다시하기", use_container_width=True):
            placeholder_display.empty()
            isPlaying = False
        
def index():
    st.sidebar.title('프로젝트')
    
    img = Image.open(os.path.join(root_dir, 'rock_paper_scissors.png'))
    st.sidebar.image(img, use_column_width=True)
    
    st.sidebar.header('가위 바위 보')
    st.write("")
    st.write("")
    st.write("")
    
    st.sidebar.subheader('소개')
    st.sidebar.write('해당 프로젝트는 이미지 분류 인공지능(Teachable Machine)을 활용한 "가위 바위 보" 게임입니다.')
    st.sidebar.divider()
    
    st.sidebar.subheader('의도')
    st.sidebar.write('원래는 웹캠을 활용하여 리얼 타임으로 게임이 진행되도록 개발할 예정이였으나 AWS EC2 프리티어 서버 성능의 문제로 인해 단일 이미지 파일을 업로드하는 형식으로 게임이 진행되도록 개발하게 되었습니다.')
    st.sidebar.divider()
    
    st.sidebar.subheader('목적')
    st.sidebar.write('유니티나 언리얼 게임 엔진에서 사용할 수 있는 AR Input System을 개발하는 것이 해당 프로젝트의 목적입니다.')
    st.sidebar.divider()
    
    st.sidebar.subheader('게임 방법')
    st.sidebar.write('가위, 바위, 보 중의 제스처 이미지를 업로드하면 게임이 시작됩니다. 상대방의 선택은 랜덤이며 게임 기록이 화면에 표시됩니다.')
    st.sidebar.divider()
    
    st.sidebar.subheader('데이터 셋')
    st.sidebar.markdown('''<a href="https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset" style="color:#ffffff;">001. 가위 바위 보 데이터 셋 (실제)</a><br>
                           <a href="https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset" style="color:#ffffff;">002. 가위 바위 보 데이터 셋 (가상 증강)</a>''', unsafe_allow_html=True)
    
    controller()
    display()

if __name__ == "__main__":
    index()