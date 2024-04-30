import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from PIL import Image
from keras.models import load_model

# TODO: streamlit 기본 폰트로 차트, EC2 한글 적용하기.

def data_processing():
    global red_wine
    red_wine = pd.read_csv(r'red_wine_quality\red_wine_quality_data.csv')
    red_wine.reset_index(drop = True, inplace=True)
    
    red_wine.rename(columns={'pH': 'ph'}, inplace=True)
    
    # 영어
    en_cols = red_wine.columns
    
    # 한국어
    kr_cols = ['고정산도', '휘발성산도', '구연산', '잔류설탕', '염화물', '유리이산화황', '총이산화황', '밀도', '수소이온농도', '황산염', '알코올', '품질']
    
    # 컬럼명 설정
    red_wine.columns = en_cols

def home():
    st.title('홈')
    
    img = Image.open(r'red_wine_quality\red_wine.jpg')
    st.image(img, use_column_width=True)
    
    st.header('대시보드')
    st.write("")
    st.write("")
    st.write("")
    
    st.subheader('품질 예측 인공지능')
    st.write('''이 대시보드는 레드 와인의 화학적 특성을 기반으로 와인의 품질을 예측하는 인공지능 모델을 시각화하고 이해하는 데 도움을 줍니다. 이 모델은 다양한 화학적 특성 데이터를 기반으로 특정 와인의 품질을 추정하며, 와인 제조업체나 소비자에게 가치 있는 정보를 제공합니다.''')
    
    st.divider()
    
    st.header('레드 와인이란?')
    st.write("")
    st.write("")
    st.write("")
    
    data = {
    "제조 과정": "레드 와인은 포도에서 얻은 주스를 발효시켜 만든 포도주 중의 하나로, 포도의 피부와 씨앗을 함께 발효시키는 과정에서 레드 와인 특유의 색상과 향미가 형성됩니다.",
    "색상 형성 요인": "레드 와인의 색상은 주로 포도의 피부에 함유된 색소인 안토시아닌에 의해 결정됩니다. 와인이 더 오래 숙성될수록 색상이 짙어지는 경향이 있습니다.",
    "화학적 성분": "레드 와인은 다양한 화학적 성분을 함유하고 있습니다. 예를 들어, 알코올, 산성도, 당도, 탄닌 등이 포함되어 있으며, 이러한 성분들이 와인의 맛과 향을 형성합니다.",
    "즐기는 방법": "레드 와인은 다양한 식사와 장소에서 즐길 수 있습니다. 많은 사람들이 와인을 다양한 음식과 조화롭게 즐기며, 와인 문화는 전 세계적으로 인기를 얻고 있습니다.",
    "특별한 순간": "와인은 즐거운 휴식을 위한 좋은 선택이며, 많은 사람들에게 특별한 순간을 만들어줍니다. 레드 와인은 그 중에서도 풍부한 향과 깊은 맛으로 인해 특별한 자리에서 자주 선택되곤 합니다."
    }

    for index, (title, content) in enumerate(data.items()):
        col1, col2 = st.columns([3.5, 10-3.5])
        with col1:
            st.subheader(title)
        with col2:
            st.write(content)
        if index != len(data) - 1:
            st.divider()

def data_analysis():
    st.title('데이터 분석')
    
    st.header('일반 조회')
    options = ['기본', '데이터 분석', '데이터 타입']
    selected_option = st.radio("조회할 유형을 선택하세요.", options)
    
    if selected_option == '기본':
        st.write(red_wine)
    elif selected_option == '데이터 분석':
        st.write(red_wine.describe())
    elif selected_option == '데이터 타입':
        temp = []
        for column in red_wine.columns:
            temp.append({'컬럼': column, '데이터 타입': red_wine[column].dtype})
        temp = pd.DataFrame(temp)
        st.write(temp)
    
    st.header('컬럼별 조회')
    selected_columns = st.multiselect("조회할 컬럼을 선택하세요.", red_wine.columns)
    if(len(selected_columns) > 0):
        st.write(red_wine[selected_columns])
        
        for column in selected_columns:
            if red_wine[column].dtype in ['int64', 'float64']:
                st.header(f"{column}의 차트")
                fig, ax = plt.subplots(figsize=(16, 9))
                
                plt.style.use('dark_background')
                fig.patch.set_alpha(0)
                ax.set_facecolor('#00000000')
                
                ax.plot(red_wine[column], color='#ce293d')
                ax.set_ylabel(column)
                ax.set_ylim(red_wine[column].min(), red_wine[column].max())
                st.pyplot(fig)
            elif red_wine[column].dtype in ['object']:
                st.header(f"{column}의 유니크")
                st.write(red_wine[column].unique())
            elif red_wine[column].dtype in ['datetime64[ns]']:
                st.header(f"{column}의 최소/최대 날짜")
                st.write(pd.DataFrame([{'최소 날짜':red_wine[column].min(), '최대 날짜':red_wine[column].max()}]))
    else:
        st.write('선택된 컬럼이 없습니다.')
        
    st.header('상관관계 분석')
    numerical_columns = red_wine.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_columns = st.multiselect("상관관계를 분석할 컬럼을 선택하세요.", numerical_columns)

    if len(selected_columns) > 0:
        correlation_matrix = red_wine[selected_columns].corr()
        
        base_color = mcolors.hex2color('#ce293d')
        
        r, g, b = base_color
        colors = [base_color]
        for i in range(1, 128):
            new_r = max(0, min(1, r - i / 256))
            new_g = max(0, min(1, g - i / 256))
            new_b = max(0, min(1, b - i / 256))
            colors.insert(0, (new_r, new_g, new_b))

        r, g, b = base_color
        for i in range(1, 128):
            new_r = max(0, min(1, r + i / 256))
            new_g = max(0, min(1, g + i / 256))
            new_b = max(0, min(1, b + i / 256))
            colors.append((new_r, new_g, new_b))

        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors)
        
        min_fontsize = 6
        max_fontsize = 12
        text_size = max(min_fontsize, min(max_fontsize, 10 / np.sqrt(len(selected_columns))))
        
        fig, ax = plt.subplots()
        plt.xticks(fontsize=text_size)
        plt.yticks(fontsize=text_size)
        
        sns.heatmap(correlation_matrix, annot=True, annot_kws={"fontsize": text_size}, cmap=cmap, fmt=".2f", ax=ax)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=text_size)
        
        plt.style.use('dark_background')
        fig.patch.set_alpha(0)
        
        st.pyplot(fig)
    else:
        st.write('선택된 컬럼이 없습니다.')

def machine_learning():
    st.title('머신러닝')
    
    st.header('설정')
    values = []
    for col in red_wine.columns[:11]:
        value = st.number_input(f'{col}', value=red_wine[col].iloc[0])
        values.append(value)
        
    inputs = np.array(values)
    
    if st.button('예측'):
        # TODO: inputs > h5 model > predict | 개발중이라서 에러생김
        model = load_model('red_wine_quality\red_wine_quality_model.h5')
        model.predict(inputs)

def index():
    data_processing()
    
    st.sidebar.title('레드 와인')
    menu = st.sidebar.selectbox("대시보드", ["홈", "데이터 분석", "머신러닝"])
    
    if menu == "홈":
        home()
    elif menu == "데이터 분석":
        data_analysis()
    elif menu == "머신러닝":
        machine_learning()
    
if __name__ == '__main__':
    index()