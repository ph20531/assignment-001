# 과제 001

## 프로젝트 001

### 아이디어
선형회귀 인공지능(Linear Regression)을 활용한 레드 와인 품질 예측

### 소개
해당 프로젝트는 레드 와인의 성분 값들을 입력받아서 품질을 예측하고<br/>
예측한 품질과 비슷한 제품들을 소개해주는 웹 대시보드 어플리케이션입니다.

### 핵심 기술
- TensorFlow Linear Regression

### 작업 과정
001. colab에서 인공지능 개발 후 scaler.pkl, model.h5파일을 저장하고 다운로드<br/>
002. 로컬에서 데이터 셋과 scaler.pkl, model.h5파일들을 임포트한 후 streamlit 웹 대쉬보드 개발<br/>
003. AWS EC2서버에 웹 대시보드 배포

### 데이터 셋
001. <a href=https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009>레드 와인 품질 데이터 셋</a><br/>
002. <a href=https://www.kaggle.com/datasets/budnyak/wine-rating-and-price>품질 관련 와인 제품 데이터 셋</a>

## 프로젝트 002

### 아이디어
이미지 분류 인공지능(Teachable Machine)을 활용한 가위 바위 보 게임

### 소개
해당 프로젝트는 이미지 분류 인공지능(Teachable Machine)을 활용한 "가위 바위 보" 게임입니다.<br/>
<br/>
원래는 웹캠을 활용하여 리얼 타임으로 게임이 진행되도록 개발할 예정이였으나<br/>
AWS EC2 프리티어 서버 성능의 문제로 인해 단일 이미지 파일을 업로드하는<br/>
형식으로 게임이 진행되도록 개발하게 되었습니다.<br/>
<br/>
유니티나 언리얼 게임 엔진에서 사용할 수 있는 AR Input System을<br/>
개발하는 것이 해당 프로젝트의 목적입니다.

### 핵심 기술
- Teachable Machine Image Classification

### 작업 과정
001. Teachable Machine으로 인공지능 개발 후 model.h5파일을 다운로드<br/>
002. 로컬에서 model.h5파일을 임포트한 후 streamlit 웹 대시보드 개발<br/>
003. AWS EC2서버에 웹 대시보드 배포

### 데이터 셋
001. <a href=https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset>가위 바위 보 데이터 셋 (실제)</a><br/>
002. <a href=https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset>가위 바위 보 데이터 셋 (가상 증강)</a>

## 가상 환경

### 패키지 리스트
001. OpenSSL
002. NumPy
003. SciPy
004. Matplotlib
005. IPython
006. Scikit-learn
007. Pandas
008. Pillow
009. Jupyter
010. Seaborn
011. Keras 2.12.0
012. TensorFlow 2.12.0

### 명령어
```bash
conda create -n venv python=3.10 openssl numpy scipy matplotlib ipython scikit-learn pandas pillow jupyter seaborn keras=2.12.0 tensorflow=2.12.0
```

### 빌드 타겟
001. keras=2.12.0<br/>
002. tensorflow=2.12.0<br/>

###### 참고로 keras와 tensorflow 패키지는 2.12.0버전으로 설치해주셔야 프로젝트가 문제없이 작동합니다.