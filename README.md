
# ✨ GRU 기반 수어 인식 프로젝트 (TensorFlow)

이 프로젝트는 TensorFlow 기반의 GRU 모델을 사용하여 MediaPipe에서 추출한 키포인트 시계열 데이터를 학습하고, 수어(수화)를 인식하는 딥러닝 파이프라인을 구현한 것입니다. 키포인트 정규화, 라벨링, 모델 학습 및 테스트 영상을 포함한 전체 흐름이 구성되어 있습니다.

---

## 프로젝트 구성

### 주요 파일

| 파일명 | 설명 |
|--------|------|
| `save_keypoint_norm.ipynb` | aihub 키포인트 데이터를 정규화 및 저장 |
| `save_data_label.ipynb` | 수어 키포인트 데이터에 라벨을 지정 |
| `save_test_video.ipynb` | 테스트용 영상 데이터 저장 처리 |
| `train_gru_tensorflow.ipynb` | TensorFlow 기반 GRU 모델 학습 수행 |
| `word_index_dict.ipynb` | 단어 인덱스 딕셔너리 생성 및 시각화 도구 |
| `application2.ipynb` | 전체 모델 통합 응용 및 평가 구현 |
| `gemini.py` | 모델 서빙 또는 보조 기능 스크립트 (추정) |

---

## 파이썬 버전 및 사용 라이브러리

python version : **python=3.9.21**

이 프로젝트는 다음과 같은 라이브러리를 필요로 합니다:


- **PyTorch**
- **opencv-python**
- **mediapipe** (키포인트 추출)
- **pillow**
- **tensorflow**
- **numpy**
- **openai** 
- **dotenv** (API 키와 같은 비밀값 불러오기)
- 주요 유틸리티: `sys`, `os` 및 표준 모듈

---

## 사용 방법

1. **필요 라이브러리 설치**
   ```bash
   pip install opencv-python mediapipe numpy torch pillow tensorflow openai python-dotenv
   ```

2. **환경 변수 설정**
   `.env` 파일 생성:
   ```env
   OPENAI_API_KEY="sk-XXXXXXXX"
   GEMINI_API_URL="xxxx"
   ```
3. **수어 인식 프로그램 실행 (캠 필요)**
   ```bash
   application.ipynb 실행
   ```
word2idx2.json 에 속한 단어에 한해서 인식 가능합니다.

---

## 모델 학습 과정




### 📂 데이터 준비
[AIhub 수어 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103)


- **시퀀스 처리 방식**:
  - 지정된 `max_length`를 초과하는 시퀀스는 잘리고, 짧은 시퀀스는 0으로 패딩됩니다.
  - 원본 시퀀스 길이는 `lengths` 텐서에 저장되어 효율적인 RNN 처리를 가능하게 합니다.

- **주요 기능**:
  - `.npz` 형식의 키포인트 또는 센서 데이터를 불러옵니다.
  - 예시 출력:
    > `print(f"⚠️ Padding {npz_path}: keypoints shape={keypoints.shape}")`

---

## 🧠 모델 구성

### 네트워크 구조

- **PyTorch 레이어**로 구성되며, LSTM/GRU/RNN 등 시퀀스 데이터에 최적화되어 있음
- **입력 차원**:
  - `x.shape → (batch, seq_length, input_features)`
  - `lengths.shape → (batch)` ← 시퀀스 길이 정보

### 주요 컴포넌트

- `nn.utils.rnn.pack_padded_sequence`를 사용하여 패딩된 시퀀스를 효율적으로 처리
- **드롭아웃 전략**:
  - 은닉 상태에 대한 순환 드롭아웃
  - 레이어 간 드롭아웃은 학습 파라미터에 따라 조절

---

## 🔁 핵심 함수

| 함수명       | 용도 설명 |
|-------------|-----------|
| `ai_chat()` | 실시간 대화 인터페이스 시작 |
| `ask(query)`| OpenAI 또는 커스텀 모델을 이용한 질의 응답 |
| `load_data()`| `.npz` 파일 등에서 데이터셋 로딩 및 전처리 |

---

## ⚙️ 모델 학습 흐름

1. **데이터 로딩**
   - `load_data`로 데이터셋 불러오기 및 정규화 수행
   - 고정 길이로 자르거나 패딩 처리

2. **모델 처리**
   - `lengths` 정보를 함께 넘겨 RNN 최적화

