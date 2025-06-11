
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
## 모델 성능
![output](https://github.com/user-attachments/assets/6c1b1efb-443f-48e7-8895-6f735e0209ea)
131개의 단어에 대해서 92% test accuarcy를 달성했다 

---

## 모델 학습 과정

1. **데이터 준비**
   - [AIhub 수어 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103)
   - 선별된 131개의 단어에 대해서 정면 영상 (F)를 이용했다
   - 총 1401개의 데이터를 사용했다
2. **키포인트 정규화 및 라벨링**
   왼손(21x2), 오른손(21x2) 총 84개의 키포인트를 이용했다.  
   - `save_keypoint_norm.ipynb` 실행
   - `save_data_label.ipynb` 실행

3. **모델 학습**
   - `train_gru_tensorflow.ipynb`에서 모델 학습 수행

4. **응용 및 결과 확인**
   - `application2.ipynb` 실행

---

**시퀀스 처리 방식**:
  - 지정된 `max_length`를 초과하는 시퀀스는 잘리고, 짧은 시퀀스는 0으로 패딩됩니다.
  - 모델 학습 시 `nn.utils.rnn.pack_padded_sequence`를 사용하여 패딩된 시퀀스를 처리합니다

---

## 모델 구성 설명
이 프로젝트는 GRU (Gated Recurrent Unit) 기반의 순환 신경망을 이용해 수어(수화) 시계열 데이터를 분류합니다.
입력은 MediaPipe를 통해 추출한 손 키포인트 좌표 시계열이며, 일반적으로 (batch_size, time_steps, keypoints) 형태의 데이터를 사용합니다.

GRU는 RNN의 일종으로, 시간적 정보를 유지하면서도 계산량이 비교적 적어 학습이 빠르다는 장점이 있습니다.
이 프로젝트에서는 다음과 같은 네트워크 구성을 따릅니다:
- GRU(units=128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)
     - 입력 시퀀스를 처리하면서 과적합을 방지하기 위해 dropout과 recurrent_dropout을 함께 사용합니다.
- Dropout(0.3)
     - GRU 외부에서도 추가적인 정규화를 적용합니다.
- Dense(units=num_classes, activation='softmax')
     - 수어 단어 분류를 위한 최종 출력층입니다.

여기서 recurrent_dropout은 **순환 연결**(이전 시점 은닉 상태로부터 현재 은닉 상태로 가는 경로)에도 드롭아웃을 적용하는 기능입니다.
일반적인 dropout이 입력과 출력의 연결을 무작위로 끊는다면, recurrent_dropout은 시계열 정보의 흐름 자체에도 일정 확률로 끊김을 줌으로써 모델이 특정 시간 패턴에 과도하게 의존하는 것을 막아줍니다.
이는 특히 학습 데이터가 제한적이거나 noise가 포함된 경우 유용하며, 모델의 일반화 성능을 높여주는 역할을 합니다.

이러한 구조 덕분에 모델은 수어 동작의 시간적 흐름과 손 모양의 변화를 효과적으로 학습하고, 다양한 문맥 속에서도 안정적인 수어 단어 예측이 가능합니다.

---

## 📌 기타 참고사항

- `light_data_label/` 디렉토리에는 라벨링된 엑셀 파일들이 포함되어 있습니다.
- `.ttf` 파일(`AppleGothic.ttf`)은 한글 시각화를 위한 폰트 리소스로 사용될 수 있습니다.
- `.json` 파일(`word2idx2.json`, `idx2word2.json`)은 모델 입출력용 단어 인덱스 매핑에 사용됩니다.
