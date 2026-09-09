import os
import numpy as np
import pandas as pd
from PIL import Image
import openvino as ov
from tensorflow import keras
import streamlit as st

# ── 1. 페이지 설정 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenVINO 제품 불량 검사",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 가죽 제품 불량 검사 AI (OpenVINO)")
st.caption("OpenVINO로 경량화된 모델을 사용하여 빠른 불량 검사를 수행합니다.")

# ── 설정 ─────────────────────────────────────────────────────────
MODEL_XML_PATH = "./weights/leather_model.xml"
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]


# ── 2. OpenVINO 모델 로드 (Streamlit 캐싱) ────────────────────────────────
@st.cache_resource
def load_ov_model():
    if not os.path.exists(MODEL_XML_PATH):
        st.error(f"OpenVINO 모델 파일이 없습니다: {MODEL_XML_PATH}")
        return None
    core = ov.Core()
    model = core.read_model(MODEL_XML_PATH)
    compiled_model = core.compile_model(model, "CPU")
    return compiled_model

compiled_model = load_ov_model()


# ── 전처리 및 추론 함수 ────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img):
    arr = preprocess(pil_img)
    output = model(arr)[model.output(0)]
    prob = float(output[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ── 3. 이미지 입력 ─────────────────────────────────────────────────────────
input_mode = st.radio("입력 방식을 선택하세요", ["파일 업로드", "카메라 촬영"], horizontal=True)

pil_img = None

if input_mode == "파일 업로드":
    uploaded_file = st.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
elif input_mode == "카메라 촬영":
    camera_file = st.camera_input("카메라로 사진을 촬영하세요")
    if camera_file is not None:
        pil_img = Image.open(camera_file)

# 이미지 미리보기
if pil_img is not None:
    st.image(pil_img, caption="입력된 이미지", use_column_width=True)

    # ── 4. 검사 실행 ──────────────────────────────────────────────────────
    if st.button("검사 시작", type="primary", use_container_width=True):
        if compiled_model is None:
            st.error("OpenVINO 모델이 로드되지 않았습니다.")
        else:
            with st.spinner("OpenVINO AI 모델이 분석 중입니다..."):
                label, prob = predict(compiled_model, pil_img)
                normal_prob = 1.0 - prob
                defect_prob = prob

            st.divider()

            # ── 5. 결과 표시 ──────────────────────────────────────────────
            if label == "정상":
                st.success(f"🎉 검사 결과: **{label}** 입니다.")
            else:
                st.error(f"⚠️ 검사 결과: **{label}** 입니다.")

            col1, col2 = st.columns(2)
            col1.metric(label="정상 확률", value=f"{normal_prob:.1%}")
            col2.metric(label="불량 확률", value=f"{defect_prob:.1%}")

            st.write("#### 📊 불량 확률 그래프")
            chart_data = pd.DataFrame({
                "상태": ["불량 확률"],
                "확률 (%)": [defect_prob * 100]
            })
            st.bar_chart(chart_data.set_index("상태"))