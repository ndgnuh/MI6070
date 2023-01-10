from vietocr.tool.config import list_configs
from eyeball import config as eyeball_configs
from os import path
from PIL import Image
import json
import ocr
import streamlit as st
import os
os.system('clear')

vietocr_model = st.experimental_singleton(ocr.vietocr_model)
detection_model = st.experimental_singleton(ocr.detector_model)

# st.write(eyeball_configs.configs)
eyeball_config = st.selectbox(
    "Detection model",
    eyeball_configs.configs.values(),
    format_func=path.basename
)
vietocr_config = st.selectbox(
    "Recognition model",
    list_configs()
)

with st.spinner():
    text_recognizer = vietocr_model(vietocr_config, 'cuda')
    text_detector = detection_model(eyeball_config)

file = st.file_uploader("Thêm ảnh")
if file is None:
    st.stop()

file = Image.open(file).convert("RGB")
col1, col2 = st.columns(2)
with col1:
    st.subheader("File gốc")
    st.image(file)

with col2:
    with st.spinner("Detecting texts"):
        results = ocr.detect_text(text_detector, file)
        boxes = [tuple(map(int, r['box'])) for r in results]
    with st.spinner("Transcribing"):
        texts = ocr.transcribe_text(text_recognizer, file, boxes)

    for (result, text) in zip(results, texts):
        result['text'] = text
        result['score'] = float(result['score'])

    output = ocr.reconstruct(file, texts, boxes)
    st.subheader("File tái dựng")
    st.image(output)

st.download_button("Tải về", json.dumps(results))
