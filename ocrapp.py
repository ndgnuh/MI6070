from vietocr.tool.config import list_configs
from PIL import Image
import json
import ocr
import streamlit as st
import os
os.system('clear')

vietocr_model = st.experimental_singleton(ocr.vietocr_model)
doctr_model = st.experimental_singleton(ocr.doctr_model)

config = st.selectbox("Model", list_configs())

with st.spinner():
    text_recognizer = vietocr_model(config, 'cuda')
    text_detector = doctr_model()

file = st.file_uploader("Thêm ảnh")
if file is None:
    st.stop()

file = Image.open(file).convert("RGB")
st.subheader("File gốc")
st.image(file)
with st.spinner("Detecting texts"):
    boxes, scores = ocr.detect_text(text_detector, file)
with st.spinner("Transcribing"):
    texts = ocr.transcribe_text(text_recognizer, file, boxes)

output = ocr.reconstruct(file, texts, boxes, scores)
st.subheader("File tái dựng")
st.image(output)

st.download_button("Tải về", json.dumps(
    {"texts": texts, "boxes": boxes, "scores": scores}))
