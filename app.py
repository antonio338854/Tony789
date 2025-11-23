import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
import urllib.request
import ssl

st.set_page_config(page_title="Vigilante 3.9", layout="centered")
st.title("👁️ Vigilância Ativa (Versão Estável)")

# --- Download Anti-Falha ---
def download_files():
    ssl._create_default_https_context = ssl._create_unverified_context
    files = ["MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel"]
    mirrors = {
        "MobileNetSSD_deploy.prototxt": ["https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"],
        "MobileNetSSD_deploy.caffemodel": ["https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel"]
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for f in files:
        if not os.path.exists(f):
            for url in mirrors[f]:
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as resp, open(f, 'wb') as out:
                        out.write(resp.read())
                    if os.path.getsize(f) > 100: break
                except: continue

download_files()

# --- Configuração Neural ---
CLASSES = ["fundo", "aviao", "bicicleta", "passaro", "barco", "garrafa", "onibus", "carro", "gato", "cadeira", "vaca", "mesa", "cachorro", "cavalo", "moto", "PESSOA", "planta", "ovelha", "sofa", "trem", "tv"]
COLOR_RED = (0, 0, 255)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# --- O NOVO PROCESSADOR (Callback) ---
# Substitui o antigo VideoProcessor que estava travando seu app
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # Processamento IA
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Desenha Retângulo
            cv2.rectangle(image, (startX, startY), (endX, endY), COLOR_RED, 2)
            label = f"{CLASSES[idx]}: {confidence*100:.0f}%"
            cv2.putText(image, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- Streaming Seguro ---
webrtc_streamer(
    key="camera-fix",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    video_frame_callback=video_frame_callback, # <--- O SEGREDO ESTÁ AQUI
    async_processing=True
)
