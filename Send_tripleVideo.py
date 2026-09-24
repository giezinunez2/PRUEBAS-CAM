import sys
import os
import cv2
import time
import numpy as np
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher

try:
    from messages import imagen_pb2 as video_frame_pb2, trinocular_pb2
except ImportError:
    import imagen_pb2 as video_frame_pb2, trinocular_pb2

# --- CONFIGURACIÓN DE PUERTOS DE CÁMARAS ---
PUERTO_FRONTAL = 0  # /dev/video0
PUERTO_TRASERA = 2  # Probar 2 o 4 si no da video

ecal_core.initialize("Python Trinocular Publisher")
pub = ProtoPublisher("trinocular_stream", trinocular_pb2.TripleVideoFrame)

cap_front = cv2.VideoCapture(PUERTO_FRONTAL, cv2.CAP_V4L2)
cap_rear = cv2.VideoCapture(PUERTO_TRASERA, cv2.CAP_V4L2)

counter = 0
quality = 80

def fill_frame(msg_frame, frame, counter, quality):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok: 
        return False
    h, w = frame.shape[:2]
    msg_frame.frame_data = buf.tobytes()
    msg_frame.width = w
    msg_frame.height = h
    msg_frame.channels = frame.shape[2] if frame.ndim == 3 else 1
    msg_frame.encoding = video_frame_pb2.JPEG
    msg_frame.frame_number = counter
    msg_frame.timestamp = time.time()
    msg_frame.compression_quality = quality
    msg_frame.is_keyframe = True
    return True

msg = trinocular_pb2.TripleVideoFrame()
print(f"Emisor eCAL Activo: Frontal ({PUERTO_FRONTAL}) | Trasera ({PUERTO_TRASERA})...")

while ecal_core.ok():
    ret_f, frame_f = cap_front.read() if cap_front.isOpened() else (False, None)
    if not ret_f or frame_f is None:
        frame_f = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame_f, f"FRONTAL ({PUERTO_FRONTAL}) NO DETECTADA", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    ret_r, frame_r = cap_rear.read() if cap_rear.isOpened() else (False, None)
    if not ret_r or frame_r is None:
        frame_r = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame_r, f"TRASERA ({PUERTO_TRASERA}) NO DETECTADA", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fill_frame(msg.center, frame_f, counter, quality)
    fill_frame(msg.right, frame_r, counter, quality)

    pub.send(msg)
    counter += 1
    time.sleep(0.03)

if cap_front.isOpened(): cap_front.release()
if cap_rear.isOpened(): cap_rear.release()
ecal_core.finalize()
