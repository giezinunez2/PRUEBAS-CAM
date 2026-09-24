import sys
import os
import cv2
import time
import numpy as np

import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher

try:
    from messages import imagen_pb2 as video_frame_pb2
except ImportError:
    import imagen_pb2 as video_frame_pb2

ecal_core.initialize("Python Video Publisher")
pub = ProtoPublisher("video_stream", video_frame_pb2.VideoFrame)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
counter = 0
quality = 80
msg = video_frame_pb2.VideoFrame()

print("Nivel 1 - Emisor de 1 cámara activo (sin IA)...")

while ecal_core.ok():
    ret, frame = cap.read() if cap.isOpened() else (False, None)

    # Fallback: si no hay cámara física, no se detiene el pipeline de prueba
    if not ret or frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "CAMARA NO DETECTADA (0)", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Webcam TX (local)", frame)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        continue

    h, w = frame.shape[:2]
    msg.frame_data = buf.tobytes()
    msg.width = w
    msg.height = h
    msg.channels = frame.shape[2] if frame.ndim == 3 else 1
    msg.encoding = video_frame_pb2.JPEG
    msg.frame_number = counter
    msg.timestamp = time.time()
    msg.compression_quality = quality
    msg.is_keyframe = True

    pub.send(msg)
    counter += 1

    if cv2.waitKey(1) == 27:  # ESC
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
ecal_core.finalize()