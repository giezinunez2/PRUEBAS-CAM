import numpy as np
import cv2

import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from ultralytics import YOLO

try:
    from messages import imagen_pb2 as video_frame_pb2
except ImportError:
    import imagen_pb2 as video_frame_pb2

ecal_core.initialize("Python Video Subscriber with AI")
sub = ProtoSubscriber("video_stream", video_frame_pb2.VideoFrame)

print("Cargando YOLOv8n...")
model = YOLO("yolov8n.pt")
model(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)  # warm-up
print("Nivel 2 - Receptor con IA activo. Esperando video de webcam_send.py...")

while ecal_core.ok():
    isReceived, msg, _ = sub.receive(100)

    if isReceived and msg.frame_data:
        buf = np.frombuffer(msg.frame_data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        results = model(frame, imgsz=320, conf=0.35, verbose=False)
        annotated_frame = results[0].plot()

        n = len(results[0].boxes)
        print(f"[IA] {n} objeto(s) detectado(s)" if n else "[IA] sin detecciones")

        cv2.imshow("Webcam AI Detection RX", annotated_frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

cv2.destroyAllWindows()
ecal_core.finalize()