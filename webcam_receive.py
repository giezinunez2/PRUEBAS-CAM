import numpy as np
import cv2

import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber

try:
    from messages import imagen_pb2 as video_frame_pb2
except ImportError:
    import imagen_pb2 as video_frame_pb2

ecal_core.initialize("Python Video Subscriber")
sub = ProtoSubscriber("video_stream", video_frame_pb2.VideoFrame)

print("Nivel 1 - Receptor de 1 cámara activo (sin IA). Esperando video...")

while ecal_core.ok():
    isReceived, msg, _ = sub.receive(100)
    if isReceived and msg.frame_data:
        buf = np.frombuffer(msg.frame_data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        cv2.imshow("Webcam RX", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

cv2.destroyAllWindows()
ecal_core.finalize()