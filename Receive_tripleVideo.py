import numpy as np
import cv2
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber

try:
    from messages import trinocular_pb2
except ImportError:
    import trinocular_pb2

ecal_core.initialize("Python Trinocular Subscriber")
sub = ProtoSubscriber("trinocular_stream", trinocular_pb2.TripleVideoFrame)


def decode_frame(msg_frame, label):
    if msg_frame is None or not msg_frame.frame_data:
        return None
    buf = np.frombuffer(msg_frame.frame_data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


print("Nivel 3 - Receptor trinocular (sin IA). Esperando frames de Send_tripleVideo.py...")

while ecal_core.ok():
    isReceived, msg, _ = sub.receive(100)
    if not isReceived:
        continue

    # FIX: ya no exige los 3 campos. Send_tripleVideo.py solo llena center + right.
    center = decode_frame(msg.center, "FRONTAL (center)") if msg.HasField("center") else None
    right = decode_frame(msg.right, "TRASERA (right)") if msg.HasField("right") else None
    left = decode_frame(msg.left, "LEFT") if msg.HasField("left") else None

    if center is not None:
        cv2.imshow("Frontal RX", center)
    if right is not None:
        cv2.imshow("Trasera RX", right)
    if left is not None:
        cv2.imshow("Left RX", left)

    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
ecal_core.finalize()