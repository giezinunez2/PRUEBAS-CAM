import numpy as np
import cv2
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber

# Importar las estructuras de mensaje compiladas
from messages import imagen_pb2 as video_frame_pb2

# --- MODIFICACIÓN: Importar YOLOv8 ---
from ultralytics import YOLO
# --------------------------------------

ecal_core.initialize("Python Video Subscriber with AI")

sub = ProtoSubscriber("video_stream", video_frame_pb2.VideoFrame)

# --- MODIFICACIÓN: Cargar el modelo preentrenado ---
# Usamos 'yolov8n.pt' (versión nano) para un mejor rendimiento en CPU.
# La primera vez se descargará automáticamente.
model = YOLO('yolov8n.pt')
# ----------------------------------------------------

while ecal_core.ok():
    isReceived, msg, _ = sub.receive(100)
    
    if isReceived:
        # 1. Decodificar la imagen del buffer (como antes)
        buf = np.frombuffer(msg.frame_data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        
        if frame is None:
            continue
            
        # --- MODIFICACIÓN: Ejecutar Detección de Objetos ---
        # El modelo 'detecta' sobre el marco original.
        results = model(frame)
        
        # 'plot()' dibuja las cajas delimitadoras y etiquetas sobre una copia.
        annotated_frame = results[0].plot()
        # ----------------------------------------------------
        
        # --- MODIFICACIÓN: Mostrar la imagen procesada ---
        # Cambiamos 'frame' por 'annotated_frame'
        cv2.imshow("Webcam AI Detection RX", annotated_frame)
        # -------------------------------------------------
        
        if cv2.waitKey(1) == 27: # Presionar ESC para salir
            break
            
cv2.destroyAllWindows()
ecal_core.finalize()