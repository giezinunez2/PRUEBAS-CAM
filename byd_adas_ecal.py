import sys
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO

# Importaciones de eCAL
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber

# Parche de compatibilidad bytearray -> bytes para Protobuf
_orig_on_receive = ProtoSubscriber._on_receive
def _patched_on_receive(self, topic_name, msg, time_stamp):
    if isinstance(msg, (bytearray, memoryview)):
        msg = bytes(msg)
    return _orig_on_receive(self, topic_name, msg, time_stamp)
ProtoSubscriber._on_receive = _patched_on_receive


# Importar estructuras Protobuf compiladas
try:
    from messages import imagen_pb2, trinocular_pb2
except ImportError:
    import imagen_pb2, trinocular_pb2


# --- 1. CONFIGURACIÓN Y MODELO DE IA ---
print("Iniciando Modelo YOLOv8n...")
model = YOLO("yolov8n.pt")

# Ancho físico estimado en metros para cálculo de distancia
CLASS_REAL_WIDTHS = {
    0: 0.45,  # Persona
    2: 1.80,  # Auto
    3: 0.80,  # Moto
    5: 2.50,  # Autobús
    7: 2.50,  # Camión
    15: 0.25, # Gato
    16: 0.35, # Perro
    39: 0.07  # Botella
}
FOCAL_LENGTH = 700.0  # Calibración focal estimada

# Variables globales para sincronización de hilos (Thread-Safe)
latest_msg = None
data_lock = threading.Lock()


# --- 2. COMUNICACIÓN ECAL ---
def ecal_callback(topic_name, msg, time_stamp):
    """Callback de eCAL para capturar los datos entrantes del bus."""
    global latest_msg
    with data_lock:
        latest_msg = msg


def decode_frame(img_pb):
    """Decodifica el buffer JPEG a OpenCV tolerando bytes directos u objetos Protobuf."""
    if img_pb is None:
        return None

    data_bytes = None
    if isinstance(img_pb, (bytes, bytearray, memoryview)):
        data_bytes = bytes(img_pb)
    elif hasattr(img_pb, "frame_data") and img_pb.frame_data:
        # Campo real usado por imagen_pb2.VideoFrame (Send_tripleVideo.py / webcam_send.py)
        data_bytes = bytes(img_pb.frame_data)
    elif hasattr(img_pb, "data") and img_pb.data:
        data_bytes = bytes(img_pb.data)
    elif hasattr(img_pb, "mat_data") and img_pb.mat_data:
        data_bytes = bytes(img_pb.mat_data)
    elif hasattr(img_pb, "buffer") and img_pb.buffer:
        data_bytes = bytes(img_pb.buffer)

    if not data_bytes:
        return None

    np_arr = np.frombuffer(data_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# --- 3. MOTOR GRÁFICO BEV 3D (VISTA DE PÁJARO) ---
def crear_panel_bev(alto, ancho, detecciones=[]):
    """Genera la cuadrícula BEV graduada en metros con el EGO CAR central."""
    panel = np.zeros((alto, ancho, 3), dtype=np.uint8)
    cx, cy = ancho // 2, alto // 2

    # Líneas guía de ejes principales (Naranja/Marrón)
    cv2.line(panel, (cx, 0), (cx, alto), (0, 140, 255), 1)
    cv2.line(panel, (0, cy), (ancho, cy), (0, 140, 255), 1)

    # Cuadrícula y escala graduada de distancias (+1.20m, -1.20m, etc.)
    for offset_y in range(-300, 320, 40):
        y_pos = cy - offset_y
        if 0 <= y_pos <= alto:
            cv2.line(panel, (cx - 140, y_pos), (cx + 140, y_pos), (35, 35, 35), 1)
            dist_m = offset_y / 100.0
            cv2.putText(panel, f"{dist_m:+.2f}m", (cx + 145, y_pos + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    # Dibujar Vehículo Propio (EGO CAR azul en el centro)
    car_w, car_h = 40, 70
    p1 = (cx - car_w // 2, cy - car_h // 2)
    p2 = (cx + car_w // 2, cy + car_h // 2)
    cv2.rectangle(panel, p1, p2, (255, 120, 0), 2)
    cv2.putText(panel, "EGO CAR", (cx - 27, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Proyectar detecciones de la IA en la cuadrícula BEV
    for cls_id, dist_m, cam_type in detecciones:
        # Convertir distancia en metros a posición en píxeles Y
        px_y = int(cy - (dist_m * 100.0))

        if 0 <= px_y <= alto:
            box_w, box_h = 30, 30
            bx1, by1 = cx - box_w // 2, px_y - box_h // 2
            bx2, by2 = cx + box_w // 2, px_y + box_h // 2

            # Recuadro amarillo para objetos detectados en BEV
            cv2.rectangle(panel, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            label = f"{model.names.get(cls_id, 'Obj')} {abs(dist_m):.2f}m"
            cv2.putText(panel, label, (bx1 - 15, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return panel


# --- 4. PROCESAMIENTO DE IA Y DISTANCIAS ---
def procesar_ia(frame, cam_name):
    """Ejecuta la inferencia YOLO y calcula las distancias reales."""
    detecciones = []
    if frame is None:
        return frame, detecciones

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > 0.4 and cls_id in CLASS_REAL_WIDTHS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_px = x2 - x1

            if width_px > 0:
                real_w = CLASS_REAL_WIDTHS[cls_id]
                dist_m = (real_w * FOCAL_LENGTH) / width_px

                # Dibujar bounding box en la imagen de la cámara
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]} {dist_m:.2f}m"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Si es cámara trasera, la distancia en el mapa BEV es negativa
                dist_bev = dist_m if cam_name == "FRONTAL" else -dist_m
                detecciones.append((cls_id, dist_bev, cam_name))

    # Superponer etiqueta de la cámara
    cv2.putText(frame, f"CAMARA {cam_name}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, detecciones


# --- 5. BUCLE PRINCIPAL Y RENDERING ---
def main():
    global latest_msg

    # Inicialización del nodo eCAL
    ecal_core.initialize("Python ADAS Subscriber")
    sub = ProtoSubscriber("trinocular_stream", trinocular_pb2.TripleVideoFrame)
    sub.set_callback(ecal_callback)

    print("Receptor eCAL BYD ADAS Activo. Esperando flujo de datos...")

    frame_count = 0
    last_detecciones = []

    while ecal_core.ok():
        msg = None
        with data_lock:
            if latest_msg is not None:
                msg = latest_msg

        if msg is None:
            time.sleep(0.005)
            continue

        frame_count += 1

        # Decodificación segura según la estructura Protobuf (center / right)
        frame_front = decode_frame(msg.center) if (hasattr(msg, "center") and msg.HasField("center")) else None
        frame_rear = decode_frame(msg.right) if (hasattr(msg, "right") and msg.HasField("right")) else (decode_frame(msg.left) if (hasattr(msg, "left") and msg.HasField("left")) else None)

        # Ajustar dimensiones de cámaras a 640x480
        if frame_front is not None:
            frame_front = cv2.resize(frame_front, (640, 480))
        else:
            frame_front = np.zeros((480, 640, 3), dtype=np.uint8)

        if frame_rear is not None:
            frame_rear = cv2.resize(frame_rear, (640, 480))
        else:
            frame_rear = np.zeros((480, 640, 3), dtype=np.uint8)

        # Optimización anti-latencia: Ejecutar IA cada 2 fotogramas
        if frame_count % 2 == 0:
            frame_front, det_f = procesar_ia(frame_front, "FRONTAL")
            frame_rear, det_r = procesar_ia(frame_rear, "TRASERA")
            last_detecciones = det_f + det_r
        else:
            cv2.putText(frame_front, "CAMARA FRONTAL", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_rear, "CAMARA TRASERA", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Columna Izquierda: Apilar cámaras (640x960 px)
        left_column = np.vstack((frame_front, frame_rear))

        # Panel Derecho: Vista 3D BEV con EGO CAR (640x960 px)
        bev_panel = crear_panel_bev(960, 640, last_detecciones)

        # Unir Columna Izquierda + Panel BEV para crear el Dashboard
        dashboard = np.hstack((left_column, bev_panel))

        # Escalar ventana para adaptarse a la pantalla (1280x720)
        dashboard_resized = cv2.resize(dashboard, (1280, 720))

        # Renderizar en la interfaz gráfica
        cv2.imshow("BYD ADAS 3D Simulator", dashboard_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    ecal_core.finalize()


if __name__ == "__main__":
    main()