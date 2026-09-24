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

# MEJORA: Warm-up del modelo para evitar el "freeze" inicial en la primera inferencia real
_dummy = np.zeros((320, 320, 3), dtype=np.uint8)
model(_dummy, verbose=False)
print("Modelo YOLOv8n listo.")

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

# MEJORA: Parámetros de IA ajustables centralizados
YOLO_IMG_SIZE = 320      # Antes 640 por defecto -> mucho más rápido en CPU
YOLO_CONF_THRESH = 0.35  # Antes 0.4 -> detecta más objetos reales

# --- Variables globales Thread-Safe ---
latest_msg = None
msg_lock = threading.Lock()

# MEJORA: Buffers compartidos entre el hilo de IA y el hilo principal (desacoplados)
latest_frame_front = None
latest_frame_rear = None
frame_lock = threading.Lock()

latest_det_front = []   # [(x1,y1,x2,y2,cls_id,dist_m), ...]
latest_det_rear = []
det_lock = threading.Lock()

is_running = True
ai_fps = 0.0


# --- 2. COMUNICACIÓN ECAL ---
def ecal_callback(topic_name, msg, time_stamp):
    """Callback de eCAL para capturar los datos entrantes del bus."""
    global latest_msg
    with msg_lock:
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


# --- 3. MOTOR GRÁFICO BEV 3D (TABLERO DE AJEDREZ COMO REFERENCIA CARTESIANA) ---
PIXELS_PER_METER = 100     # Escala: 100 px = 1 metro
CELL_METERS = 0.5          # MEJORA: tamaño de cada celda del "tablero de ajedrez"
CELL_PX = int(PIXELS_PER_METER * CELL_METERS)

def crear_panel_bev(alto, ancho, detecciones=[]):
    """
    Genera el panel BEV con un patrón tipo TABLERO DE AJEDREZ.
    Cada celda representa CELL_METERS x CELL_METERS reales, sirviendo
    como referencia visual de escala en el plano cartesiano (X, Z).
    """
    panel = np.zeros((alto, ancho, 3), dtype=np.uint8)
    cx, cy = ancho // 2, alto // 2

    # MEJORA: Dibujar el tablero de ajedrez (celdas alternadas)
    color_a = (32, 32, 32)   # celda oscura
    color_b = (52, 52, 52)   # celda clara
    cols = ancho // CELL_PX + 2
    rows = alto // CELL_PX + 2

    # Offset para que el tablero quede perfectamente centrado en el EGO CAR
    offset_x = cx % CELL_PX
    offset_y = cy % CELL_PX

    for row in range(-1, rows):
        for col in range(-1, cols):
            x1 = col * CELL_PX + offset_x
            y1 = row * CELL_PX + offset_y
            x2 = x1 + CELL_PX
            y2 = y1 + CELL_PX
            if x2 < 0 or y2 < 0 or x1 > ancho or y1 > alto:
                continue
            # Índice de celda relativo al centro para alternar el color correctamente
            idx_col = round((x1 - cx) / CELL_PX)
            idx_row = round((y1 - cy) / CELL_PX)
            color = color_a if (idx_col + idx_row) % 2 == 0 else color_b
            cv2.rectangle(panel, (max(x1, 0), max(y1, 0)),
                          (min(x2, ancho), min(y2, alto)), color, -1)

    # MEJORA: Líneas mayores cada 1 metro (más brillantes) para lectura rápida
    meter_px = PIXELS_PER_METER
    for x in range(cx % meter_px, ancho, meter_px):
        cv2.line(panel, (x, 0), (x, alto), (70, 70, 70), 1)
    for y in range(cy % meter_px, alto, meter_px):
        cv2.line(panel, (0, y), (ancho, y), (70, 70, 70), 1)

    # Ejes principales (naranja) sobre el EGO CAR
    cv2.line(panel, (cx, 0), (cx, alto), (0, 140, 255), 1)
    cv2.line(panel, (0, cy), (ancho, cy), (0, 140, 255), 1)

    # Etiquetas de distancia cada metro
    for offset in range(-4, 5):
        y_pos = cy - offset * meter_px
        if 0 <= y_pos <= alto and offset != 0:
            cv2.putText(panel, f"{offset:+d}m", (cx + 8, y_pos - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)

    # Vehículo Propio (EGO CAR azul en el centro)
    car_w, car_h = 40, 70
    p1 = (cx - car_w // 2, cy - car_h // 2)
    p2 = (cx + car_w // 2, cy + car_h // 2)
    cv2.rectangle(panel, p1, p2, (255, 120, 0), 2)
    cv2.putText(panel, "EGO CAR", (cx - 27, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Proyectar detecciones de la IA en la cuadrícula BEV
    for cls_id, dist_m, cam_type in detecciones:
        px_y = int(cy - (dist_m * PIXELS_PER_METER))
        if 0 <= px_y <= alto:
            box_w, box_h = 30, 30
            bx1, by1 = cx - box_w // 2, px_y - box_h // 2
            bx2, by2 = cx + box_w // 2, px_y + box_h // 2
            cv2.rectangle(panel, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            label = f"{model.names.get(cls_id, 'Obj')} {abs(dist_m):.2f}m"
            cv2.putText(panel, label, (bx1 - 15, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return panel


# --- 4. PROCESAMIENTO DE IA Y DISTANCIAS ---
def calcular_detecciones(boxes, cam_name):
    """Convierte resultados de YOLO en lista de detecciones con distancia real."""
    detecciones = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > YOLO_CONF_THRESH and cls_id in CLASS_REAL_WIDTHS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_px = x2 - x1
            if width_px > 0:
                real_w = CLASS_REAL_WIDTHS[cls_id]
                dist_m = (real_w * FOCAL_LENGTH) / width_px
                dist_bev = dist_m if cam_name == "FRONTAL" else -dist_m
                detecciones.append((x1, y1, x2, y2, cls_id, dist_m, dist_bev))
    return detecciones


def dibujar_boxes(frame, detecciones, cam_name):
    """Dibuja las cajas y etiquetas sobre el frame de la cámara."""
    if frame is None:
        return frame
    for (x1, y1, x2, y2, cls_id, dist_m, dist_bev) in detecciones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{model.names[cls_id]} {dist_m:.2f}m"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"CAMARA {cam_name}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


# --- 5. HILO DE INFERENCIA IA (DESACOPLADO DEL RENDER -> ELIMINA LATENCIA) ---
def inference_worker():
    """
    MEJORA CLAVE DE LATENCIA:
    Corre YOLO en un hilo independiente, en batch (frontal+trasera juntas),
    a la velocidad máxima que la CPU permita, sin bloquear el refresco de video.
    """
    global latest_det_front, latest_det_rear, ai_fps

    while is_running:
        with frame_lock:
            f_front = None if latest_frame_front is None else latest_frame_front.copy()
            f_rear = None if latest_frame_rear is None else latest_frame_rear.copy()

        if f_front is None and f_rear is None:
            time.sleep(0.01)
            continue

        t0 = time.time()
        batch = []
        idx_map = []  # 'FRONTAL' o 'TRASERA' por posición del batch
        if f_front is not None:
            batch.append(f_front)
            idx_map.append("FRONTAL")
        if f_rear is not None:
            batch.append(f_rear)
            idx_map.append("TRASERA")

        # MEJORA: una sola llamada batch en vez de dos llamadas separadas
        results = model(batch, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_THRESH, verbose=False)

        det_f, det_r = [], []
        n_detecciones = 0
        for i, res in enumerate(results):
            dets = calcular_detecciones(res.boxes, idx_map[i])
            n_detecciones += len(dets)
            if idx_map[i] == "FRONTAL":
                det_f = dets
            else:
                det_r = dets

        with det_lock:
            latest_det_front = det_f
            latest_det_rear = det_r

        dt = time.time() - t0
        ai_fps = 1.0 / dt if dt > 0 else 0.0

        # MEJORA: diagnóstico en consola para saber si la IA realmente detecta algo
        if n_detecciones == 0:
            print(f"[IA] Sin detecciones este ciclo (FPS IA: {ai_fps:.1f}). "
                  f"Si usas el patrón simulado (circulo), es normal: no es un objeto COCO real.")
        else:
            print(f"[IA] {n_detecciones} objeto(s) detectado(s) (FPS IA: {ai_fps:.1f})")


# --- 6. BUCLE PRINCIPAL Y RENDERING ---
def main():
    global latest_msg, latest_frame_front, latest_frame_rear, is_running

    ecal_core.initialize("Python ADAS Subscriber")
    sub = ProtoSubscriber("trinocular_stream", trinocular_pb2.TripleVideoFrame)
    sub.set_callback(ecal_callback)

    # MEJORA: lanzar el hilo de IA independiente del render
    ai_thread = threading.Thread(target=inference_worker, daemon=True)
    ai_thread.start()

    print("Receptor eCAL BYD ADAS Activo. Esperando flujo de datos...")

    while ecal_core.ok():
        msg = None
        with msg_lock:
            if latest_msg is not None:
                msg = latest_msg

        if msg is None:
            time.sleep(0.005)
            continue

        # Decodificación según la estructura Protobuf (center / right)
        frame_front = decode_frame(msg.center) if (hasattr(msg, "center") and msg.HasField("center")) else None
        frame_rear = decode_frame(msg.right) if (hasattr(msg, "right") and msg.HasField("right")) else (decode_frame(msg.left) if (hasattr(msg, "left") and msg.HasField("left")) else None)

        if frame_front is not None:
            frame_front = cv2.resize(frame_front, (640, 480))
        else:
            frame_front = np.zeros((480, 640, 3), dtype=np.uint8)

        if frame_rear is not None:
            frame_rear = cv2.resize(frame_rear, (640, 480))
        else:
            frame_rear = np.zeros((480, 640, 3), dtype=np.uint8)

        # MEJORA: entregar los frames actuales al hilo de IA (sin bloquear el render)
        with frame_lock:
            latest_frame_front = frame_front.copy()
            latest_frame_rear = frame_rear.copy()

        # MEJORA: usar las últimas detecciones disponibles (aunque vengan de un frame anterior)
        with det_lock:
            det_f = list(latest_det_front)
            det_r = list(latest_det_rear)

        frame_front = dibujar_boxes(frame_front, det_f, "FRONTAL")
        frame_rear = dibujar_boxes(frame_rear, det_r, "TRASERA")

        detecciones_bev = [(d[4], d[6], "FRONTAL") for d in det_f] + \
                           [(d[4], d[6], "TRASERA") for d in det_r]

        # Columna Izquierda: Apilar cámaras (640x960 px)
        left_column = np.vstack((frame_front, frame_rear))

        # Panel Derecho: Vista BEV tipo tablero de ajedrez (640x960 px)
        bev_panel = crear_panel_bev(960, 640, detecciones_bev)

        dashboard = np.hstack((left_column, bev_panel))
        dashboard_resized = cv2.resize(dashboard, (1280, 720))

        # MEJORA: overlay de FPS de IA para monitoreo de latencia en vivo
        cv2.putText(dashboard_resized, f"IA FPS: {ai_fps:.1f}", (10, 710),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("BYD ADAS 3D Simulator", dashboard_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    is_running = False
    time.sleep(0.1)
    cv2.destroyAllWindows()
    ecal_core.finalize()


if __name__ == "__main__":
    main()