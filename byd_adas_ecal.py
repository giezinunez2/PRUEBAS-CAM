import sys
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO

import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber

# Parche de compatibilidad bytearray -> bytes para Protobuf
_orig_on_receive = ProtoSubscriber._on_receive
def _patched_on_receive(self, topic_name, msg, time_stamp):
    if isinstance(msg, (bytearray, memoryview)):
        msg = bytes(msg)
    return _orig_on_receive(self, topic_name, msg, time_stamp)
ProtoSubscriber._on_receive = _patched_on_receive

try:
    from messages import imagen_pb2, trinocular_pb2
except ImportError:
    import imagen_pb2, trinocular_pb2

print("Iniciando Modelo YOLOv8n...")
model = YOLO("yolov8n.pt")
model(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
print("Modelo YOLOv8n cargado.")

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
FOCAL_LENGTH = 700.0
YOLO_IMG_SIZE = 320
YOLO_CONF_THRESH = 0.35

latest_msg = None
msg_lock = threading.Lock()

latest_frame_front = None
latest_frame_rear = None
frame_lock = threading.Lock()

latest_det_front = []
latest_det_rear = []
det_lock = threading.Lock()

is_running = True
ai_fps = 0.0

def ecal_callback(topic_name, msg, time_stamp):
    global latest_msg
    with msg_lock:
        latest_msg = msg

def decode_frame(img_pb):
    if img_pb is None:
        return None
    data_bytes = None
    if isinstance(img_pb, (bytes, bytearray, memoryview)):
        data_bytes = bytes(img_pb)
    elif hasattr(img_pb, "frame_data") and img_pb.frame_data:
        data_bytes = bytes(img_pb.frame_data)
    if not data_bytes:
        return None
    np_arr = np.frombuffer(data_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

PIXELS_PER_METER = 100
CELL_METERS = 0.5
CELL_PX = int(PIXELS_PER_METER * CELL_METERS)

def crear_panel_bev(alto, ancho, detecciones=[]):
    panel = np.zeros((alto, ancho, 3), dtype=np.uint8)
    cx, cy = ancho // 2, alto // 2

    color_a = (32, 32, 32)
    color_b = (52, 52, 52)
    cols = ancho // CELL_PX + 2
    rows = alto // CELL_PX + 2

    offset_x = cx % CELL_PX
    offset_y = cy % CELL_PX

    for row in range(-1, rows):
        for col in range(-1, cols):
            x1 = col * CELL_PX + offset_x
            y1 = row * CELL_PX + offset_y
            x2, y2 = x1 + CELL_PX, y1 + CELL_PX
            if x2 < 0 or y2 < 0 or x1 > ancho or y1 > alto:
                continue
            idx_col = round((x1 - cx) / CELL_PX)
            idx_row = round((y1 - cy) / CELL_PX)
            color = color_a if (idx_col + idx_row) % 2 == 0 else color_b
            cv2.rectangle(panel, (max(x1, 0), max(y1, 0)), (min(x2, ancho), min(y2, alto)), color, -1)

    meter_px = PIXELS_PER_METER
    for x in range(cx % meter_px, ancho, meter_px):
        cv2.line(panel, (x, 0), (x, alto), (70, 70, 70), 1)
    for y in range(cy % meter_px, alto, meter_px):
        cv2.line(panel, (0, y), (ancho, y), (70, 70, 70), 1)

    cv2.line(panel, (cx, 0), (cx, alto), (0, 140, 255), 1)
    cv2.line(panel, (0, cy), (ancho, cy), (0, 140, 255), 1)

    for offset in range(-4, 5):
        y_pos = cy - offset * meter_px
        if 0 <= y_pos <= alto and offset != 0:
            cv2.putText(panel, f"{offset:+d}m", (cx + 8, y_pos - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)

    car_w, car_h = 40, 70
    p1 = (cx - car_w // 2, cy - car_h // 2)
    p2 = (cx + car_w // 2, cy + car_h // 2)
    cv2.rectangle(panel, p1, p2, (255, 120, 0), 2)
    cv2.putText(panel, "EGO CAR", (cx - 27, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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

def calcular_detecciones(boxes, cam_name):
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

def inference_worker():
    global latest_det_front, latest_det_rear, ai_fps
    while is_running:
        with frame_lock:
            f_front = None if latest_frame_front is None else latest_frame_front.copy()
            f_rear = None if latest_frame_rear is None else latest_frame_rear.copy()

        if f_front is None and f_rear is None:
            time.sleep(0.01)
            continue

        t0 = time.time()
        batch, idx_map = [], []
        if f_front is not None:
            batch.append(f_front)
            idx_map.append("FRONTAL")
        if f_rear is not None:
            batch.append(f_rear)
            idx_map.append("TRASERA")

        results = model(batch, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_THRESH, verbose=False)
        det_f, det_r = [], []
        for i, res in enumerate(results):
            dets = calcular_detecciones(res.boxes, idx_map[i])
            if idx_map[i] == "FRONTAL":
                det_f = dets
            else:
                det_r = dets

        with det_lock:
            latest_det_front = det_f
            latest_det_rear = det_r

        dt = time.time() - t0
        ai_fps = 1.0 / dt if dt > 0 else 0.0

def main():
    global latest_msg, latest_frame_front, latest_frame_rear, is_running

    ecal_core.initialize("Python ADAS Subscriber")
    sub = ProtoSubscriber("trinocular_stream", trinocular_pb2.TripleVideoFrame)
    sub.set_callback(ecal_callback)

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

        frame_front = decode_frame(msg.center) if (hasattr(msg, "center") and msg.HasField("center")) else None
        frame_rear = decode_frame(msg.right) if (hasattr(msg, "right") and msg.HasField("right")) else (decode_frame(msg.left) if (hasattr(msg, "left") and msg.HasField("left")) else None)

        frame_front = cv2.resize(frame_front, (640, 480)) if frame_front is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        frame_rear = cv2.resize(frame_rear, (640, 480)) if frame_rear is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        with frame_lock:
            latest_frame_front = frame_front.copy()
            latest_frame_rear = frame_rear.copy()

        with det_lock:
            det_f = list(latest_det_front)
            det_r = list(latest_det_rear)

        frame_front = dibujar_boxes(frame_front, det_f, "FRONTAL")
        frame_rear = dibujar_boxes(frame_rear, det_r, "TRASERA")

        detecciones_bev = [(d[4], d[6], "FRONTAL") for d in det_f] + [(d[4], d[6], "TRASERA") for d in det_r]

        left_column = np.vstack((frame_front, frame_rear))
        bev_panel = crear_panel_bev(960, 640, detecciones_bev)

        dashboard = np.hstack((left_column, bev_panel))
        dashboard_resized = cv2.resize(dashboard, (1280, 720))

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
