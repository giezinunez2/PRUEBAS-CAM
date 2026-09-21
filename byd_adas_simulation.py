import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# Mapeo de IDs de COCO a nombres en español
CLASS_NAMES = {
    0: "Persona",
    2: "Auto",
    3: "Moto",
    5: "Autobus",
    7: "Camion",
    15: "Gato",
    16: "Perro"
}

# -------------------------------------------------------------
# 1. Lector de Cámara Asíncrono con Blindaje de Errores
# -------------------------------------------------------------
class AsyncCamera:
    def __init__(self, src, name="Camera"):
        self.name = name
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.ret = False
        self.frame = None
        
        if not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara '{self.name}' en /dev/video{src}")
            self.stopped = True
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret, self.frame = self.cap.read()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.ret, self.frame = ret, frame
            time.sleep(0.005) 

    def read(self):
        if not self.ret or self.frame is None:
            return False, None
        return self.ret, self.frame.copy()

    def release(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

# -------------------------------------------------------------
# 2. Renderizado de Cubos 3D con Etiqueta de Nombre
# -------------------------------------------------------------
def draw_3d_cuboid(img, x1, y1, x2, y2, color=(0, 255, 0), scale=0.20, label=""):
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * scale), int(h * scale)
    
    # Vértices Cara Frontal
    f_tl, f_tr = (x1, y1), (x2, y1)
    f_bl, f_br = (x1, y2), (x2, y2)
    
    # Vértices Cara Trasera (Perspectiva 3D)
    b_tl, b_tr = (x1 + dx, y1 - dy), (x2 - dx, y1 - dy)
    b_bl, b_br = (x1 + dx, y2 - dy), (x2 - dx, y2 - dy)
    
    # Aristas traseras y de profundidad
    dark_color = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))
    cv2.rectangle(img, b_tl, b_br, dark_color, 1)
    
    for front_pt, back_pt in zip([f_tl, f_tr, f_bl, f_br], [b_tl, b_tr, b_bl, b_br]):
        cv2.line(img, front_pt, back_pt, color, 1, cv2.LINE_AA)
        
    # Cara Frontal
    cv2.rectangle(img, f_tl, f_br, color, 2, cv2.LINE_AA)

    # Renderizar nombre de la clase
    if label:
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        top_y = max(y1 - dy - 5, 15)
        
        cv2.rectangle(img, (x1, top_y - text_h - 4), (x1 + text_w + 6, top_y + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 3, top_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# -------------------------------------------------------------
# 3. Procesamiento de IA
# -------------------------------------------------------------
latest_front_objects = []
latest_rear_objects = []
is_running = True

def inference_worker(model, cam_front, cam_rear, target_classes):
    global latest_front_objects, latest_rear_objects, is_running
    
    while is_running:
        ret_f, frame_front = cam_front.read()
        ret_r, frame_rear = cam_rear.read()
        
        if ret_f and frame_front is not None:
            res_f = model(frame_front, imgsz=320, verbose=False)[0]
            temp_front = []
            for box in res_f.boxes:
                cls = int(box.cls[0])
                if cls in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    foot_x, foot_y = (x1 + x2) / 2, y2
                    rel_x = (foot_x - 320) * 0.5
                    rel_y = 70 + (480 - foot_y) * 0.75
                    temp_front.append((x1, y1, x2, y2, rel_x, rel_y, cls))
            latest_front_objects = temp_front

        if ret_r and frame_rear is not None:
            res_r = model(frame_rear, imgsz=320, verbose=False)[0]
            temp_rear = []
            for box in res_r.boxes:
                cls = int(box.cls[0])
                if cls in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    foot_x, foot_y = (x1 + x2) / 2, y2
                    rel_x = (320 - foot_x) * 0.5
                    rel_y = -70 - (480 - foot_y) * 0.75
                    temp_rear.append((x1, y1, x2, y2, rel_x, rel_y, cls))
            latest_rear_objects = temp_rear
            
        time.sleep(0.005)

# -------------------------------------------------------------
# 4. Hilo Principal
# -------------------------------------------------------------
def main():
    global is_running
    print("Iniciando Simulación ADAS con Nombres en 3D...")
    
    model = YOLO("yolov8n_openvino_model/")
    TARGET_CLASSES = list(CLASS_NAMES.keys())

    # AJUSTA AQUÍ LOS ÍNDICES SEGÚN TU SISTEMA (Nodos pares: 0, 2, 4...)
    cam_front = AsyncCamera(3, "Frontal")
    cam_rear = AsyncCamera(5, "Trasera")

    ai_thread = threading.Thread(
        target=inference_worker, 
        args=(model, cam_front, cam_rear, TARGET_CLASSES), 
        daemon=True
    )
    ai_thread.start()

    CANVAS_W, CANVAS_H = 1200, 700
    BEV_CENTER_X, BEV_CENTER_Y = 900, 350

    while True:
        ret_f, frame_front = cam_front.read()
        ret_r, frame_rear = cam_rear.read()

        # Generar cuadros vacíos si la cámara no está disponible para no colapsar el programa
        if not ret_f or frame_front is None:
            frame_front = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame_front, "CAMARA FRONTAL NO DISPONIBLE", (40, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if not ret_r or frame_rear is None:
            frame_rear = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame_rear, "CAMARA TRASERA NO DISPONIBLE", (40, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        dashboard = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        dashboard[:] = (20, 24, 33)

        # Proyección 3D Cámara Frontal
        for (x1, y1, x2, y2, rx, ry, cls) in latest_front_objects:
            name = CLASS_NAMES.get(cls, "Objeto")
            draw_3d_cuboid(frame_front, x1, y1, x2, y2, color=(0, 255, 0), label=name)
            
        # Proyección 3D Cámara Trasera
        for (x1, y1, x2, y2, rx, ry, cls) in latest_rear_objects:
            name = CLASS_NAMES.get(cls, "Objeto")
            draw_3d_cuboid(frame_rear, x1, y1, x2, y2, color=(0, 165, 255), label=name)

        # Paneles laterales
        frame_f_resized = cv2.resize(frame_front, (560, 320))
        frame_r_resized = cv2.resize(frame_rear, (560, 320))
        dashboard[20:340, 20:580] = frame_f_resized
        dashboard[360:680, 20:580] = frame_r_resized
        
        cv2.putText(dashboard, "CAMARA FRONTAL 3D", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(dashboard, "CAMARA TRASERA 3D", (30, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Radar Isométrico 3D
        cv2.rectangle(dashboard, (600, 20), (1180, 680), (30, 35, 48), -1)
        cv2.rectangle(dashboard, (600, 20), (1180, 680), (80, 90, 110), 2)
        
        for depth_y in range(60, 660, 80):
            cv2.line(dashboard, (620, depth_y), (1160, depth_y), (40, 48, 65), 1)
        cv2.line(dashboard, (BEV_CENTER_X - 60, 20), (BEV_CENTER_X - 60, 680), (60, 70, 90), 1)
        cv2.line(dashboard, (BEV_CENTER_X + 60, 20), (BEV_CENTER_X + 60, 680), (60, 70, 90), 1)

        # Vehículo Principal
        draw_3d_cuboid(dashboard, BEV_CENTER_X - 25, BEV_CENTER_Y - 45, 
                        BEV_CENTER_X + 25, BEV_CENTER_Y + 45, color=(255, 140, 0), scale=0.3, label="EGO CAR")

        # Proyectar Objetos en el Radar
        all_objects = latest_front_objects + latest_rear_objects
        for (x1, y1, x2, y2, rel_x, rel_y, cls) in all_objects:
            obj_x = int(BEV_CENTER_X + rel_x)
            obj_y = int(BEV_CENTER_Y - rel_y)
            if 620 < obj_x < 1160 and 40 < obj_y < 660:
                name = CLASS_NAMES.get(cls, "")
                draw_3d_cuboid(dashboard, obj_x - 18, obj_y - 25, 
                               obj_x + 18, obj_y + 25, color=(0, 0, 255), scale=0.25, label=name)

        cv2.imshow("BYD ADAS 3D Simulator", dashboard)
        
        if cv2.waitKey(1) & 0xFF == 27:
            is_running = False
            break

    cam_front.release()
    cam_rear.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()