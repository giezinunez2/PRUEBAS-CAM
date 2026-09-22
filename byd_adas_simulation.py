import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# Ancho físico estimado promedio en metros por clase conocida.
# Si el modelo detecta algo que no está aquí (ej. botella, silla, mochila), 
# el código le asignará automáticamente 0.50m (50 cm) por defecto.
CLASS_REAL_WIDTHS = {
    0: 0.45,  # Persona (usado solo de respaldo, priorizamos altura)
    2: 1.80,  # Auto
    3: 0.80,  # Moto
    5: 2.50,  # Autobús
    7: 2.50,  # Camión
    15: 0.25, # Gato
    16: 0.35, # Perro
    39: 0.07  # Botella
}

# -------------------------------------------------------------
# Configuración Geométrica y de Calibración
# -------------------------------------------------------------
FOCAL_LENGTH = 700.0     # ¡Ajusta este valor con la fórmula matemática real para tu cámara!
CAMERA_HEIGHT = 0.50     # Altura de la cámara respecto al suelo en metros
HORIZON_Y = 240          # Posición Y del horizonte en píxeles (mitad de 480)
STEP_M = 0.20            # Cada casilla del radar equivale a 20 cm
PIXELS_PER_METER = 225   # Escala visual del radar
BUMPER_OFFSET_M = 0.05   # Distancia lente-parachoques (5 cm)

CAM_FRONT_INDEX = 1
CAM_REAR_INDEX = 2

def format_dist(dist_m):
    """Convierte distancias a centímetros si es menor a 1 metro, o a metros si es mayor."""
    dist_cm = dist_m * 100
    if abs(dist_m) < 1.0:
        return f"{int(dist_cm)}cm"
    else:
        return f"{dist_m:.2f}m"

def get_color_by_distance(dist_m):
    """Retorna un color (BGR) basado en la distancia (Verde, Amarillo, Rojo)."""
    if dist_m < 1.50:
        return (0, 0, 255)    # ROJO (Peligro crítico < 1.5m)
    elif dist_m < 3.00:
        return (0, 255, 255)  # AMARILLO (Precaución < 3.0m)
    else:
        return (0, 255, 0)    # VERDE (Seguro > 3.0m)

# -------------------------------------------------------------
# 1. Lector Asíncrono de Cámara
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
            print(f"[ERROR] No se pudo abrir la cámara '{self.name}' en índice {src}")
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
# 2. Renderizado de Cubos 3D
# -------------------------------------------------------------
def draw_3d_cuboid(img, x1, y1, x2, y2, color=(0, 255, 0), scale=0.20, label=""):
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * scale), int(h * scale)
    
    f_tl, f_tr = (x1, y1), (x2, y1)
    f_bl, f_br = (x1, y2), (x2, y2)
    
    b_tl, b_tr = (x1 + dx, y1 - dy), (x2 - dx, y1 - dy)
    b_bl, b_br = (x1 + dx, y2 - dy), (x2 - dx, y2 - dy)
    
    dark_color = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))
    cv2.rectangle(img, b_tl, b_br, dark_color, 1)
    
    for front_pt, back_pt in zip([f_tl, f_tr, f_bl, f_br], [b_tl, b_tr, b_bl, b_br]):
        cv2.line(img, front_pt, back_pt, color, 1, cv2.LINE_AA)
        
    cv2.rectangle(img, f_tl, f_br, color, 2, cv2.LINE_AA)

    if label:
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_w, text_h = text_size
        top_y = max(y1 - dy - 5, 15)
        
        cv2.rectangle(img, (x1, top_y - text_h - 4), (x1 + text_w + 6, top_y + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 3, top_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# -------------------------------------------------------------
# 3. Malla Métrica con Escala Mixta (BEV Grid)
# -------------------------------------------------------------
def draw_metric_grid(dashboard, x1=610, y1=30, x2=1170, y2=670, center_x=890, center_y=350, step_px=45):
    cv2.rectangle(dashboard, (x1, y1), (x2, y2), (18, 22, 30), -1)
    
    for x in range(center_x, x2, step_px):
        cv2.line(dashboard, (x, y1), (x, y2), (40, 50, 68), 1, cv2.LINE_AA)
    for x in range(center_x, x1, -step_px):
        cv2.line(dashboard, (x, y1), (x, y2), (40, 50, 68), 1, cv2.LINE_AA)
        
    for y in range(center_y, y2, step_px):
        cv2.line(dashboard, (x1, y), (x2, y), (40, 50, 68), 1, cv2.LINE_AA)
    for y in range(center_y, y1, -step_px):
        cv2.line(dashboard, (x1, y), (x2, y), (40, 50, 68), 1, cv2.LINE_AA)

    cv2.line(dashboard, (center_x, y1), (center_x, y2), (0, 165, 255), 1, cv2.LINE_AA)
    cv2.line(dashboard, (x1, center_y), (x2, center_y), (0, 165, 255), 1, cv2.LINE_AA)

    for i, py in enumerate(range(center_y - step_px, y1 + 10, -step_px)):
        dist_m = (i + 1) * STEP_M
        tag = f"+{format_dist(dist_m)}"
        cv2.putText(dashboard, tag, (center_x + 6, py - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 180), 1, cv2.LINE_AA)
        cv2.circle(dashboard, (center_x, py), 2, (0, 255, 255), -1)

    for i, py in enumerate(range(center_y + step_px, y2 - 10, step_px)):
        dist_m = (i + 1) * STEP_M
        tag = f"-{format_dist(dist_m)}"
        cv2.putText(dashboard, tag, (center_x + 6, py + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 180), 1, cv2.LINE_AA)
        cv2.circle(dashboard, (center_x, py), 2, (0, 255, 255), -1)

    cv2.rectangle(dashboard, (x1, y1), (x2, y2), (70, 85, 110), 2)

# -------------------------------------------------------------
# 4. Procesamiento de IA con Geometría Híbrida
# -------------------------------------------------------------
latest_front_objects = []
latest_rear_objects = []
is_running = True

def process_detections(boxes, model_names, is_rear=False):
    objects = []
    for box in boxes:
        cls = int(box.cls[0])
        name = model_names[cls]  # Obtenemos el nombre directamente de YOLO
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        foot_x, foot_y = (x1 + x2) / 2.0, y2
        w_px = max(x2 - x1, 1)
        h_px = max(y2 - y1, 1)
        
        # 1. Estimación visual 
        if cls == 0: 
            real_h = 1.70  # Altura de persona
            z_visual = (real_h * FOCAL_LENGTH) / h_px
        else:
            # MAGIA: Si no sabemos qué es, asumimos un ancho de 50 cm
            real_w = CLASS_REAL_WIDTHS.get(cls, 0.50)
            z_visual = (real_w * FOCAL_LENGTH) / w_px
        
        # 2. Estimación por plano del suelo
        v = max(foot_y, HORIZON_Y + 1)
        z_ground = (CAMERA_HEIGHT * FOCAL_LENGTH) / (v - HORIZON_Y)
        
        # Algoritmo Híbrido actualizado
        if foot_y >= 450:
            z_m = z_visual
        else:
            z_m = min(z_ground, z_visual)
            
        # --- PARCHE DE SATURACIÓN DE CÁMARA (TRUNCAMIENTO) ---
        if w_px >= 580 or h_px >= 440:
            z_m = 0.15  # Forzar colisión inminente si bloquea toda la pantalla
            
        # Limitar distancia mínima física detectable a 0.05m (5 cm)
        z_m = max(z_m, 0.05)
        
        # Desplazamiento lateral X
        if not is_rear:
            x_m = ((foot_x - 320.0) * z_m) / FOCAL_LENGTH
            rel_x = x_m * PIXELS_PER_METER
            rel_y = (z_m + BUMPER_OFFSET_M) * PIXELS_PER_METER
        else:
            x_m = ((320.0 - foot_x) * z_m) / FOCAL_LENGTH
            rel_x = x_m * PIXELS_PER_METER
            rel_y = -(z_m + BUMPER_OFFSET_M) * PIXELS_PER_METER
            
        dist_total_m = z_m + BUMPER_OFFSET_M
        
        # Guardamos la variable 'name' en lugar de 'cls'
        objects.append((x1, y1, x2, y2, rel_x, rel_y, name, dist_total_m))
    return objects

def inference_worker(model, cam_front, cam_rear, model_names):
    global latest_front_objects, latest_rear_objects, is_running
    
    while is_running:
        ret_f, frame_front = cam_front.read()
        ret_r, frame_rear = cam_rear.read()
        
        if ret_f and frame_front is not None:
            res_f = model(frame_front, imgsz=320, conf=0.25, verbose=False)[0]
            latest_front_objects = process_detections(res_f.boxes, model_names, is_rear=False)

        if ret_r and frame_rear is not None:
            res_r = model(frame_rear, imgsz=320, conf=0.25, verbose=False)[0]
            latest_rear_objects = process_detections(res_r.boxes, model_names, is_rear=True)
            
        time.sleep(0.005)

# -------------------------------------------------------------
# 5. Bucle Principal y Renderizado
# -------------------------------------------------------------
def main():
    global is_running
    print("Iniciando Sistema ADAS Universal (Detección Multi-Objeto)...")
    
    model = YOLO("yolov8n_openvino_model/") # Cambiar a "yolov8n.pt" si no usas OpenVINO
    
    cam_front = AsyncCamera(CAM_FRONT_INDEX, "Frontal")
    cam_rear = AsyncCamera(CAM_REAR_INDEX, "Trasera")

    # Pasamos model.names para que reconozca los 80 objetos automáticamente
    ai_thread = threading.Thread(
        target=inference_worker, 
        args=(model, cam_front, cam_rear, model.names), 
        daemon=True
    )
    ai_thread.start()

    CANVAS_W, CANVAS_H = 1200, 700
    BEV_CENTER_X, BEV_CENTER_Y = 890, 350

    while True:
        ret_f, frame_front = cam_front.read()
        ret_r, frame_rear = cam_rear.read()

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

        # Renderizar Objetos Cámara Frontal
        critical_proximity = False
        for (x1, y1, x2, y2, rx, ry, obj_name, dist_m) in latest_front_objects:
            color = get_color_by_distance(dist_m)
            lbl = f"{obj_name.upper()} {format_dist(dist_m)}" # Traduce texto a mayúsculas
            draw_3d_cuboid(frame_front, x1, y1, x2, y2, color=color, label=lbl)
            
            if dist_m < 1.50:
                critical_proximity = True
            
        # Renderizar Objetos Cámara Trasera
        for (x1, y1, x2, y2, rx, ry, obj_name, dist_m) in latest_rear_objects:
            color = get_color_by_distance(dist_m)
            lbl = f"{obj_name.upper()} {format_dist(dist_m)}"
            draw_3d_cuboid(frame_rear, x1, y1, x2, y2, color=color, label=lbl)

        # Integración de Paneles al Dashboard
        dashboard[20:340, 20:580] = cv2.resize(frame_front, (560, 320))
        dashboard[360:680, 20:580] = cv2.resize(frame_rear, (560, 320))
        
        cv2.putText(dashboard, "CAMARA FRONTAL", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(dashboard, "CAMARA TRASERA", (30, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Malla Radar BEV
        draw_metric_grid(dashboard, x1=610, y1=30, x2=1170, y2=670, 
                         center_x=BEV_CENTER_X, center_y=BEV_CENTER_Y, step_px=45)

        # Vehículo Principal (EGO CAR)
        draw_3d_cuboid(dashboard, BEV_CENTER_X - 25, BEV_CENTER_Y - 45, 
                       BEV_CENTER_X + 25, BEV_CENTER_Y + 45, color=(255, 140, 0), scale=0.3, label="EGO CAR")

        # Proyección en Plano Top-Down BEV
        all_objects = latest_front_objects + latest_rear_objects
        for (x1, y1, x2, y2, rel_x, rel_y, obj_name, dist_m) in all_objects:
            obj_x = int(BEV_CENTER_X + rel_x)
            obj_y = int(BEV_CENTER_Y - rel_y)
            
            obj_x_clamped = max(615, min(1165, obj_x))
            obj_y_clamped = max(35, min(665, obj_y))
            
            lbl = f"{obj_name.upper()} {format_dist(dist_m)}"
            color = get_color_by_distance(dist_m)
            
            draw_3d_cuboid(dashboard, obj_x_clamped - 18, obj_y_clamped - 25, 
                           obj_x_clamped + 18, obj_y_clamped + 25, color=color, scale=0.25, label=lbl)

        # Banner de Advertencia de Cercanía Crítica
        if critical_proximity:
            cv2.rectangle(dashboard, (620, 40), (1160, 85), (0, 0, 180), -1)
            cv2.putText(dashboard, "ALERTA: OBJETO CERCANO EN FRENTE", (640, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("BYD ADAS 3D Simulator", dashboard)
        
        if cv2.waitKey(1) & 0xFF == 27: # Presiona ESC en la ventana para salir de forma segura
            is_running = False
            break

    cam_front.release()
    cam_rear.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()