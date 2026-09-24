import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

CLASS_REAL_WIDTHS = {
    0: 0.45, 2: 1.80, 3: 0.80, 5: 2.50, 7: 2.50, 15: 0.25, 16: 0.35, 39: 0.07
}

FOCAL_LENGTH = 700.0
CAMERA_HEIGHT = 0.50
HORIZON_Y = 240
STEP_M = 0.20
PIXELS_PER_METER = 225
BUMPER_OFFSET_M = 0.05

# Coherente con PUERTO_FRONTAL / PUERTO_TRASERA de Send_tripleVideo.py
PUERTO_FRONTAL = 0
PUERTO_TRASERA = 2

data_lock = threading.Lock()

def format_dist(dist_m):
    dist_cm = dist_m * 100
    return f"{int(dist_cm)}cm" if abs(dist_m) < 1.0 else f"{dist_m:.2f}m"

def get_color_by_distance(dist_m):
    if dist_m < 1.50:
        return (0, 0, 255)
    elif dist_m < 3.00:
        return (0, 255, 255)
    else:
        return (0, 255, 0)

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


# MEJORA: mismo estilo de "tablero de ajedrez" que byd_adas_ecal.py, para coherencia visual
def draw_metric_grid(dashboard, x1=610, y1=30, x2=1170, y2=670, center_x=890, center_y=350, step_px=45):
    color_a = (26, 30, 40)
    color_b = (18, 22, 30)

    col = 0
    for x in range(x1, x2, step_px):
        row = 0
        for y in range(y1, y2, step_px):
            idx_col = round((x - center_x) / step_px)
            idx_row = round((y - center_y) / step_px)
            color = color_a if (idx_col + idx_row) % 2 == 0 else color_b
            cv2.rectangle(dashboard, (x, y), (min(x + step_px, x2), min(y + step_px, y2)), color, -1)
            row += 1
        col += 1

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


latest_front_objects = []
latest_rear_objects = []
is_running = True

def process_detections(boxes, model_names, is_rear=False):
    objects = []
    for box in boxes:
        cls = int(box.cls[0])
        name = model_names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        foot_x, foot_y = (x1 + x2) / 2.0, y2
        w_px = max(x2 - x1, 1)
        h_px = max(y2 - y1, 1)

        if cls == 0:
            real_h = 1.70
            z_visual = (real_h * FOCAL_LENGTH) / h_px
        else:
            real_w = CLASS_REAL_WIDTHS.get(cls, 0.50)
            z_visual = (real_w * FOCAL_LENGTH) / w_px

        v = max(foot_y, HORIZON_Y + 1)
        z_ground = (CAMERA_HEIGHT * FOCAL_LENGTH) / (v - HORIZON_Y)

        z_m = z_visual if foot_y >= 450 else min(z_ground, z_visual)

        if w_px >= 580 or h_px >= 440:
            z_m = 0.15

        z_m = max(z_m, 0.05)

        if not is_rear:
            x_m = ((foot_x - 320.0) * z_m) / FOCAL_LENGTH
            rel_x = x_m * PIXELS_PER_METER
            rel_y = (z_m + BUMPER_OFFSET_M) * PIXELS_PER_METER
        else:
            x_m = ((320.0 - foot_x) * z_m) / FOCAL_LENGTH
            rel_x = x_m * PIXELS_PER_METER
            rel_y = -(z_m + BUMPER_OFFSET_M) * PIXELS_PER_METER

        dist_total_m = z_m + BUMPER_OFFSET_M
        objects.append((x1, y1, x2, y2, rel_x, rel_y, name, dist_total_m))
    return objects

def inference_worker(model, cam_front, cam_rear, model_names):
    global latest_front_objects, latest_rear_objects, is_running

    while is_running:
        ret_f, frame_front = cam_front.read()
        ret_r, frame_rear = cam_rear.read()

        objs_f, objs_r = [], []

        if ret_f and frame_front is not None:
            res_f = model(frame_front, imgsz=320, conf=0.35, verbose=False)[0]
            objs_f = process_detections(res_f.boxes, model_names, is_rear=False)

        if ret_r and frame_rear is not None:
            res_r = model(frame_rear, imgsz=320, conf=0.35, verbose=False)[0]
            objs_r = process_detections(res_r.boxes, model_names, is_rear=True)

        with data_lock:
            latest_front_objects = objs_f
            latest_rear_objects = objs_r

        time.sleep(0.005)

def main():
    global is_running
    print("Nivel 5 - Simulación standalone (sin eCAL) activa...")

    model = YOLO("yolov8n.pt")
    model(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)  # warm-up

    cam_front = AsyncCamera(PUERTO_FRONTAL, "Frontal")
    cam_rear = AsyncCamera(PUERTO_TRASERA, "Trasera")

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

        with data_lock:
            front_objs = list(latest_front_objects)
            rear_objs = list(latest_rear_objects)

        critical_proximity = False
        for (x1, y1, x2, y2, rx, ry, obj_name, dist_m) in front_objs:
            color = get_color_by_distance(dist_m)
            lbl = f"{obj_name.upper()} {format_dist(dist_m)}"
            draw_3d_cuboid(frame_front, x1, y1, x2, y2, color=color, label=lbl)
            if dist_m < 1.50:
                critical_proximity = True

        for (x1, y1, x2, y2, rx, ry, obj_name, dist_m) in rear_objs:
            color = get_color_by_distance(dist_m)
            lbl = f"{obj_name.upper()} {format_dist(dist_m)}"
            draw_3d_cuboid(frame_rear, x1, y1, x2, y2, color=color, label=lbl)

        dashboard[20:340, 20:580] = cv2.resize(frame_front, (560, 320))
        dashboard[360:680, 20:580] = cv2.resize(frame_rear, (560, 320))

        cv2.putText(dashboard, "CAMARA FRONTAL", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(dashboard, "CAMARA TRASERA", (30, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        draw_metric_grid(dashboard, x1=610, y1=30, x2=1170, y2=670,
                         center_x=BEV_CENTER_X, center_y=BEV_CENTER_Y, step_px=45)

        draw_3d_cuboid(dashboard, BEV_CENTER_X - 25, BEV_CENTER_Y - 45,
                       BEV_CENTER_X + 25, BEV_CENTER_Y + 45, color=(255, 140, 0), scale=0.3, label="EGO CAR")

        all_objects = front_objs + rear_objs
        for (x1, y1, x2, y2, rel_x, rel_y, obj_name, dist_m) in all_objects:
            obj_x = int(BEV_CENTER_X + rel_x)
            obj_y = int(BEV_CENTER_Y - rel_y)

            obj_x_clamped = max(615, min(1165, obj_x))
            obj_y_clamped = max(35, min(665, obj_y))

            lbl = f"{obj_name.upper()} {format_dist(dist_m)}"
            color = get_color_by_distance(dist_m)

            draw_3d_cuboid(dashboard, obj_x_clamped - 18, obj_y_clamped - 25,
                           obj_x_clamped + 18, obj_y_clamped + 25, color=color, scale=0.25, label=lbl)

        if critical_proximity:
            cv2.rectangle(dashboard, (620, 40), (1160, 85), (0, 0, 180), -1)
            cv2.putText(dashboard, "ALERTA: OBJETO CERCANO EN FRENTE", (640, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("BYD ADAS 3D Simulator (Standalone)", dashboard)

        if cv2.waitKey(1) & 0xFF == 27:
            is_running = False
            break

    cam_front.release()
    cam_rear.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()