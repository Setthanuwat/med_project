import cv2
import numpy as np
import serial
import time
from datetime import datetime
import threading
import queue
import os
from ultralytics import YOLO

# ========================= 
# VIDEO SOURCE CONFIG
# ========================= 
USE_VIDEO_FILE = True  # เปลี่ยนเป็น True เพื่อใช้ไฟล์วิดีโอ, False เพื่อใช้กล้อง
VIDEO_FILE_PATH = "WIN_20260112_11_17_22_Pro.mp4"  # ระบุ path ของไฟล์วิดีโอที่ต้องการทดสอบ
CAMERA_INDEX = 1  # Index ของกล้อง (ใช้เมื่อ USE_VIDEO_FILE = False)

# ========================= 
# ESP32 CONFIG
# ========================= 
ESP32_PORT = 'COM3'
ESP32_BAUDRATE = 115200
ENABLE_ESP32 = False  # เปลี่ยนเป็น False เมื่อทดสอบกับวิดีโอ (ไม่ต้องการ ESP32)

# ========================= 
# VIDEO & SNAPSHOT CONFIG
# ========================= 
VIDEO_FOLDER = "D:/med_project/recordings/"
SNAPSHOT_FOLDER = "D:/med_project/snapshots/"
VIDEO_FPS = 30
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# ========================= 
# LEVEL CONFIG
# ========================= 
LEVEL_THICKNESS = 2
LEVEL_COUNT = 6  # มี 5 เส้น = 6 ช่อง (0-5)

# กำหนดตำแหน่ง Y ของเส้นแต่ละเส้น (จากบนลงล่าง)
LEVEL_Y_POSITIONS = [300, 450, 600, 750, 900]

# กำหนดสีแดงต่างกันสำหรับแต่ละ Level (BGR format)
LEVEL_COLORS = {
    5: (0, 255, 0),      # L5 - เขียว
    4: (255, 100, 240),  # L4 - ม่วง
    3: (0, 255, 255),    # L3 - เหลือง
    2: (255, 255, 0),    # L2 - ฟ้า
    1: (0, 0, 255),      # L1 - แดง
}

LEVEL_COUNT_COLOR = (0, 255, 255)  # สีเหลืองสำหรับตัวเลข

# =========================
# LEVEL SCORE CONFIG
# =========================
LEVEL_SCORES = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}

# =========================
# TOTAL FLY COUNT INPUT (สำหรับการคำนวณแมลงที่มองไม่เห็น)
# =========================
TOTAL_FLIES_PER_TUBE = [
    0,  # Tube 1
    0,  # Tube 2
    0,  # Tube 3
    0,  # Tube 4
    0   # Tube 5
]

# =========================
# TUBE CONFIGS
# =========================
TUBE_CONFIGS = [
    {'offset_from_left': 420,  'width': 180, 'top_offset': 40,  'bottom_offset': 0},
    {'offset_from_left': 620,  'width': 200, 'top_offset': 40, 'bottom_offset': 0},
    {'offset_from_left': 850,  'width': 190, 'top_offset': 40, 'bottom_offset': 0},
    {'offset_from_left': 1110, 'width': 185, 'top_offset': 40, 'bottom_offset': 0},
    {'offset_from_left': 1350, 'width': 195, 'top_offset': 40, 'bottom_offset': 0},
]
# สีสำหรับ tube borders (BGR format)
colors = [
    (0, 0, 0),  # Tube 1
    (0, 0, 0),  # Tube 2
    (0, 0, 0),  # Tube 3
    (0, 0, 0),  # Tube 4
    (0, 0, 0),  # Tube 5
]

# =========================
# SIDEBAR CONFIG
# =========================
SIDEBAR_WIDTH = 400
SIDEBAR_COLOR = (210, 210, 210)  # สีเทาเข้ม

# =========================
# BUTTON CONFIG
# =========================
BUTTON_WIDTH = 340
BUTTON_HEIGHT = 60
BUTTON_MARGIN = 20
BUTTON_START_Y = 700

# =========================
# YOLO CONFIG
# =========================
YOLO_MODEL_PATH = 'yolo26n_v2.pt'
YOLO_CONF_THRESHOLD = 0.65
YOLO_INPUT_SIZE = 640
OVERLAP_PERCENT = 0.5

# =========================
# LOAD YOLO MODEL
# =========================
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✓ Loaded YOLO model from: {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    print("  จะใช้เฉพาะ OpenCV Threshold method")
    yolo_model = None

# ========================= 
# GLOBAL VARIABLES
# ========================= 
is_running = False
emergency_stop = False
waiting_for_5sec_capture = False
capture_5sec_time = 0
WAIT_DURATION = 5

# ========================= 
# ESP32 SERIAL HANDLER
# ========================= 
def send_esp32_command(ser, command):
    """ส่งคำสั่งไปยัง ESP32"""
    if ser and ENABLE_ESP32:
        try:
            ser.write(f'{command}\n'.encode())
            print(f">>> ส่งคำสั่ง '{command}' ไปยัง ESP32")
            return True
        except Exception as e:
            print(f"Error sending command to ESP32: {e}")
            return False
    else:
        print(f">>> [จำลอง] ส่งคำสั่ง '{command}' ไปยัง ESP32")
        return True

# =========================
# YOLO HELPER FUNCTIONS
# =========================
def create_level_crops(tube_roi, tube_y_start, x_start):
    """แบ่ง tube เป็น crops ตาม levels โดยมี overlap"""
    h, w = tube_roi.shape[:2]
    
    if h == 0 or w == 0:
        return []
    
    LEVEL_Y_DRAW = [y - 50 for y in LEVEL_Y_POSITIONS]
    
    LEVEL_Y_IN_TUBE = []
    for y in LEVEL_Y_DRAW:
        y_relative = y - tube_y_start
        if 0 < y_relative < h:
            LEVEL_Y_IN_TUBE.append(int(y_relative))
    
    if len(LEVEL_Y_IN_TUBE) == 0:
        crop_img = tube_roi.copy()
        if crop_img.size > 0:
            return [{
                'img': crop_img,
                'y_start': 0,
                'y_end': h,
                'level': 0,
                'original_y_start': 0,
                'original_y_end': h
            }]
        else:
            return []
    
    boundaries = [0] + LEVEL_Y_IN_TUBE + [h]
    
    crops = []
    
    for i in range(len(boundaries) - 1):
        y_start = int(boundaries[i])
        y_end = int(boundaries[i + 1])
        
        if y_end <= y_start:
            continue
        
        crop_height = y_end - y_start
        overlap_pixels = int(crop_height * OVERLAP_PERCENT)
        
        crop_y_start = max(0, y_start - overlap_pixels)
        crop_y_end = min(h, y_end + overlap_pixels)
        
        if crop_y_end <= crop_y_start:
            continue
        
        if (crop_y_end - crop_y_start) < 10:
            continue
        
        crop_img = tube_roi[crop_y_start:crop_y_end, :].copy()
        
        if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            continue
        
        level_num = len(boundaries) - 2 - i
        
        crops.append({
            'img': crop_img,
            'y_start': crop_y_start,
            'y_end': crop_y_end,
            'level': level_num,
            'original_y_start': y_start,
            'original_y_end': y_end
        })
    
    return crops

def resize_to_yolo_input(img, target_size=640):
    """Resize image to YOLO input size with padding"""
    if img is None or img.size == 0:
        return None, None, None, None
    
    h, w = img.shape[:2]
    
    if h == 0 or w == 0:
        return None, None, None, None
    
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w == 0 or new_h == 0:
        return None, None, None, None
    
    resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas, scale, x_offset, y_offset

def detect_flies_yolo(crop_info, conf_threshold=0.25):
    """ใช้ YOLO detect แมลงวันใน crop"""
    if yolo_model is None:
        return []
    
    crop_img = crop_info['img']
    
    if crop_img is None or crop_img.size == 0:
        return []
    
    result = resize_to_yolo_input(crop_img, YOLO_INPUT_SIZE)
    
    if result[0] is None:
        return []
    
    yolo_input, scale, x_offset, y_offset = result
    
    results = yolo_model(yolo_input, conf=conf_threshold, verbose=False)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            x1_orig = int((x1 - x_offset) / scale)
            y1_orig = int((y1 - y_offset) / scale)
            x2_orig = int((x2 - x_offset) / scale)
            y2_orig = int((y2 - y_offset) / scale)
            
            if x1_orig < 0 or y1_orig < 0:
                continue
            
            w = x2_orig - x1_orig
            h = y2_orig - y1_orig
            
            detections.append({
                'x': x1_orig,
                'y': y1_orig,
                'w': w,
                'h': h,
                'conf': conf
            })
    
    return detections

def remove_duplicates(all_detections, iou_threshold=0.5):
    """ลบ detection ที่ซ้ำกันด้วย NMS"""
    if len(all_detections) == 0:
        return []
    
    boxes = []
    scores = []
    
    for det in all_detections:
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        boxes.append([x, y, x+w, y+h])
        scores.append(det['conf'])
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )
    
    filtered = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered.append(all_detections[i])
    
    return filtered

# ========================= 
# PROCESS FRAME FUNCTION (OpenCV Only)
# ========================= 
def process_frame(img):
    """ประมวลผลภาพด้วย OpenCV เท่านั้น"""
    orig = img.copy()
    h_img, w_img = img.shape[:2]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel_white = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_white)
    
    roi_y_start = 0
    roi_y_end = 1000
    
    tube_positions = []
    for i, cfg in enumerate(TUBE_CONFIGS):
        x_start = cfg['offset_from_left']
        x_end = x_start + cfg['width']
        tube_positions.append({
            'tube_num': i + 1,
            'x_start': x_start,
            'x_end': x_end
        })
    
    fly_counts = []
    tube_level_results = []
    tube_level_scores = []
    flies_above_L1 = 0
    
    LEVEL_Y_DRAW = [y - 50 for y in LEVEL_Y_POSITIONS]
    
    for i, tube in enumerate(tube_positions):
        cfg = TUBE_CONFIGS[i]
        x_start = tube['x_start']
        x_end = tube['x_end']
        tube_y_start = roi_y_start + cfg['top_offset']
        tube_y_end = roi_y_end - cfg['bottom_offset']
        
        level_counts = [0] * LEVEL_COUNT
        tube_fly_count = 0
        
        cv2.rectangle(orig, (x_start, tube_y_start), (x_end, tube_y_end), colors[i], 2)
        cv2.putText(orig, f"Tube {i+1}", (x_start + 50, tube_y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
        
        tube_roi = img[tube_y_start:tube_y_end, x_start:x_end]
        
        if tube_roi.size > 0:
            gray_roi = cv2.cvtColor(tube_roi, cv2.COLOR_BGR2GRAY)
            blur_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
            th_roi = cv2.adaptiveThreshold(
                blur_roi, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 17, 7
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            th_clean = cv2.morphologyEx(th_roi, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours_fly, _ = cv2.findContours(
                th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2:]
            
            for cnt in contours_fly:
                area = cv2.contourArea(cnt)
                if 20 < area < 200:
                    tube_fly_count += 1
                    fx, fy, fw, fh = cv2.boundingRect(cnt)
                    actual_x = x_start + fx
                    actual_y = tube_y_start + fy
                    
                    cv2.rectangle(orig, (actual_x, actual_y),
                                (actual_x + fw, actual_y + fh), (0, 0, 255), 1)
                    
                    cy = actual_y + fh // 2
                    
                    if cy < LEVEL_Y_DRAW[4]:
                        flies_above_L1 += 1
                    
                    assigned = False
                    for lv_idx in range(len(LEVEL_Y_DRAW)):
                        if cy < LEVEL_Y_DRAW[lv_idx]:
                            level_counts[5 - lv_idx] += 1
                            assigned = True
                            break
                    
                    if not assigned:
                        level_counts[0] += 1
        
        total_expected = TOTAL_FLIES_PER_TUBE[i] if i < len(TOTAL_FLIES_PER_TUBE) else 0
        if total_expected > 0 and tube_fly_count < total_expected:
            unseen_flies = total_expected - tube_fly_count
            level_counts[5] += unseen_flies
        
        for lv_idx, y_draw in enumerate(LEVEL_Y_DRAW):
            display_level = 5 - lv_idx
            cv2.line(orig, (x_start, y_draw), (x_end, y_draw),
                    LEVEL_COLORS[display_level], LEVEL_THICKNESS)
            
            if i == 4:
                label_text = f"L{display_level}"
                label_x = x_end + 15
                label_y = y_draw - 55
                
                cv2.putText(orig, label_text, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           LEVEL_COLORS[display_level], 3)
        
        fly_counts.append(total_expected if total_expected > 0 else tube_fly_count)
        tube_level_results.append(level_counts)
        
        score_per_level = []
        for lv in range(LEVEL_COUNT):
            score_per_level.append(level_counts[lv] * LEVEL_SCORES[lv])
        tube_level_scores.append(score_per_level)
    
    total_flies = sum(fly_counts)
    all_flies_below_L1 = (total_flies > 0) and (flies_above_L1 == 0)
    
    return orig, all_flies_below_L1, total_flies, fly_counts, tube_level_results, tube_level_scores

# =========================
# PROCESS FRAME WITH YOLO
# =========================
def process_frame_with_yolo(img):
    """ประมวลผลภาพด้วยทั้ง OpenCV และ YOLO"""
    orig = img.copy()
    h_img, w_img = img.shape[:2]
    
    print("\n" + "="*60)
    print("PROCESSING WITH DUAL ALGORITHM (OpenCV + YOLO)")
    print("="*60)
    
    roi_y_start = 0
    roi_y_end = 1000
    
    tube_positions = []
    for i, cfg in enumerate(TUBE_CONFIGS):
        x_start = cfg['offset_from_left']
        x_end = x_start + cfg['width']
        tube_positions.append({
            'tube_num': i + 1,
            'x_start': x_start,
            'x_end': x_end
        })
    
    opencv_results = []
    yolo_results = []
    combined_results = []
    
    fly_counts = []
    tube_level_results = []
    tube_level_scores = []
    flies_above_L1 = 0
    
    LEVEL_Y_DRAW = [y - 50 for y in LEVEL_Y_POSITIONS]
    
    for i, tube in enumerate(tube_positions):
        cfg = TUBE_CONFIGS[i]
        x_start = tube['x_start']
        x_end = tube['x_end']
        tube_y_start = roi_y_start + cfg['top_offset']
        tube_y_end = roi_y_end - cfg['bottom_offset']
        
        print(f"\n--- Tube {i+1} ---")
        
        cv2.rectangle(orig, (x_start, tube_y_start), (x_end, tube_y_end), colors[i], 2)
        cv2.putText(orig, f"Tube {i+1}", (x_start + 50, tube_y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
        
        tube_roi = img[tube_y_start:tube_y_end, x_start:x_end]
        
        # OpenCV
        opencv_detections = []
        if tube_roi.size > 0:
            gray_roi = cv2.cvtColor(tube_roi, cv2.COLOR_BGR2GRAY)
            blur_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
            th_roi = cv2.adaptiveThreshold(
                blur_roi, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 17, 7
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            th_clean = cv2.morphologyEx(th_roi, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours_fly, _ = cv2.findContours(
                th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2:]
            
            for cnt in contours_fly:
                area = cv2.contourArea(cnt)
                if 20 < area < 200:
                    fx, fy, fw, fh = cv2.boundingRect(cnt)
                    opencv_detections.append({
                        'x': x_start + fx,
                        'y': tube_y_start + fy,
                        'w': fw,
                        'h': fh,
                        'method': 'opencv'
                    })
        
        print(f"  OpenCV detected: {len(opencv_detections)} flies")
        
        # YOLO
        yolo_detections = []
        if yolo_model is not None and tube_roi.size > 0:
            crops = create_level_crops(tube_roi, tube_y_start, x_start)
            
            if len(crops) == 0:
                print(f"  Warning: No valid crops created for Tube {i+1}")
            else:
                all_tube_detections = []
                
                for crop_info in crops:
                    if crop_info['img'] is None or crop_info['img'].size == 0:
                        continue
                    
                    detections = detect_flies_yolo(crop_info, YOLO_CONF_THRESHOLD)
                    
                    for det in detections:
                        det['y'] += crop_info['y_start']
                        det['tube_y'] = det['y'] + tube_y_start
                        det['tube_x'] = det['x'] + x_start
                    
                    all_tube_detections.extend(detections)
                
                unique_detections = remove_duplicates(all_tube_detections, iou_threshold=0.4)
                
                for det in unique_detections:
                    yolo_detections.append({
                        'x': int(det['tube_x']),
                        'y': int(det['tube_y']),
                        'w': det['w'],
                        'h': det['h'],
                        'conf': det['conf'],
                        'method': 'yolo'
                    })
            
            print(f"  YOLO detected: {len(yolo_detections)} flies")
        
        # COMBINE
        all_detections = opencv_detections + yolo_detections
        
        level_counts = [0] * LEVEL_COUNT
        
        for det in all_detections:
            color = (0, 0, 255)
            thickness = 1
            
            cv2.rectangle(orig, (det['x'], det['y']),
                        (det['x'] + det['w'], det['y'] + det['h']), color, thickness)
            
            cy = det['y'] + det['h'] // 2
            
            if cy < LEVEL_Y_DRAW[4]:
                flies_above_L1 += 1
            
            assigned = False
            for lv_idx in range(len(LEVEL_Y_DRAW)):
                if cy < LEVEL_Y_DRAW[lv_idx]:
                    level_counts[5 - lv_idx] += 1
                    assigned = True
                    break
            
            if not assigned:
                level_counts[0] += 1
        
        tube_fly_count = len(all_detections)
        print(f"  Combined total: {tube_fly_count} flies")
        
        total_expected = TOTAL_FLIES_PER_TUBE[i] if i < len(TOTAL_FLIES_PER_TUBE) else 0
        if total_expected > 0 and tube_fly_count < total_expected:
            unseen_flies = total_expected - tube_fly_count
            level_counts[5] += unseen_flies
            print(f"  Added {unseen_flies} unseen flies to L5")
        
        for lv_idx, y_draw in enumerate(LEVEL_Y_DRAW):
            display_level = 5 - lv_idx
            cv2.line(orig, (x_start, y_draw), (x_end, y_draw),
                    LEVEL_COLORS[display_level], LEVEL_THICKNESS)
            
            if i == 4:
                label_text = f"L{display_level}"
                label_x = x_end + 15
                label_y = y_draw - 55
                cv2.putText(orig, label_text, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           LEVEL_COLORS[display_level], 3)
        
        fly_counts.append(total_expected if total_expected > 0 else tube_fly_count)
        tube_level_results.append(level_counts)
        
        score_per_level = []
        for lv in range(LEVEL_COUNT):
            score_per_level.append(level_counts[lv] * LEVEL_SCORES[lv])
        tube_level_scores.append(score_per_level)
        
        opencv_results.append(len(opencv_detections))
        yolo_results.append(len(yolo_detections))
        combined_results.append(tube_fly_count)
    
    total_flies = sum(fly_counts)
    all_flies_below_L1 = (total_flies > 0) and (flies_above_L1 == 0)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"OpenCV detections per tube: {opencv_results}")
    print(f"YOLO detections per tube:   {yolo_results}")
    print(f"Combined total per tube:    {combined_results}")
    print(f"Grand total: {total_flies} flies")
    print("="*60 + "\n")
    
    return orig, all_flies_below_L1, total_flies, fly_counts, tube_level_results, tube_level_scores

# =========================
# CREATE GUI WITH SIDEBAR
# =========================
def create_gui_frame(orig, fly_counts, tube_level_results, tube_level_scores):
    """สร้าง GUI พร้อม Sidebar"""
    h_orig, w_orig = orig.shape[:2]
    
    min_x = min([cfg['offset_from_left'] for cfg in TUBE_CONFIGS]) - 50
    max_x = max([cfg['offset_from_left'] + cfg['width'] for cfg in TUBE_CONFIGS]) + 100
    roi_y_start = 0
    roi_y_end = 1000
    
    cropped_orig = orig[roi_y_start:roi_y_end, min_x:max_x]
    h_crop, w_crop = cropped_orig.shape[:2]
    
    new_img = np.ones((h_crop, w_crop + SIDEBAR_WIDTH, 3), dtype=np.uint8) * 255
    
    new_img[:, :SIDEBAR_WIDTH] = SIDEBAR_COLOR
    
    new_img[:h_crop, SIDEBAR_WIDTH:SIDEBAR_WIDTH+w_crop] = cropped_orig
    
    # TABLE
    table_x = 20
    table_y = 140
    row_h = 35
    left_w = 60
    col_w = 60
    font_scale = 0.65
    thick = 2
    header_gap = 5

    levels = ['L5','L4','L3','L2','L1','L0']
    tubes = [f'T{i+1}' for i in range(len(tube_level_results))]

    level_totals = []
    for c in range(len(tubes)):
        total = sum(tube_level_results[c])
        level_totals.append(total)

    table_w = left_w + col_w * len(tubes)
    table_h = row_h * (len(levels) + 2) + header_gap
    
    # Header box
    header_box_y = 30
    header_box_h = 80
    cv2.rectangle(new_img, (table_x, header_box_y+30), 
                  (table_x + table_w-140, header_box_y + header_box_h+30),
                  (0, 0, 0), -1)
    cv2.rectangle(new_img, 
                  (table_x, header_box_y+30), 
                  (table_x + table_w-140, header_box_y + header_box_h+30),
                  (255, 255, 255), 2)
    cv2.putText(new_img, "Drosophila No.", (table_x + 13, header_box_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Table background
    cv2.rectangle(new_img, (table_x, table_y-row_h),
                  (table_x+table_w, table_y+table_h-25),
                  (30, 30, 30), -1)
    cv2.rectangle(new_img, (table_x, table_y-row_h),
                  (table_x+table_w, table_y+table_h-25),
                  (200, 200, 200), 2)

    cv2.line(new_img, (table_x, table_y + header_gap), 
             (table_x+table_w, table_y + header_gap), 
             (200,200,200), 2)

    for r in range(len(levels)):
        y_line = table_y + header_gap + (r+1)*row_h
        cv2.line(new_img, (table_x, y_line), 
                 (table_x+table_w, y_line), 
                 (100,100,100), 1)

    total_line_y = table_y + header_gap + (len(levels)+1)*row_h
    cv2.line(new_img, (table_x, total_line_y-row_h), 
             (table_x+table_w, total_line_y-row_h), 
             (200,200,200), 2)

    cv2.line(new_img, (table_x+left_w, table_y-row_h), 
             (table_x+left_w, table_y+table_h-25), 
             (200,200,200), 2)

    for c in range(1, len(tubes)):
        x_line = table_x + left_w + c*col_w
        cv2.line(new_img, (x_line, table_y-row_h), 
                 (x_line, table_y+table_h-25), 
                 (100,100,100), 1)

    # Header row
    header_y = table_y - 10
    cv2.putText(new_img, "Tube", (table_x + 5, header_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for i, t in enumerate(tubes):
        text_x = table_x + left_w + i*col_w + 15
        cv2.putText(new_img, t, (text_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    # Data rows
    for r, lvl in enumerate(levels):
        y = table_y + header_gap + (r+1)*row_h - 10
        
        level_index = 5 - r
        if level_index in LEVEL_COLORS:
            label_color = LEVEL_COLORS[level_index]
        else:
            label_color = (255, 255, 255)
        
        cv2.putText(new_img, lvl, (table_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thick)
        
        for c in range(len(tubes)):
            val = tube_level_results[c][level_index]
            text_x = table_x + left_w + c*col_w + 20
            cv2.putText(new_img, str(val), (text_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thick)

    # Total row
    total_y = table_y + header_gap + (len(levels)+1)*row_h - 10
    cv2.putText(new_img, "Total", (table_x + 5, total_y+5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for c in range(len(tubes)):
        cv2.putText(new_img, str(level_totals[c]),
                    (table_x + left_w + c*col_w + 20, total_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    # SCORE TABLE
    score_totals = []
    for c in range(len(tubes)):
        total = sum(tube_level_scores[c])
        score_totals.append(total)

    score_table_y = table_y + table_h + 100

    score_header_y = score_table_y - 110
    cv2.rectangle(new_img, (table_x, score_header_y+150), 
                  (table_x + table_w-260, score_header_y + 30),
                  (0, 0, 0), -1)
    cv2.rectangle(new_img, 
                  (table_x, score_header_y+150), 
                  (table_x + table_w-260, score_header_y + 30),
                  (255, 255, 255), 2)
    cv2.putText(new_img, "Score", (table_x +13, score_header_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.rectangle(new_img, (table_x, score_table_y-row_h),
                  (table_x+table_w, score_table_y+table_h-25),
                  (30, 30, 30), -1)
    cv2.rectangle(new_img, (table_x, score_table_y-row_h),
                  (table_x+table_w, score_table_y+table_h-25),
                  (200, 200, 200), 2)

    cv2.line(new_img, (table_x, score_table_y + header_gap), 
             (table_x+table_w, score_table_y + header_gap), 
             (200,200,200), 2)

    for r in range(len(levels)):
        y_line = score_table_y + header_gap + (r+1)*row_h
        cv2.line(new_img, (table_x, y_line), 
                 (table_x+table_w, y_line), 
                 (100,100,100), 1)

    score_total_line_y = score_table_y + header_gap + (len(levels)+1)*row_h
    cv2.line(new_img, (table_x, score_total_line_y-row_h), 
             (table_x+table_w, score_total_line_y-row_h), 
             (200,200,200), 2)

    cv2.line(new_img, (table_x+left_w, score_table_y-row_h), 
             (table_x+left_w, score_table_y+table_h-25), 
             (200,200,200), 2)

    for c in range(1, len(tubes)):
        x_line = table_x + left_w + c*col_w
        cv2.line(new_img, (x_line, score_table_y-row_h), 
                 (x_line, score_table_y+table_h-25), 
                 (100,100,100), 1)
        
    header_y = score_table_y - 10
    cv2.putText(new_img, "Tube", (table_x + 5, header_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for i, t in enumerate(tubes):
        cv2.putText(new_img, t,
                    (table_x + left_w + i*col_w + 15, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for r, lvl in enumerate(levels):
        y = score_table_y + header_gap + (r+1)*row_h - 10
    
        level_index = 5 - r
        if level_index in LEVEL_COLORS:
            label_color = LEVEL_COLORS[level_index]
        else:
            label_color = (255, 255, 255)
        
        cv2.putText(new_img, lvl, (table_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thick)
        
        for c in range(len(tubes)):
            score_val = tube_level_scores[c][5-r]
            cv2.putText(new_img, str(score_val),
                        (table_x + left_w + c*col_w + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thick)

    score_total_y = score_table_y + header_gap + (len(levels)+1)*row_h - 10
    cv2.putText(new_img, "Total", (table_x + 5, score_total_y+5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for c in range(len(tubes)):
        cv2.putText(new_img, str(score_totals[c]),
                    (table_x + left_w + c*col_w + 20, score_total_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)
    
    return new_img

# =========================
# DRAW BUTTONS ON SIDEBAR
# =========================
def draw_buttons(sidebar_img, is_running):
    """วาดปุ่ม Start และ Emergency Stop บน sidebar"""
    button_x = BUTTON_MARGIN
    
    # Start Button
    start_y = BUTTON_START_Y
    start_color = (100, 200, 100) if not is_running else (150, 150, 150)
    cv2.rectangle(sidebar_img, 
                  (button_x, start_y), 
                  (button_x + BUTTON_WIDTH, start_y + BUTTON_HEIGHT),
                  start_color, -1)
    cv2.rectangle(sidebar_img, 
                  (button_x, start_y), 
                  (button_x + BUTTON_WIDTH, start_y + BUTTON_HEIGHT),
                  (0, 0, 0), 3)
    
    text = "START" if not is_running else "RUNNING..."
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = button_x + (BUTTON_WIDTH - text_size[0]) // 2
    text_y = start_y + (BUTTON_HEIGHT + text_size[1]) // 2
    cv2.putText(sidebar_img, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Emergency Stop Button
    stop_y = start_y + BUTTON_HEIGHT + BUTTON_MARGIN
    stop_color = (50, 50, 200)
    cv2.rectangle(sidebar_img, 
                  (button_x, stop_y), 
                  (button_x + BUTTON_WIDTH, stop_y + BUTTON_HEIGHT),
                  stop_color, -1)
    cv2.rectangle(sidebar_img, 
                  (button_x, stop_y), 
                  (button_x + BUTTON_WIDTH, stop_y + BUTTON_HEIGHT),
                  (0, 0, 0), 3)
    
    text = "EMERGENCY STOP"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    text_x = button_x + (BUTTON_WIDTH - text_size[0]) // 2
    text_y = stop_y + (BUTTON_HEIGHT + text_size[1]) // 2
    cv2.putText(sidebar_img, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return sidebar_img, (button_x, start_y, BUTTON_WIDTH, BUTTON_HEIGHT), \
           (button_x, stop_y, BUTTON_WIDTH, BUTTON_HEIGHT)

# ========================= 
# SAVE SNAPSHOT FUNCTION
# ========================= 
def save_snapshot_with_gui(frame, timestamp, total_flies, fly_counts, level_results, level_scores, snapshot_type="capture"):
    """บันทึกภาพพร้อม GUI"""
    gui_frame = create_gui_frame(frame, fly_counts, level_results, level_scores)
    
    filename = f"{SNAPSHOT_FOLDER}{snapshot_type}_{timestamp}_flies{total_flies}.jpg"
    cv2.imwrite(filename, gui_frame)
    print(f"💾 บันทึกภาพพร้อม GUI: {filename}")
    return filename

# =========================
# MOUSE CALLBACK
# =========================
def mouse_callback(event, x, y, flags, param):
    """จัดการคลิกเมาส์บนปุ่ม"""
    global is_running, emergency_stop
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_btn, stop_btn, ser = param
        
        # ตรวจสอบว่าคลิกที่ Start Button หรือไม่
        if (start_btn[0] <= x <= start_btn[0] + start_btn[2] and
            start_btn[1] <= y <= start_btn[1] + start_btn[3]):
            if not is_running:
                print("\n" + "="*60)
                print(">>> กดปุ่ม START - เริ่มประมวลผล")
                print("="*60)
                is_running = True
                emergency_stop = False
                send_esp32_command(ser, 'motor_start')
        
        # ตรวจสอบว่าคลิกที่ Emergency Stop Button หรือไม่
        elif (stop_btn[0] <= x <= stop_btn[0] + stop_btn[2] and
              stop_btn[1] <= y <= stop_btn[1] + stop_btn[3]):
            if is_running:
                print("\n" + "="*60)
                print(">>> กดปุ่ม EMERGENCY STOP - หยุดฉุกเฉิน")
                print("="*60)
                is_running = False
                emergency_stop = True
                send_esp32_command(ser, 'emergency_stop')

# ========================= 
# MAIN PROGRAM
# ========================= 
def main():
    global is_running, emergency_stop, waiting_for_5sec_capture, capture_5sec_time
    
    # เปิดแหล่งวิดีโอ (กล้องหรือไฟล์)
    if USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_FILE_PATH)
        if not cap.isOpened():
            print(f"❌ ไม่สามารถเปิดไฟล์วิดีโอ: {VIDEO_FILE_PATH}")
            print("กรุณาตรวจสอบ path ของไฟล์")
            return
        print(f"✓ เปิดไฟล์วิดีโอ: {VIDEO_FILE_PATH}")
        print(f"  - Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(f"  - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not cap.isOpened():
            print(f"❌ ไม่สามารถเปิดกล้อง index {CAMERA_INDEX}!")
            return
        print(f"✓ เปิดกล้อง index {CAMERA_INDEX}")
    
    # เชื่อมต่อ ESP32
    ser = None
    if ENABLE_ESP32:
        try:
            ser = serial.Serial(ESP32_PORT, ESP32_BAUDRATE, timeout=1)
            time.sleep(2)
            print(f"✓ เชื่อมต่อ ESP32 ที่ {ESP32_PORT} สำเร็จ")
        except Exception as e:
            print(f"✗ ไม่สามารถเชื่อมต่อ ESP32: {e}")
            print("✓ ทำงานต่อโดยไม่มี ESP32")
            ser = None
    else:
        print("✓ โหมดทดสอบ - ไม่เชื่อมต่อ ESP32")
    
    print(f"\n{'='*50}")
    print(f"โหมด: {'VIDEO FILE' if USE_VIDEO_FILE else 'CAMERA'}")
    print(f"โฟลเดอร์ภาพ: {SNAPSHOT_FOLDER}")
    print(f"{'='*50}")
    print("คำสั่ง:")
    print("  - คลิกปุ่ม START เพื่อเริ่มประมวลผล")
    print("  - คลิกปุ่ม EMERGENCY STOP เพื่อหยุดฉุกเฉิน")
    if USE_VIDEO_FILE:
        print("  - กด SPACE เพื่อ Pause/Resume วิดีโอ")
        print("  - กด 'r' เพื่อ Restart วิดีโอ")
    print("  - กดปุ่มขยายหน้าต่างมุมขวาบนเพื่อขยายเต็มจอ")
    print("  - กด 'q' เพื่อออกจากโปรแกรม")
    print(f"{'='*50}\n")
    
    frame_count = 0
    snapshot_count = 0
    last_check_result = None
    paused = False
    frame = None  # Initialize frame
    
    # สร้างหน้าต่างและ set mouse callback
    cv2.namedWindow("Fly Counter - Test Mode", cv2.WINDOW_NORMAL)
    
    while True:
        if not paused or not USE_VIDEO_FILE:
            ret, new_frame = cap.read()
            if not ret:
                if USE_VIDEO_FILE:
                    print("\n🎬 วิดีโอเล่นจบแล้ว - กด 'r' เพื่อเริ่มใหม่ หรือ 'q' เพื่อออก")
                    paused = True
                    # ถ้ายังไม่มี frame เลย ให้ข้ามไป
                    if frame is None:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('r'):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_count = 0
                            paused = False
                            print("🔄 RESTART วิดีโอ")
                        continue
                else:
                    print("❌ ไม่สามารถอ่าน frame จากกล้อง")
                    break
            else:
                frame = new_frame
        else:
            # ถ้า paused ให้ใช้ frame เดิม
            pass
        
        # ตรวจสอบว่ามี frame หรือไม่
        if frame is None:
            continue
            
        frame_count += 1
        current_time = time.time()
        
        # สร้าง display frame
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # ถ้ากำลัง running จะประมวลผลด้วย OpenCV
        if is_running and not emergency_stop and not paused:
            # ประมวลผล
            processed_frame, all_below_L1, total_flies, fly_counts, level_results, level_scores = process_frame(frame)
            display_frame = processed_frame
            
            # ตรวจสอบสถานะแมลงวัน
            if not waiting_for_5sec_capture:
                # ตรวจสอบว่าแมลงอยู่ใต้ L1 หรือยัง
                if all_below_L1 and total_flies > 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> แมลงทั้งหมดอยู่ใต้เส้น L1 - เริ่มนับ 5 วินาที")
                    waiting_for_5sec_capture = True
                    capture_5sec_time = current_time
                    send_esp32_command(ser, 'motor_stop')
                else:
                    # แมลงยังไม่ลงครบ - ส่งสัญญาณ motor_start ต่อเนื่อง
                    if frame_count % 30 == 0:  # ทุก 30 frames
                        send_esp32_command(ser, 'motor_start')
                    
                    last_check_result = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'total_flies': total_flies,
                        'all_below_L1': all_below_L1,
                        'status': 'Waiting for flies to settle...'
                    }
            else:
                # กำลังรอครบ 5 วินาที
                elapsed = current_time - capture_5sec_time
                remaining = WAIT_DURATION - elapsed
                
                if remaining > 0:
                    last_check_result = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'total_flies': total_flies,
                        'all_below_L1': all_below_L1,
                        'status': f'Waiting for 5s capture: {remaining:.1f}s'
                    }
                else:
                    # ครบ 5 วินาที - แคปภาพด้วย DUAL ALGORITHM
                    print(f"\n{'='*50}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ครบ 5 วินาที - แคปและประมวลผลด้วย DUAL ALGORITHM...")
                    
                    processed_frame_yolo, all_below_L1_final, total_flies_final, fly_counts_final, \
                    level_results_final, level_scores_final = process_frame_with_yolo(frame)
                    
                    print(f"Total flies: {total_flies_final}")
                    print(f"Flies per tube: {fly_counts_final}")
                    
                    # บันทึกภาพ
                    timestamp_5sec = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_count += 1
                    save_snapshot_with_gui(processed_frame_yolo, timestamp_5sec, total_flies_final, 
                                          fly_counts_final, level_results_final, level_scores_final, 
                                          snapshot_type=f"final_{snapshot_count:03d}")
                    
                    print(f"✓ บันทึกภาพที่วินาทีที่ 5 เรียบร้อย")
                    print(f"{'='*50}\n")
                    
                    # รีเซ็ตสถานะและหยุด
                    waiting_for_5sec_capture = False
                    is_running = False
                    last_check_result = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'total_flies': total_flies_final,
                        'all_below_L1': True,
                        'status': 'Complete - Ready for next run'
                    }
        
        # สร้าง GUI frame พร้อมปุ่ม
        min_x = min([cfg['offset_from_left'] for cfg in TUBE_CONFIGS]) - 50
        max_x = max([cfg['offset_from_left'] + cfg['width'] for cfg in TUBE_CONFIGS]) + 100
        roi_y_start = 0
        roi_y_end = 1000
        
        cropped_display = display_frame[roi_y_start:roi_y_end, min_x:max_x]
        h_crop, w_crop = cropped_display.shape[:2]
        
        gui_img = np.ones((h_crop, w_crop + SIDEBAR_WIDTH, 3), dtype=np.uint8) * 255
        gui_img[:, :SIDEBAR_WIDTH] = SIDEBAR_COLOR
        gui_img[:h_crop, SIDEBAR_WIDTH:SIDEBAR_WIDTH+w_crop] = cropped_display
        
        # วาดปุ่ม
        gui_img, start_btn_rect, stop_btn_rect = draw_buttons(gui_img, is_running)
        
        # แสดงสถานะ
        status_y = 50
        
        # แสดงโหมด
        mode_text = "VIDEO MODE" if USE_VIDEO_FILE else "CAMERA MODE"
        cv2.putText(gui_img, mode_text, (20, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if USE_VIDEO_FILE:
            # แสดง frame number และสถานะ pause
            frame_info = f"Frame: {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}"
            if paused:
                frame_info += " [PAUSED]"
            cv2.putText(gui_img, frame_info, (20, status_y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        if last_check_result:
            cv2.putText(gui_img, f"Status: {last_check_result['status']}", 
                       (20, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(gui_img, f"Last Check: {last_check_result['time']}", 
                       (20, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(gui_img, f"Snapshots: {snapshot_count}", 
                   (20, h_crop - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # แสดงผล
        cv2.imshow("Fly Counter - Test Mode", gui_img)
        
        # Set mouse callback with button positions
        cv2.setMouseCallback("Fly Counter - Test Mode", mouse_callback, 
                            (start_btn_rect, stop_btn_rect, ser))
        
        # ตรวจสอบคีย์บอร์ด
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and USE_VIDEO_FILE:
            paused = not paused
            print(f"{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")
        elif key == ord('r') and USE_VIDEO_FILE:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            paused = False
            print("🔄 RESTART วิดีโอ")
    
    # ปิดทุกอย่าง
    print("\nกำลังปิดโปรแกรม...")
    if is_running:
        send_esp32_command(ser, 'motor_stop')
    
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    
    print(f"\n{'='*50}")
    print(f"สรุป:")
    print(f"จำนวน frames ที่แสดง: {frame_count}")
    print(f"จำนวนภาพที่บันทึก: {snapshot_count}")
    print(f"โฟลเดอร์ภาพ: {SNAPSHOT_FOLDER}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
