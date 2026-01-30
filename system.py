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
# ESP32 CONFIG
# ========================= 
ESP32_PORT = 'COM3'
ESP32_BAUDRATE = 115200
TRIGGER_COMMAND = 'CAPTURE'

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
    {'offset_from_left': 280,  'width': 200, 'top_offset': 40, 'bottom_offset': 3},
    {'offset_from_left': 540,  'width': 200, 'top_offset': 40, 'bottom_offset': 3},
    {'offset_from_left': 815,  'width': 200, 'top_offset': 40, 'bottom_offset': 3},
    {'offset_from_left': 1095, 'width': 200, 'top_offset': 40, 'bottom_offset': 3},
    {'offset_from_left': 1375, 'width': 215, 'top_offset': 40, 'bottom_offset': 3},
]

# สีสำหรับ tube borders (BGR format) - สีสวยกว่าสีดำ
colors = [
    (0, 0, 0),  # Tube 1 - 
    (0, 0, 0),  # Tube 2 - 
    (0, 0, 0),  # Tube 3 - 
    (0, 0, 0),  # Tube 4 -
    (0, 0, 0),  # Tube 5 - 
]

# =========================
# SIDEBAR CONFIG
# =========================
SIDEBAR_WIDTH = 400
SIDEBAR_COLOR = (210, 210, 210)  # สีเทาเข้ม

# =========================
# YOLO CONFIG
# =========================
YOLO_MODEL_PATH = 'yolo26n_v2.pt'  # path ไปยัง trained model
YOLO_CONF_THRESHOLD = 0.65  # confidence threshold
YOLO_INPUT_SIZE = 640  # ขนาด input ของ YOLO
OVERLAP_PERCENT = 0.5  # overlap 50%

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
trigger_queue = queue.Queue()
waiting_for_5sec_capture = False
capture_5sec_time = 0
WAIT_DURATION = 5  # รอ 5 วินาทีหลังจากได้สัญญาณแรก

# ========================= 
# ESP32 SERIAL READER THREAD
# ========================= 
def read_esp32(ser, trigger_queue):
    """อ่านข้อมูลจาก ESP32 ในเธรดแยก"""
    print("เริ่มอ่านข้อมูลจาก ESP32...")
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                print(f"ESP32: {line}")
                if TRIGGER_COMMAND in line:
                    trigger_queue.put(True)
                    print(">>> ได้รับสัญญาณถ่ายภาพจาก ESP32!")
        except Exception as e:
            print(f"Error reading ESP32: {e}")
            time.sleep(0.1)

# =========================
# YOLO HELPER FUNCTIONS
# =========================
def create_level_crops(tube_roi, tube_y_start, x_start):
    """แบ่ง tube เป็น crops ตาม levels โดยมี overlap"""
    h, w = tube_roi.shape[:2]
    
    # ตรวจสอบว่า tube_roi ไม่ว่าง
    if h == 0 or w == 0:
        return []
    
    # คำนวณตำแหน่ง level เหมือนกับใน process_frame
    LEVEL_Y_DRAW = [y - 50 for y in LEVEL_Y_POSITIONS]  # ตำแหน่งจริงในภาพ
    
    # แปลงเป็น relative position ใน tube ROI
    LEVEL_Y_IN_TUBE = []
    for y in LEVEL_Y_DRAW:
        y_relative = y - tube_y_start
        # เก็บเฉพาะที่อยู่ใน tube
        if 0 < y_relative < h:
            LEVEL_Y_IN_TUBE.append(int(y_relative))
    
    # ถ้าไม่มี level ใน tube ให้ใช้ whole tube
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
    
    # เพิ่ม boundary ที่ top และ bottom
    boundaries = [0] + LEVEL_Y_IN_TUBE + [h]
    
    crops = []
    
    for i in range(len(boundaries) - 1):
        y_start = int(boundaries[i])
        y_end = int(boundaries[i + 1])
        
        # ตรวจสอบขนาด
        if y_end <= y_start:
            continue
        
        # คำนวณ overlap
        crop_height = y_end - y_start
        overlap_pixels = int(crop_height * OVERLAP_PERCENT)
        
        # ขยาย crop ขึ้นและลง
        crop_y_start = max(0, y_start - overlap_pixels)
        crop_y_end = min(h, y_end + overlap_pixels)
        
        # ตรวจสอบอีกครั้งหลัง overlap
        if crop_y_end <= crop_y_start:
            continue
        
        # ตรวจสอบว่าขนาดมากพอ (อย่างน้อย 10 pixels)
        if (crop_y_end - crop_y_start) < 10:
            continue
        
        # crop ภาพ
        crop_img = tube_roi[crop_y_start:crop_y_end, :].copy()
        
        # ตรวจสอบว่า crop มีขนาดจริง
        if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            continue
        
        # ระบุ level
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
    
    # ป้องกันขนาด 0
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
    
    # Resize to YOLO input
    result = resize_to_yolo_input(crop_img, YOLO_INPUT_SIZE)
    
    if result[0] is None:  # ถ้า resize ไม่สำเร็จ
        return []
    
    yolo_input, scale, x_offset, y_offset = result
    
    # Run YOLO detection
    results = yolo_model(yolo_input, conf=conf_threshold, verbose=False)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            # แปลงกลับไปยัง crop coordinate
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
# PROCESS FRAME FUNCTION
# ========================= 
def process_frame(img):
    """ประมวลผลภาพและคืนค่าภาพที่วาดแล้ว + ข้อมูล"""
    orig = img.copy()
    h_img, w_img = img.shape[:2]
    
    # WHITE MASK
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel_white = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_white)
    
    # BASELINE DETECTION
    roi_y_start = 0
    roi_y_end = 1000
    
    # TUBE POSITIONS
    tube_positions = []
    for i, cfg in enumerate(TUBE_CONFIGS):
        x_start = cfg['offset_from_left']
        x_end = x_start + cfg['width']
        tube_positions.append({
            'tube_num': i + 1,
            'x_start': x_start,
            'x_end': x_end
        })
    
    # ANALYZE EACH TUBE
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
        
        # draw tube
        cv2.rectangle(orig, (x_start, tube_y_start), (x_end, tube_y_end), colors[i], 2)
        cv2.putText(orig, f"Tube {i+1}", (x_start + 50, tube_y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
        
        # extract ROI
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
                    
                    # draw fly
                    cv2.rectangle(orig, (actual_x, actual_y),
                                (actual_x + fw, actual_y + fh), (0, 0, 255), 1)
                    
                    # LEVEL CALC
                    cy = actual_y + fh // 2
                    
                    # นับแมลงที่อยู่เหนือ L1
                    if cy < LEVEL_Y_DRAW[4]:
                        flies_above_L1 += 1
                    
                    # หาว่าแมลงอยู่ระหว่างเส้นไหน
                    assigned = False
                    for lv_idx in range(len(LEVEL_Y_DRAW)):
                        if cy < LEVEL_Y_DRAW[lv_idx]:
                            level_counts[5 - lv_idx] += 1
                            assigned = True
                            break
                    
                    if not assigned:
                        level_counts[0] += 1
        
        # เพิ่มแมลงที่ไม่เห็นให้กับ L5 (ถ้ามีการตั้งค่า)
        total_expected = TOTAL_FLIES_PER_TUBE[i] if i < len(TOTAL_FLIES_PER_TUBE) else 0
        if total_expected > 0 and tube_fly_count < total_expected:
            unseen_flies = total_expected - tube_fly_count
            level_counts[5] += unseen_flies
        
        # draw level lines
        for lv_idx, y_draw in enumerate(LEVEL_Y_DRAW):
            display_level = 5 - lv_idx
            cv2.line(orig, (x_start, y_draw), (x_end, y_draw),
                    LEVEL_COLORS[display_level], LEVEL_THICKNESS)
            
            if i == 4:  # หลอดที่ 5
                label_text = f"L{display_level}"
                label_x = x_end + 15
                label_y = y_draw - 55
                
                cv2.putText(orig, label_text, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           LEVEL_COLORS[display_level], 3)
        
        fly_counts.append(total_expected if total_expected > 0 else tube_fly_count)
        tube_level_results.append(level_counts)
        
        # SCORE CALC
        score_per_level = []
        for lv in range(LEVEL_COUNT):
            score_per_level.append(level_counts[lv] * LEVEL_SCORES[lv])
        tube_level_scores.append(score_per_level)
    
    # SUMMARY
    total_flies = sum(fly_counts)
    all_flies_below_L1 = (total_flies > 0) and (flies_above_L1 == 0)
    
    return orig, all_flies_below_L1, total_flies, fly_counts, tube_level_results, tube_level_scores

# =========================
# PROCESS FRAME WITH YOLO (สำหรับภาพวินาทีที่ 5)
# =========================
def process_frame_with_yolo(img):
    """ประมวลผลภาพด้วยทั้ง OpenCV และ YOLO แล้วรวมผล"""
    orig = img.copy()
    h_img, w_img = img.shape[:2]
    
    print("\n" + "="*60)
    print("PROCESSING WITH DUAL ALGORITHM (OpenCV + YOLO)")
    print("="*60)
    
    # BASELINE DETECTION
    roi_y_start = 0
    roi_y_end = 1000
    
    # TUBE POSITIONS
    tube_positions = []
    for i, cfg in enumerate(TUBE_CONFIGS):
        x_start = cfg['offset_from_left']
        x_end = x_start + cfg['width']
        tube_positions.append({
            'tube_num': i + 1,
            'x_start': x_start,
            'x_end': x_end
        })
    
    # เก็บผลลัพธ์แยกจาก 2 วิธี
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
        
        # draw tube
        cv2.rectangle(orig, (x_start, tube_y_start), (x_end, tube_y_end), colors[i], 2)
        cv2.putText(orig, f"Tube {i+1}", (x_start + 50, tube_y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
        
        # extract ROI
        tube_roi = img[tube_y_start:tube_y_end, x_start:x_end]
        
        # ===== METHOD 1: OpenCV Threshold =====
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
        
        # ===== METHOD 2: YOLO =====
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
        
        # ===== COMBINE RESULTS =====
        # รวม detections จากทั้ง 2 วิธี
        all_detections = opencv_detections + yolo_detections
        
        # วาดกรอบ: สีแดงทั้งหมด ไม่แสดงค่าความมั่นใจ
        level_counts = [0] * LEVEL_COUNT
        
        for det in all_detections:
            # ใช้สีแดงเหมือนกันทั้งหมด
            color = (0, 0, 255)  # แดง
            thickness = 1
            
            cv2.rectangle(orig, (det['x'], det['y']),
                        (det['x'] + det['w'], det['y'] + det['h']), color, thickness)
            
            # คำนวณ level
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
        
        # เพิ่มแมลงที่ไม่เห็น
        total_expected = TOTAL_FLIES_PER_TUBE[i] if i < len(TOTAL_FLIES_PER_TUBE) else 0
        if total_expected > 0 and tube_fly_count < total_expected:
            unseen_flies = total_expected - tube_fly_count
            level_counts[5] += unseen_flies
            print(f"  Added {unseen_flies} unseen flies to L5")
        
        # draw level lines
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
        
        # SCORE CALC
        score_per_level = []
        for lv in range(LEVEL_COUNT):
            score_per_level.append(level_counts[lv] * LEVEL_SCORES[lv])
        tube_level_scores.append(score_per_level)
        
        opencv_results.append(len(opencv_detections))
        yolo_results.append(len(yolo_detections))
        combined_results.append(tube_fly_count)
    
    # SUMMARY
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
    
    # หาขอบเขตของหลอดทดลอง
    min_x = min([cfg['offset_from_left'] for cfg in TUBE_CONFIGS]) - 50
    max_x = max([cfg['offset_from_left'] + cfg['width'] for cfg in TUBE_CONFIGS]) + 100
    roi_y_start = 0
    roi_y_end = 1000
    
    # ตัดเฉพาะส่วน ROI ที่มีหลอด
    cropped_orig = orig[roi_y_start:roi_y_end, min_x:max_x]
    h_crop, w_crop = cropped_orig.shape[:2]
    
    # สร้างภาพใหม่พร้อม sidebar (พื้นหลังสีขาว)
    new_img = np.ones((h_crop, w_crop + SIDEBAR_WIDTH, 3), dtype=np.uint8) * 255
    
    # วาง sidebar
    new_img[:, :SIDEBAR_WIDTH] = SIDEBAR_COLOR
    
    # วางภาพที่ตัดแล้ว
    new_img[:h_crop, SIDEBAR_WIDTH:SIDEBAR_WIDTH+w_crop] = cropped_orig
    
    # =========================
    # DRAW TABLE บน SIDEBAR
    # =========================
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
    
    # Header box - Drosophila No.
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

    # Horizontal lines
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

    # Vertical lines
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

    # =========================
    # DRAW SCORE TABLE
    # =========================
    score_totals = []
    for c in range(len(tubes)):
        total = sum(tube_level_scores[c])
        score_totals.append(total)

    score_table_y = table_y + table_h + 100

    # Header box - Score
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

    # Score table background
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
# SAVE SNAPSHOT FUNCTION
# ========================= 
def save_snapshot_with_gui(frame, timestamp, total_flies, fly_counts, level_results, level_scores, snapshot_type="trigger"):
    """บันทึกภาพพร้อม GUI"""
    # สร้าง GUI frame
    gui_frame = create_gui_frame(frame, fly_counts, level_results, level_scores)
    
    filename = f"{SNAPSHOT_FOLDER}{snapshot_type}_{timestamp}_flies{total_flies}.jpg"
    cv2.imwrite(filename, gui_frame)
    print(f"💾 บันทึกภาพพร้อม GUI: {filename}")
    return filename


# ========================= 
# MAIN PROGRAM
# ========================= 
def main():
    global waiting_for_5sec_capture, capture_5sec_time
    
    # เปิดกล้อง
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้!")
        return
    
    # เชื่อมต่อ ESP32
    ser = None
    try:
        ser = serial.Serial(ESP32_PORT, ESP32_BAUDRATE, timeout=1)
        time.sleep(2)
        print(f"✓ เชื่อมต่อ ESP32 ที่ {ESP32_PORT} สำเร็จ")
        
        esp32_thread = threading.Thread(target=read_esp32, args=(ser, trigger_queue), daemon=True)
        esp32_thread.start()
    except Exception as e:
        print(f"✗ ไม่สามารถเชื่อมต่อ ESP32: {e}")
        print("✓ ทำงานต่อโดยไม่มี ESP32 (กด 't' เพื่อจำลองการถ่าย)")
        ser = None
    
    print(f"\n{'='*50}")
    print(f"โฟลเดอร์ภาพ: {SNAPSHOT_FOLDER}")
    print(f"{'='*50}")
    print("คำสั่ง:")
    print("  - กด 'q' เพื่อออก")
    print("  - กด 't' เพื่อจำลองสัญญาณถ่ายภาพ (ถ้าไม่มี ESP32)")
    print("  - ESP32 ส่ง 'CAPTURE' เพื่อเริ่มถ่าย")
    print(f"{'='*50}\n")
    
    frame_count = 0
    snapshot_count = 0
    last_trigger_info = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่าน frame")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # สร้าง display frame สำหรับแสดงผล (ไม่บันทึกวิดีโอ)
        display_frame = frame.copy()
        
        # แสดงสถานะ
        h, w = display_frame.shape[:2]
        status_color = (0, 255, 0)
        status_text = "Live Camera - Ready"
        
        # ตรวจสอบว่ากำลังรอครบ 5 วินาทีหรือไม่
        if waiting_for_5sec_capture:
            elapsed = current_time - capture_5sec_time
            remaining = WAIT_DURATION - elapsed
            
            if remaining > 0:
                # แสดงการนับถอยหลัง
                status_color = (0, 165, 255)
                status_text = f"Waiting for 5s capture: {remaining:.1f}s"
            else:
                # ครบ 5 วินาที - แคปภาพและประมวลผล
                print(f"\n{'='*50}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ครบ 5 วินาที - กำลังแคปและประมวลผลด้วย DUAL ALGORITHM...")
                
                # ประมวลผล frame ปัจจุบันด้วยทั้ง OpenCV และ YOLO
                processed_frame, all_below_L1, total_flies, fly_counts, level_results, level_scores = process_frame_with_yolo(frame)
                
                print(f"Total flies: {total_flies}")
                print(f"Flies per tube: {fly_counts}")
                
                # บันทึกภาพพร้อม GUI
                timestamp_5sec = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_count += 1
                save_snapshot_with_gui(processed_frame, timestamp_5sec, total_flies, fly_counts, 
                                      level_results, level_scores, snapshot_type=f"5sec_{snapshot_count:03d}")
                
                print(f"✓ บันทึกภาพที่วินาทีที่ 5 เรียบร้อย")
                print(f"{'='*50}\n")
                
                # รีเซ็ตสถานะ
                waiting_for_5sec_capture = False
        
        # ตรวจสอบสัญญาณจาก ESP32
        if not trigger_queue.empty():
            trigger_queue.get()
            trigger_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"\n{'='*50}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ได้รับสัญญาณถ่าย! กำลังประมวลผล...")
            
            # ประมวลผล frame ปัจจุบัน
            processed_frame, all_below_L1, total_flies, fly_counts, level_results, level_scores = process_frame(frame)
            
            print(f"Total flies: {total_flies}")
            print(f"Flies per tube: {fly_counts}")
            print(f"All flies below L1: {all_below_L1}")
            
            # บันทึกภาพพร้อม GUI
            snapshot_count += 1
            save_snapshot_with_gui(processed_frame, trigger_timestamp, total_flies, fly_counts, 
                                  level_results, level_scores, snapshot_type=f"trigger_{snapshot_count:03d}")
            
            # เก็บข้อมูลสำหรับแสดงผล
            last_trigger_info = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'total_flies': total_flies,
                'all_below_L1': all_below_L1
            }
            
            # ตรวจสอบเงื่อนไข: ถ้าแมลงทั้งหมดอยู่ใต้ L1 → ตั้งเวลารอ 5 วินาที
            if all_below_L1:
                print(">>> แมลงทั้งหมดอยู่ใต้เส้น L1 - จะแคปภาพอีกครั้งเมื่อครบ 5 วินาที!")
                waiting_for_5sec_capture = True
                capture_5sec_time = current_time
            else:
                if total_flies == 0:
                    print(">>> ไม่มีแมลง - ไม่แคปภาพเพิ่ม")
                else:
                    print(">>> มีแมลงอยู่เหนือเส้น L1 - ไม่แคปภาพเพิ่ม")
                
                # ส่งสัญญาณกลับไป ESP32 ให้เคาะแมลงวัน
                if ser:
                    try:
                        ser.write(b'TAP\n')
                        print(">>> ส่งสัญญาณ 'TAP' ไปยัง ESP32 เพื่อเคาะแมลงวัน!")
                    except Exception as e:
                        print(f"Error sending TAP command: {e}")
            
            print(f"{'='*50}\n")
        
        # แสดงข้อมูลบน display frame
        cv2.putText(display_frame, status_text, (w - 450, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if last_trigger_info:
            result_text = f"Last: {last_trigger_info['time']} | Flies: {last_trigger_info['total_flies']}"
            result_color = (0, 255, 0) if last_trigger_info['all_below_L1'] else (0, 165, 255)
            cv2.putText(display_frame, result_text, (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
        
        cv2.putText(display_frame, f"Snapshots: {snapshot_count}", (w - 300, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # แสดงผล
        scale_percent = 50
        width = int(display_frame.shape[1] * scale_percent / 100)
        height = int(display_frame.shape[0] * scale_percent / 100)
        resized = cv2.resize(display_frame, (width, height))
        
        cv2.imshow("Fly Counter - Live Camera", resized)
        
        # ควบคุมด้วยคีย์บอร์ด
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            # จำลองสัญญาณ
            trigger_queue.put(True)
    
    # ปิดทุกอย่าง
    print("\nกำลังปิดโปรแกรม...")
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
