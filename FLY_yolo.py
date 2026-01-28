import cv2
import numpy as np
import pandas as pd
import os
import glob
from ultralytics import YOLO

# ========================= 
# LEVEL CONFIG
# ========================= 
LEVEL_THICKNESS = 2
LEVEL_COUNT = 6  # มี 5 เส้น = 6 ช่อง (0-5)

# กำหนดตำแหน่ง Y ของเส้นแต่ละเส้น (จากบนลงล่าง)
LEVEL_Y_POSITIONS = [300, 450, 600, 750, 900]

# กำหนดสีต่างกันสำหรับแต่ละ Level (BGR format)
LEVEL_COLORS = {
    5: (0, 255, 0),      # L5 - เขียว
    4: (255, 0, 0),      # L4 - น้ำเงิน
    3: (0, 255, 255),    # L3 - เหลือง
    2: (255, 255, 0),    # L2 - ฟ้า
    1: (0, 0, 255),      # L1 - แดง
}

LEVEL_COUNT_COLOR = (0, 255, 255)

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
# TOTAL FLY COUNT INPUT
# =========================
TOTAL_FLIES_PER_TUBE = [
    15,  # Tube 1
    15,  # Tube 2
    15,  # Tube 3
    15,  # Tube 4
    15   # Tube 5
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

# =========================
# YOLO CONFIG
# =========================
YOLO_MODEL_PATH = 'yolo26n_v2.pt'  # path ไปยัง trained model
YOLO_CONF_THRESHOLD = 0.65  # confidence threshold
YOLO_INPUT_SIZE = 640  # ขนาด input ของ YOLO
OVERLAP_PERCENT = 0.5  # overlap 30%

# =========================
# LOAD YOLO MODEL
# =========================
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✓ Loaded YOLO model from: {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    print("กรุณาตรวจสอบ path ของ model")
    yolo_model = None

# =========================
# FUNCTION: CROP REGIONS WITH OVERLAP
# =========================
def create_level_crops(tube_roi, tube_y_start, x_start):
    """
    แบ่ง tube เป็น crops ตาม levels โดยมี overlap 30%
    
    Returns:
        crops: list of {'img': crop_img, 'y_start': int, 'y_end': int, 'level_range': tuple}
    """
    h, w = tube_roi.shape[:2]
    LEVEL_Y_DRAW = [y - tube_y_start for y in LEVEL_Y_POSITIONS]
    
    # เพิ่ม boundary ที่ top และ bottom
    boundaries = [0] + LEVEL_Y_DRAW + [h]
    
    crops = []
    
    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]
        
        # คำนวณ overlap
        crop_height = y_end - y_start
        overlap_pixels = int(crop_height * OVERLAP_PERCENT)
        
        # ขยาย crop ขึ้นและลง
        crop_y_start = max(0, y_start - overlap_pixels)
        crop_y_end = min(h, y_end + overlap_pixels)
        
        # crop ภาพ
        crop_img = tube_roi[crop_y_start:crop_y_end, :]
        
        # ระบุ level range ที่ crop นี้ครอบคลุม
        level_num = len(boundaries) - 2 - i  # L5 = 5, L4 = 4, ..., L0 = 0
        
        crops.append({
            'img': crop_img,
            'y_start': crop_y_start,  # relative to tube
            'y_end': crop_y_end,
            'level': level_num,
            'original_y_start': y_start,  # boundary ที่แท้จริงของ level
            'original_y_end': y_end
        })
    
    return crops

# =========================
# FUNCTION: RESIZE TO YOLO INPUT
# =========================
def resize_to_yolo_input(img, target_size=640):
    """
    Resize image to YOLO input size (square) with padding
    """
    h, w = img.shape[:2]
    
    # คำนวณ scale
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # resize
    resized = cv2.resize(img, (new_w, new_h))
    
    # สร้าง canvas สีดำ
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # วางภาพกลาง canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas, scale, x_offset, y_offset

# =========================
# FUNCTION: DETECT FLIES WITH YOLO
# =========================
def detect_flies_yolo(crop_info, conf_threshold=0.25):
    """
    ใช้ YOLO detect แมลงวันใน crop
    
    Returns:
        detections: list of {'x': int, 'y': int, 'w': int, 'h': int, 'conf': float}
                    (coordinates relative to original crop)
    """
    if yolo_model is None:
        return []
    
    crop_img = crop_info['img']
    
    # Resize to YOLO input
    yolo_input, scale, x_offset, y_offset = resize_to_yolo_input(crop_img, YOLO_INPUT_SIZE)
    
    # Run YOLO detection
    results = yolo_model(yolo_input, conf=conf_threshold, verbose=False)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates (in yolo_input coordinate)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            # แปลงกลับไปยัง crop coordinate
            x1_orig = int((x1 - x_offset) / scale)
            y1_orig = int((y1 - y_offset) / scale)
            x2_orig = int((x2 - x_offset) / scale)
            y2_orig = int((y2 - y_offset) / scale)
            
            # ตรวจสอบว่าอยู่ใน valid range
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

# =========================
# FUNCTION: REMOVE DUPLICATE DETECTIONS
# =========================
def remove_duplicates(all_detections, iou_threshold=0.5):
    """
    ลบ detection ที่ซ้ำกันจาก overlapping crops
    โดยใช้ Non-Maximum Suppression (NMS)
    """
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
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )
    
    # Filter detections
    filtered = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered.append(all_detections[i])
    
    return filtered

# =========================
# FUNCTION: PROCESS IMAGE WITH YOLO
# =========================
def process_image(img_path):
    """ประมวลผลรูปภาพเดียวด้วย YOLO"""
    
    # อ่านรูปภาพ
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None, None
    
    orig = img.copy()
    h_img, w_img = img.shape[:2]

    print(f"\nProcessing: {os.path.basename(img_path)}")
    print(f"Image size: {w_img} x {h_img}")
    
    # สร้างภาพสำหรับแสดง detection areas
    detection_vis = np.zeros((h_img, w_img, 3), dtype=np.uint8)

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
    tube_level_results = []
    tube_level_scores = []
    fly_counts = []
    colors = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]

    print("=== TUBE ANALYSIS WITH YOLO ===")

    for i, tube in enumerate(tube_positions):
        cfg = TUBE_CONFIGS[i]
        x_start = tube['x_start']
        x_end = tube['x_end']
        tube_y_start = roi_y_start + cfg['top_offset']
        tube_y_end   = roi_y_end   - cfg['bottom_offset']

        level_counts = [0] * LEVEL_COUNT
        
        # draw tube boundary
        cv2.rectangle(
            orig,
            (x_start, tube_y_start),
            (x_end, tube_y_end),
            colors[i], 2
        )

        cv2.putText(
            orig, f"Tube {i+1}",
            (x_start + 50, tube_y_start - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2
        )

        # extract tube ROI
        tube_roi = img[tube_y_start:tube_y_end, x_start:x_end]
        LEVEL_Y_DRAW = [y - 50 for y in LEVEL_Y_POSITIONS]

        if tube_roi.size > 0:
            # สร้าง crops สำหรับแต่ละ level พร้อม overlap
            crops = create_level_crops(tube_roi, tube_y_start, x_start)
            
            print(f"\nTube {i+1}: Created {len(crops)} crops with {OVERLAP_PERCENT*100}% overlap")
            
            # Detect แมลงวันในแต่ละ crop
            all_tube_detections = []
            
            for crop_idx, crop_info in enumerate(crops):
                detections = detect_flies_yolo(crop_info, YOLO_CONF_THRESHOLD)
                
                # แปลง coordinates จาก crop เป็น tube coordinate
                for det in detections:
                    det['y'] += crop_info['y_start']  # offset ใน tube
                    det['tube_y'] = det['y'] + tube_y_start  # absolute y
                    det['tube_x'] = det['x'] + x_start  # absolute x
                    det['level'] = crop_info['level']
                
                all_tube_detections.extend(detections)
                
                # แสดง crop area บน detection_vis
                crop_y_abs = crop_info['y_start'] + tube_y_start
                crop_y_end_abs = crop_info['y_end'] + tube_y_start
                cv2.rectangle(
                    detection_vis,
                    (x_start, crop_y_abs),
                    (x_end, crop_y_end_abs),
                    (0, 100, 100), 1
                )
            
            # ลบ detections ที่ซ้ำกันจาก overlap
            unique_detections = remove_duplicates(all_tube_detections, iou_threshold=0.4)
            
            print(f"  Total detections before NMS: {len(all_tube_detections)}")
            print(f"  Unique detections after NMS: {len(unique_detections)}")
            
            # นับแมลงวันในแต่ละ level
            for det in unique_detections:
                # วาด bounding box
                cv2.rectangle(
                    orig,
                    (int(det['tube_x']), int(det['tube_y'])),
                    (int(det['tube_x'] + det['w']), int(det['tube_y'] + det['h'])),
                    (0, 255, 0), 2
                )
                
                # แสดง confidence
                conf_text = f"{det['conf']:.2f}"
                cv2.putText(
                    orig, conf_text,
                    (int(det['tube_x']), int(det['tube_y']) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )
                
                # คำนวณ level (ใช้จุดกึ่งกลาง bounding box)
                cy = det['tube_y'] + det['h'] // 2
                
                assigned = False
                for lv_idx in range(len(LEVEL_Y_POSITIONS)):
                    if cy < LEVEL_Y_DRAW[lv_idx]:
                        level_counts[5 - lv_idx] += 1
                        assigned = True
                        break
                
                if not assigned:
                    level_counts[0] += 1
            
            tube_fly_count = len(unique_detections)
            
            # เพิ่มแมลงที่ไม่เห็นให้กับ L5
            total_expected = TOTAL_FLIES_PER_TUBE[i] if i < len(TOTAL_FLIES_PER_TUBE) else 0
            
            if total_expected > 0 and tube_fly_count < total_expected:
                unseen_flies = total_expected - tube_fly_count
                level_counts[5] += unseen_flies
                print(f"  Detected {tube_fly_count} flies, Expected {total_expected} flies")
                print(f"  -> Added {unseen_flies} unseen flies to L5")
        
        # draw level lines & labels
        for lv_idx, y_draw in enumerate(LEVEL_Y_DRAW):
            display_level = 5 - lv_idx
            cv2.line(orig, (x_start, y_draw), (x_end, y_draw),
                    LEVEL_COLORS[display_level], LEVEL_THICKNESS)
            
            if i == 4:  # หลอดที่ 5
                label_text = f"L{display_level}"
                label_x = x_end + 15
                label_y = y_draw + 10
                
                cv2.putText(orig, label_text,
                           (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1.5,
                           LEVEL_COLORS[display_level],
                           3)

        fly_counts.append(total_expected if total_expected > 0 else tube_fly_count)
        tube_level_results.append(level_counts)
        
        # SCORE CALC
        score_per_level = []
        for lv in range(LEVEL_COUNT):
            score_per_level.append(level_counts[lv] * LEVEL_SCORES[lv])
        tube_level_scores.append(score_per_level)
        
        print(f"Tube {i+1}: Total = {fly_counts[i]} flies (Detected: {tube_fly_count})")
        print(f"  Levels : {level_counts[::-1]}")

    # DRAW LEGEND
    legend_x = w_img - 280
    legend_y = 20
    legend_w = 200
    legend_h = 220
    line_spacing = 30

    cv2.rectangle(orig, 
                  (legend_x, legend_y), 
                  (legend_x + legend_w, legend_y + legend_h),
                  (0, 0, 0), -1)

    cv2.rectangle(orig, 
                  (legend_x, legend_y), 
                  (legend_x + legend_w, legend_y + legend_h),
                  (255, 255, 255), 2)

    cv2.putText(orig, "Level Colors",
                (legend_x + 10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for idx, level_num in enumerate([5, 4, 3, 2, 1]):
        y_pos = legend_y + 50 + idx * line_spacing
        color = LEVEL_COLORS[level_num]
        
        cv2.line(orig,
                 (legend_x + 10, y_pos),
                 (legend_x + 60, y_pos),
                 color, 3)
        
        cv2.putText(orig, f"L{level_num}",
                    (legend_x + 70, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # DRAW TABLE (เหมือนเดิม)
    table_x = 20
    table_y = 120
    row_h = 35
    left_w = 60
    col_w = 60
    font_scale = 0.7
    thick = 2
    header_gap = 5

    levels = ['L5','L4','L3','L2','L1','L0']
    tubes  = [f'T{i+1}' for i in range(len(tube_level_results))]

    level_totals = []
    for c in range(len(tubes)):
        total = sum(tube_level_results[c])
        level_totals.append(total)

    table_w = left_w + col_w * len(tubes)
    table_h = row_h * (len(levels) + 2) + header_gap

    # Header box
    cv2.rectangle(orig, (table_x, 40), 
                  (table_x + 110, 120),
                  (0, 0, 0), -1)
    cv2.rectangle(orig, 
                  (table_x, 40), 
                  (table_x + 110, 120),
                  (255, 255, 255), 2)
    cv2.putText(orig, "Result", (table_x + 5, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thick)

    # Table background
    cv2.rectangle(orig, (table_x, table_y-row_h),
                  (table_x+table_w, table_y+table_h-25),
                  (0,0,0), -1)
    cv2.rectangle(orig, (table_x, table_y-row_h),
                  (table_x+table_w, table_y+table_h-25),
                  (255,255,255), 2)

    # Horizontal lines
    cv2.line(orig, (table_x, table_y + header_gap), 
             (table_x+table_w, table_y + header_gap), 
             (255,255,255), 1)

    for r in range(len(levels)):
        y_line = table_y + header_gap + (r+1)*row_h
        cv2.line(orig, (table_x, y_line), 
                 (table_x+table_w, y_line), 
                 (100,100,100), 1)

    total_line_y = table_y + header_gap + (len(levels)+1)*row_h
    cv2.line(orig, (table_x, total_line_y-row_h), 
             (table_x+table_w, total_line_y-row_h), 
             (255,255,255), 2)

    # Vertical lines
    cv2.line(orig, (table_x+left_w, table_y-row_h), 
             (table_x+left_w, table_y+table_h-25), 
             (255,255,255), 1)

    for c in range(1, len(tubes)):
        x_line = table_x + left_w + c*col_w
        cv2.line(orig, (x_line, table_y-row_h), 
                 (x_line, table_y+table_h-25), 
                 (100,100,100), 1)

    # Header row
    header_y = table_y - 10
    cv2.putText(orig, "Tube", (table_x + 5, header_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for i, t in enumerate(tubes):
        text_x = table_x + left_w + i*col_w + 15
        cv2.putText(orig, t, (text_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    # Data rows
    for r, lvl in enumerate(levels):
        y = table_y + header_gap + (r+1)*row_h - 10
        
        cv2.putText(orig, lvl, (table_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick)
        
        for c in range(len(tubes)):
            level_index = 5 - r
            val = tube_level_results[c][level_index]
            text_x = table_x + left_w + c*col_w + 20
            cv2.putText(orig, str(val), (text_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick)

    # Total row
    total_y = table_y + header_gap + (len(levels)+1)*row_h - 10
    cv2.putText(orig, "Total", (table_x + 5, total_y+5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for c in range(len(tubes)):
        cv2.putText(orig, str(level_totals[c]),
                    (table_x + left_w + c*col_w + 13, total_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    # DRAW SCORE TABLE (เหมือนเดิม)
    score_totals = []
    for c in range(len(tubes)):
        total = sum(tube_level_scores[c])
        score_totals.append(total)

    score_table_y = table_y + table_h + 100

    cv2.rectangle(orig, (table_x, score_table_y), 
                  (table_x + 110, score_table_y-80),
                  (0, 0, 0), -1)
    cv2.rectangle(orig, 
                  (table_x, score_table_y), 
                  (table_x + 110, score_table_y-80),
                  (255, 255, 255), 2)
    cv2.putText(orig, "Score", (table_x + 5, score_table_y-50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thick)

    cv2.rectangle(orig, (table_x, score_table_y-row_h),
                  (table_x+table_w, score_table_y+table_h-25),
                  (0,0,0), -1)
    cv2.rectangle(orig, (table_x, score_table_y-row_h),
                  (table_x+table_w, score_table_y+table_h-25),
                  (255,255,255), 2)

    cv2.line(orig, (table_x, score_table_y + header_gap), 
             (table_x+table_w, score_table_y + header_gap), 
             (255,255,255), 1)

    for r in range(len(levels)):
        y_line = score_table_y + header_gap + (r+1)*row_h
        cv2.line(orig, (table_x, y_line), 
                 (table_x+table_w, y_line), 
                 (100,100,100), 1)

    score_total_line_y = score_table_y + header_gap + (len(levels)+1)*row_h
    cv2.line(orig, (table_x, score_total_line_y-row_h), 
             (table_x+table_w, score_total_line_y-row_h), 
             (255,255,255), 2)

    cv2.line(orig, (table_x+left_w, score_table_y-row_h), 
             (table_x+left_w, score_table_y+table_h-25), 
             (255,255,255), 1)

    for c in range(1, len(tubes)):
        x_line = table_x + left_w + c*col_w
        cv2.line(orig, (x_line, score_table_y-row_h), 
                 (x_line, score_table_y+table_h-25), 
                 (100,100,100), 1)
        
    header_y = score_table_y - 10
    cv2.putText(orig, "Tube", (table_x + 5, header_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for i, t in enumerate(tubes):
        cv2.putText(orig, t,
                    (table_x + left_w + i*col_w + 15, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for r, lvl in enumerate(levels):
        y = score_table_y + header_gap + (r+1)*row_h - 10
        cv2.putText(orig, lvl, (table_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick)
        for c in range(len(tubes)):
            score_val = tube_level_scores[c][5-r]
            cv2.putText(orig, str(score_val),
                        (table_x + left_w + c*col_w + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick)

    score_total_y = score_table_y + header_gap + (len(levels)+1)*row_h - 10
    cv2.putText(orig, "Total", (table_x + 5, score_total_y+5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)

    for c in range(len(tubes)):
        cv2.putText(orig, str(score_totals[c]),
                    (table_x + left_w + c*col_w + 13, score_total_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thick)
    
    return orig, detection_vis

# =========================
# FUNCTION: RESIZE IMAGE
# =========================
def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height))

# =========================
# MAIN PROGRAM
# =========================
def main():
    # ตรวจสอบว่า YOLO model โหลดสำเร็จหรือไม่
    if yolo_model is None:
        print("\n!!! ไม่สามารถโหลด YOLO model ได้ !!!")
        print("กรุณาตรวจสอบ path ของ model ใน YOLO_MODEL_PATH")
        return
    
    # กำหนด path ของโฟลเดอร์ที่เก็บรูปภาพ
    folder_path = "D:/med_project/frames_by_time2/"
    
    # หารูปภาพทั้งหมดในโฟลเดอร์
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(folder_path, "*.jpeg"))) + \
                  sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    if len(image_files) == 0:
        print(f"ไม่พบรูปภาพในโฟลเดอร์: {folder_path}")
        return
    
    print(f"พบรูปภาพทั้งหมด {len(image_files)} รูป")
    print("\nคำแนะนำ:")
    print("- กด 'N' หรือ 'n' เพื่อไปรูปถัดไป")
    print("- กด 'P' หรือ 'p' เพื่อกลับรูปก่อนหน้า")
    print("- กด 'Q' หรือ 'q' เพื่อออกจากโปรแกรม")
    print("="*50)
    
    current_index = 0
    
    while True:
        # ประมวลผลรูปภาพปัจจุบัน
        result_img, detection_img = process_image(image_files[current_index])
        
        if result_img is None:
            print(f"ข้ามรูปที่ {current_index + 1}")
            current_index += 1
            if current_index >= len(image_files):
                print("จบการประมวลผลทั้งหมดแล้ว")
                break
            continue
        
        # แสดงผล - ภาพผลลัพธ์หลัก
        display_img = resize_image(result_img, 70)
        
        # เพิ่มข้อความแสดงลำดับรูปภาพ
        text = f"Image {current_index + 1}/{len(image_files)} - YOLO Detection - Press 'N' for next, 'P' for previous, 'Q' to quit"
        cv2.putText(display_img, text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("YOLO Fly Counter - Navigate with N/P/Q", display_img)
        
        # แสดงผล - ภาพแสดง crop areas
        display_detection = resize_image(detection_img, 70)
        
        text_detection = f"Detection Areas - Showing crop regions with 30% overlap"
        cv2.putText(display_detection, text_detection,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Crop Areas Visualization", display_detection)
        
        # รอการกดปุ่ม
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('n') or key == ord('N'):
                current_index += 1
                if current_index >= len(image_files):
                    print("\nถึงรูปสุดท้ายแล้ว")
                    cv2.destroyAllWindows()
                    return
                break
            
            elif key == ord('p') or key == ord('P'):
                if current_index > 0:
                    current_index -= 1
                    break
                else:
                    print("\nอยู่ที่รูปแรกแล้ว")
            
            elif key == ord('q') or key == ord('Q'):
                print("\nออกจากโปรแกรม")
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()