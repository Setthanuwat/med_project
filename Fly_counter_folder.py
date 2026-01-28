import cv2
import numpy as np
import pandas as pd
import os
import glob

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
    4: (255, 0, 0),     # L4 - น้ำเงิน
    3: (0, 255, 255),    # L3 - เหลือง
    2: (255, 255, 0),     # L2 - ฟ้า
    1: (0, 0, 255),    # L1 - แดง
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

USE_RELATIVE_SPACING = False

# =========================
# FUNCTION: PROCESS IMAGE
# =========================
def process_image(img_path):
    """ประมวลผลรูปภาพเดียว"""
    
    # อ่านรูปภาพ
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None, None
    
    orig = img.copy()
    h_img, w_img = img.shape[:2]

    print(f"\nProcessing: {os.path.basename(img_path)}")
    print(f"Image size: {w_img} x {h_img}")
    
    # สร้างภาพขาวดำเต็มจอสำหรับรวม th_clean จากทุกหลอด
    full_threshold_img = np.zeros((h_img, w_img), dtype=np.uint8)

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
    tube_rois = []
    fly_counts = []
    tube_level_results = []
    tube_level_scores = []
    colors = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]

    print("=== TUBE ANALYSIS ===")

    for i, tube in enumerate(tube_positions):
        cfg = TUBE_CONFIGS[i]
        x_start = tube['x_start']
        x_end = tube['x_end']
        tube_y_start = roi_y_start + cfg['top_offset']
        tube_y_end   = roi_y_end   - cfg['bottom_offset']

        level_counts = [0] * LEVEL_COUNT
        tube_fly_count = 0
        
        # draw tube
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

        # extract ROI
        tube_roi = img[tube_y_start:tube_y_end, x_start:x_end]
        tube_rois.append(tube_roi)
        LEVEL_Y_DRAW = [y - 50 for y in LEVEL_Y_POSITIONS]

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
            
            # วาง th_clean ของหลอดนี้ลงบนภาพเต็มจอ
            full_threshold_img[tube_y_start:tube_y_end, x_start:x_end] = th_clean
            
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
                    
                    # draw fly on original
                    cv2.rectangle(
                        orig,
                        (actual_x, actual_y),
                        (actual_x + fw, actual_y + fh),
                        (0, 0, 255), 1
                    )
                    
                    # LEVEL CALC
                    cy = actual_y + fh // 2
                    
                    assigned = False
                    for lv_idx in range(len(LEVEL_Y_POSITIONS)):
                        if cy < LEVEL_Y_DRAW[lv_idx]:
                            level_counts[5 - lv_idx] += 1
                            assigned = True
                            break
                    
                    if not assigned:
                        level_counts[0] += 1
        
        # เพิ่มแมลงที่ไม่เห็นให้กับ L5
        total_expected = TOTAL_FLIES_PER_TUBE[i] if i < len(TOTAL_FLIES_PER_TUBE) else 0
        
        if total_expected > 0 and tube_fly_count < total_expected:
            unseen_flies = total_expected - tube_fly_count
            level_counts[5] += unseen_flies
            print(f"Tube {i+1}: Detected {tube_fly_count} flies, Expected {total_expected} flies")
            print(f"  -> Added {unseen_flies} unseen flies to L5")
        
        # draw level lines & counts
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

    # DRAW TABLE
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

    # DRAW SCORE TABLE
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
    
    return orig, full_threshold_img

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
    # กำหนด path ของโฟลเดอร์ที่เก็บรูปภาพ
    folder_path = "D:/med_project/frames_by_time2/"
    
    # หารูปภาพทั้งหมดในโฟลเดอร์ (รองรับ .jpg, .jpeg, .png)
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
        text = f"Image {current_index + 1}/{len(image_files)} - Press 'N' for next, 'P' for previous, 'Q' to quit"
        cv2.putText(display_img, text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Fly Counter - Navigate with N/P/Q", display_img)
        
        # แสดงผล - ภาพ Threshold (ขาวดำ) ที่ใช้หา contours
        display_threshold = resize_image(detection_img, 70)
        
        # เพิ่มข้อความบนภาพ threshold
        text_threshold = f"Threshold Image - White = Potential fly detection areas"
        cv2.putText(display_threshold, text_threshold,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        cv2.imshow("Threshold Image (after preprocessing)", display_threshold)
        
        # รอการกดปุ่ม
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # กด 'N' หรือ 'n' ไปรูปถัดไป
            if key == ord('n') or key == ord('N'):
                current_index += 1
                if current_index >= len(image_files):
                    print("\nถึงรูปสุดท้ายแล้ว")
                    cv2.destroyAllWindows()
                    return
                break
            
            # กด 'P' หรือ 'p' กลับรูปก่อนหน้า
            elif key == ord('p') or key == ord('P'):
                if current_index > 0:
                    current_index -= 1
                    break
                else:
                    print("\nอยู่ที่รูปแรกแล้ว")
            
            # กด 'Q' หรือ 'q' ออกจากโปรแกรม
            elif key == ord('q') or key == ord('Q'):
                print("\nออกจากโปรแกรม")
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
