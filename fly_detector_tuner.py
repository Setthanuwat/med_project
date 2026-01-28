import cv2
import numpy as np
import os
import glob

# =========================
# GLOBAL VARIABLES
# =========================
current_params = {
    'blur_kernel': 3,           # ขนาด Gaussian Blur (1-15, คี่)
    'adaptive_block': 17,       # Block size สำหรับ Adaptive Threshold (3-99, คี่)
    'adaptive_c': 7,            # ค่า C สำหรับ Adaptive Threshold (0-20)
    'morph_size': 3,            # ขนาด kernel สำหรับ Morphology (1-15, คี่)
    'morph_iter': 2,            # จำนวนรอบ Morphology (0-10)
    'min_area': 20,             # พื้นที่ต่ำสุดของแมลงวัน (10-100)
    'max_area': 200,            # พื้นที่สูงสุดของแมลงวัน (100-500)
    'hsv_v_min': 160,           # HSV V min สำหรับ white mask (0-255)
    'hsv_v_max': 255,           # HSV V max สำหรับ white mask (0-255)
    'hsv_s_max': 50,            # HSV S max สำหรับ white mask (0-255)
}

current_image = None
current_index = 0
image_files = []
tube_configs = [
    {'offset_from_left': 420,  'width': 180, 'top_offset': 40,  'bottom_offset': 0},
    {'offset_from_left': 620,  'width': 200, 'top_offset': 40, 'bottom_offset': 0},
    {'offset_from_left': 850,  'width': 190, 'top_offset': 40, 'bottom_offset': 0},
    {'offset_from_left': 1110, 'width': 185, 'top_offset': 40, 'bottom_offset': 0},
    {'offset_from_left': 1350, 'width': 195, 'top_offset': 40, 'bottom_offset': 0},
]

# =========================
# CALLBACK FUNCTIONS
# =========================
def nothing(x):
    """Callback function สำหรับ trackbar"""
    pass

def update_blur(val):
    # ต้องเป็นเลขคี่
    if val % 2 == 0:
        val = max(1, val - 1)
    current_params['blur_kernel'] = val
    cv2.setTrackbarPos('Blur Kernel', 'Controls', val)
    process_and_display()

def update_adaptive_block(val):
    # ต้องเป็นเลขคี่และมากกว่า 1
    if val < 3:
        val = 3
    if val % 2 == 0:
        val = val + 1
    current_params['adaptive_block'] = val
    cv2.setTrackbarPos('Adaptive Block', 'Controls', val)
    process_and_display()

def update_adaptive_c(val):
    current_params['adaptive_c'] = val
    process_and_display()

def update_morph_size(val):
    # ต้องเป็นเลขคี่
    if val % 2 == 0:
        val = max(1, val - 1)
    current_params['morph_size'] = val
    cv2.setTrackbarPos('Morph Size', 'Controls', val)
    process_and_display()

def update_morph_iter(val):
    current_params['morph_iter'] = val
    process_and_display()

def update_min_area(val):
    current_params['min_area'] = val
    process_and_display()

def update_max_area(val):
    current_params['max_area'] = val
    process_and_display()

def update_hsv_v_min(val):
    current_params['hsv_v_min'] = val
    process_and_display()

def update_hsv_v_max(val):
    current_params['hsv_v_max'] = val
    process_and_display()

def update_hsv_s_max(val):
    current_params['hsv_s_max'] = val
    process_and_display()

# =========================
# PROCESS IMAGE FUNCTION
# =========================
def process_and_display():
    """ประมวลผลภาพด้วยค่า parameters ปัจจุบัน"""
    if current_image is None:
        return
    
    img = current_image.copy()
    h_img, w_img = img.shape[:2]
    
    # สร้างภาพผลลัพธ์
    result = img.copy()
    full_threshold = np.zeros((h_img, w_img), dtype=np.uint8)
    
    roi_y_start = 0
    roi_y_end = 1000
    
    total_flies = 0
    
    # วนลูปแต่ละหลอด
    for i, cfg in enumerate(tube_configs):
        x_start = cfg['offset_from_left']
        x_end = x_start + cfg['width']
        tube_y_start = roi_y_start + cfg['top_offset']
        tube_y_end = roi_y_end - cfg['bottom_offset']
        
        # วาดกรอบหลอด
        cv2.rectangle(result, (x_start, tube_y_start), (x_end, tube_y_end), (255, 255, 0), 2)
        
        # Extract ROI
        tube_roi = img[tube_y_start:tube_y_end, x_start:x_end]
        
        if tube_roi.size > 0:
            # Apply preprocessing
            gray_roi = cv2.cvtColor(tube_roi, cv2.COLOR_BGR2GRAY)
            
            # Gaussian Blur
            blur_k = current_params['blur_kernel']
            blur_roi = cv2.GaussianBlur(gray_roi, (blur_k, blur_k), 0)
            
            # Adaptive Threshold
            block_size = current_params['adaptive_block']
            c_val = current_params['adaptive_c']
            th_roi = cv2.adaptiveThreshold(
                blur_roi, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size, c_val
            )
            
            # Morphology
            morph_k = current_params['morph_size']
            morph_i = current_params['morph_iter']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
            th_clean = cv2.morphologyEx(th_roi, cv2.MORPH_OPEN, kernel, iterations=morph_i)
            
            # วาง threshold image
            full_threshold[tube_y_start:tube_y_end, x_start:x_end] = th_clean
            
            # Find contours
            contours_fly, _ = cv2.findContours(
                th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2:]
            
            tube_fly_count = 0
            min_area = current_params['min_area']
            max_area = current_params['max_area']
            
            for cnt in contours_fly:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    tube_fly_count += 1
                    fx, fy, fw, fh = cv2.boundingRect(cnt)
                    actual_x = x_start + fx
                    actual_y = tube_y_start + fy
                    
                    # วาดกรอบสีแดง
                    cv2.rectangle(
                        result,
                        (actual_x, actual_y),
                        (actual_x + fw, actual_y + fh),
                        (0, 0, 255), 2
                    )
            
            total_flies += tube_fly_count
            
            # แสดงจำนวนบนหลอด
            cv2.putText(
                result, f"T{i+1}: {tube_fly_count}",
                (x_start + 10, tube_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
    
    # แสดงข้อมูล parameters บนภาพ
    info_y = 30
    line_height = 25
    cv2.putText(result, f"Total Flies: {total_flies}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    info_y += line_height
    cv2.putText(result, f"Blur: {current_params['blur_kernel']}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    info_y += line_height
    cv2.putText(result, f"Block: {current_params['adaptive_block']}, C: {current_params['adaptive_c']}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    info_y += line_height
    cv2.putText(result, f"Morph: {current_params['morph_size']}x{current_params['morph_iter']}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    info_y += line_height
    cv2.putText(result, f"Area: {current_params['min_area']}-{current_params['max_area']}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # แสดงชื่อไฟล์
    cv2.putText(result, f"Image: {os.path.basename(image_files[current_index])} ({current_index+1}/{len(image_files)})",
                (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Resize สำหรับแสดงผล
    scale = 0.6
    result_resized = cv2.resize(result, None, fx=scale, fy=scale)
    threshold_resized = cv2.resize(full_threshold, None, fx=scale, fy=scale)
    
    # แสดงผล
    cv2.imshow('Result with Detections', result_resized)
    cv2.imshow('Threshold Image', threshold_resized)

# =========================
# LOAD IMAGE FUNCTION
# =========================
def load_image(index):
    """โหลดภาพใหม่"""
    global current_image, current_index
    
    if 0 <= index < len(image_files):
        current_index = index
        img = cv2.imread(image_files[index])
        if img is not None:
            current_image = img
            process_and_display()
            print(f"\nLoaded: {os.path.basename(image_files[index])} ({index+1}/{len(image_files)})")
            return True
    return False

# =========================
# SAVE PARAMETERS FUNCTION
# =========================
def save_parameters():
    """บันทึกค่า parameters ลงไฟล์"""
    filename = "best_parameters.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("BEST PARAMETERS FOR FLY DETECTION\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("# Preprocessing Parameters\n")
        f.write(f"BLUR_KERNEL = {current_params['blur_kernel']}\n")
        f.write(f"ADAPTIVE_BLOCK_SIZE = {current_params['adaptive_block']}\n")
        f.write(f"ADAPTIVE_C = {current_params['adaptive_c']}\n")
        f.write(f"MORPH_KERNEL_SIZE = {current_params['morph_size']}\n")
        f.write(f"MORPH_ITERATIONS = {current_params['morph_iter']}\n\n")
        
        f.write("# Detection Parameters\n")
        f.write(f"MIN_AREA = {current_params['min_area']}\n")
        f.write(f"MAX_AREA = {current_params['max_area']}\n\n")
        
        f.write("# HSV Parameters (if used)\n")
        f.write(f"HSV_V_MIN = {current_params['hsv_v_min']}\n")
        f.write(f"HSV_V_MAX = {current_params['hsv_v_max']}\n")
        f.write(f"HSV_S_MAX = {current_params['hsv_s_max']}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("Copy these values to your main script\n")
        f.write("=" * 50 + "\n")
    
    print(f"\n✓ Parameters saved to: {filename}")

# =========================
# MAIN FUNCTION
# =========================
def main():
    global image_files
    
    # กำหนด path ของโฟลเดอร์
    folder_path = "D:/med_project/frames_by_time/"
    
    # หารูปภาพทั้งหมด
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(folder_path, "*.jpeg"))) + \
                  sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    if len(image_files) == 0:
        print(f"ไม่พบรูปภาพในโฟลเดอร์: {folder_path}")
        return
    
    print(f"พบรูปภาพทั้งหมด {len(image_files)} รูป")
    
    # สร้างหน้าต่าง
    # สร้างหน้าต่าง
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result with Detections', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Threshold Image', cv2.WINDOW_NORMAL)

    # กำหนดขนาดหน้าต่าง (กว้าง, สูง)
    cv2.resizeWindow('Controls', 600, 400)
    cv2.resizeWindow('Result with Detections', 1200, 800)
    cv2.resizeWindow('Threshold Image', 800, 600)

    
    # สร้าง trackbars
    cv2.createTrackbar('Blur Kernel', 'Controls', current_params['blur_kernel'], 15, update_blur)
    cv2.createTrackbar('Adaptive Block', 'Controls', current_params['adaptive_block'], 99, update_adaptive_block)
    cv2.createTrackbar('Adaptive C', 'Controls', current_params['adaptive_c'], 20, update_adaptive_c)
    cv2.createTrackbar('Morph Size', 'Controls', current_params['morph_size'], 15, update_morph_size)
    cv2.createTrackbar('Morph Iter', 'Controls', current_params['morph_iter'], 10, update_morph_iter)
    cv2.createTrackbar('Min Area', 'Controls', current_params['min_area'], 100, update_min_area)
    cv2.createTrackbar('Max Area', 'Controls', current_params['max_area'], 500, update_max_area)
    
    # โหลดภาพแรก
    load_image(0)
    
    print("\n" + "=" * 60)
    print("คำแนะนำการใช้งาน:")
    print("=" * 60)
    print("- ปรับค่าต่างๆ ด้วย Trackbar ในหน้าต่าง 'Controls'")
    print("- กด 'N' หรือ 'n' = ไปรูปถัดไป")
    print("- กด 'P' หรือ 'p' = กลับรูปก่อนหน้า")
    print("- กด 'S' หรือ 's' = บันทึกค่า parameters ปัจจุบัน")
    print("- กด 'R' หรือ 'r' = รีเซ็ตค่าเป็นค่าเริ่มต้น")
    print("- กด 'Q' หรือ 'q' = ออกจากโปรแกรม")
    print("=" * 60)
    print("\nเริ่มปรับค่าได้เลย!")
    
    # Main loop
    while True:
        key = cv2.waitKey(50) & 0xFF
        
        # Next image
        if key == ord('n') or key == ord('N'):
            if current_index < len(image_files) - 1:
                load_image(current_index + 1)
            else:
                print("ถึงรูปสุดท้ายแล้ว")
        
        # Previous image
        elif key == ord('p') or key == ord('P'):
            if current_index > 0:
                load_image(current_index - 1)
            else:
                print("อยู่ที่รูปแรกแล้ว")
        
        # Save parameters
        elif key == ord('s') or key == ord('S'):
            save_parameters()
        
        # Reset parameters
        elif key == ord('r') or key == ord('R'):
            current_params.update({
                'blur_kernel': 3,
                'adaptive_block': 17,
                'adaptive_c': 7,
                'morph_size': 3,
                'morph_iter': 2,
                'min_area': 20,
                'max_area': 200,
                'hsv_v_min': 160,
                'hsv_v_max': 255,
                'hsv_s_max': 50,
            })
            
            # อัพเดท trackbars
            cv2.setTrackbarPos('Blur Kernel', 'Controls', current_params['blur_kernel'])
            cv2.setTrackbarPos('Adaptive Block', 'Controls', current_params['adaptive_block'])
            cv2.setTrackbarPos('Adaptive C', 'Controls', current_params['adaptive_c'])
            cv2.setTrackbarPos('Morph Size', 'Controls', current_params['morph_size'])
            cv2.setTrackbarPos('Morph Iter', 'Controls', current_params['morph_iter'])
            cv2.setTrackbarPos('Min Area', 'Controls', current_params['min_area'])
            cv2.setTrackbarPos('Max Area', 'Controls', current_params['max_area'])
            
            process_and_display()
            print("\n✓ รีเซ็ตค่าเป็นค่าเริ่มต้นแล้ว")
        
        # Quit
        elif key == ord('q') or key == ord('Q'):
            print("\nออกจากโปรแกรม")
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
