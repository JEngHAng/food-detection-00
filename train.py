from ultralytics import YOLO

def main():
    # โหลดโมเดลพื้นฐาน (สามารถเปลี่ยนเป็น yolov8n.pt, yolov8s.pt, yolov8m.pt ฯลฯ)
    model = YOLO("yolov8n.pt")

    # เทรนโมเดลใหม่
    model.train(
        data="food-detect-5/data.yaml",   # path ไปยังไฟล์ dataset ของคุณ
        epochs=50,                  # จำนวนรอบการเทรน
        imgsz=640,                  # ขนาดภาพ
        batch=8,                    # batch size (ปรับตาม VRAM)
        device=0,                   # ใช้ GPU หมายเลข 0
        name="food_detect_model",   # ชื่อโฟลเดอร์ที่บันทึกผล
        workers=4                   # จำนวน worker ในการโหลดข้อมูล
    )

    # บันทึกโมเดล
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
