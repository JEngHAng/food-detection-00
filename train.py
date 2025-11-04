from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    # เทรนโมเดลใหม่
    model.train(
        data="food-detect-7/data.yaml",   # path ไปยังไฟล์ dataset
        epochs=200,                  # จำนวนรอบการเทรน
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
