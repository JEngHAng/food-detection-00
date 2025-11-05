from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    # เริ่มเทรนโมเดล
    model.train(
        data="food-detect-7/data.yaml",  # path ไปยัง dataset (ต้องมี train/val และชื่อ class)
        epochs=400,                      # จำนวนรอบการเทรน (300 พอเหมาะกับ dataset ~1.6k รูป)
        imgsz=640,                       # ขนาดภาพระหว่างเทรน
        batch=16,                        # จำนวนภาพต่อ batch (ถ้า VRAM < 8GB ใช้ 8 แทน)
        device="cuda",                   # ใช้ GPU ถ้ามี (cuda), ถ้าไม่มีจะใช้ cpu อัตโนมัติ
        name="food_detect_model",        # ชื่อโฟลเดอร์สำหรับบันทึกผลเทรน
        workers=4,                       # จำนวน thread สำหรับโหลดข้อมูล (มากขึ้นถ้า CPU แรง)
        amp=True,                        # ใช้ Automatic Mixed Precision เทรนเร็วและกินแรมน้อยลง
        augment=True,                    # เปิด data augmentation เพื่อเพิ่มความหลากหลายของข้อมูล
        lr0=0.002,                       # ค่าเริ่มต้น learning rate (ปรับขึ้นเล็กน้อยให้เรียนรู้เร็ว)
        lrf=0.01,                        # ค่า learning rate ปลายทางเมื่อเทรนใกล้จบ
        momentum=0.937,                  # ใช้ momentum ค่ามาตรฐาน YOLO (ช่วยให้การเรียนรู้เสถียร)
        weight_decay=0.0005,             # ป้องกัน overfitting โดยลดน้ำหนัก parameter ที่ไม่สำคัญ
        patience=50,                     # ถ้า val loss ไม่ดีขึ้นเกิน 50 รอบ จะหยุดเทรนอัตโนมัติ
        val=True,                        # ให้ validate หลังแต่ละ epoch เพื่อดู performance
        seed=42,                         # ตั้งค่า random seed ให้เทรนได้ผลซ้ำได้
        cos_lr=True,                     # ใช้ cosine learning rate schedule เพื่อการลดค่า lr อย่างนุ่มนวล
        cache=True                       # โหลด dataset ไว้ใน RAM เพื่อเพิ่มความเร็วในการเทรน
    )

    # เมื่อเทรนเสร็จจะได้ไฟล์ weights เช่น:
    # runs/train/food_detect_model_v2/weights/best.pt
    print("Training completed successfully. Check your weights folder!")

if __name__ == "__main__":
    main()
