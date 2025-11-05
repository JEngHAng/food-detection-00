import os

def clean_labels(path, max_class=3):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".txt"):
                file_path = os.path.join(root, f)
                with open(file_path, "r") as file:
                    lines = file.readlines()
                new_lines = [l for l in lines if int(l.split()[0]) <= max_class]
                if len(new_lines) != len(lines):
                    with open(file_path, "w") as file:
                        file.writelines(new_lines)
                    print(f"✅ Cleaned: {file_path}")

# เรียกใช้งาน
clean_labels("food-detect-7/train/labels")
clean_labels("food-detect-7/valid/labels")
