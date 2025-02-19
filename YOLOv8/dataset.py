import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        :param image_dir: مسیر فولدر تصاویر
        :param label_dir: مسیر فولدر لیبل‌ها
        :param transform: هر نوع ترنسفورماتی که بخوای می‌تونی به مدل بدی (مثل تغییر اندازه یا نرمال‌سازی)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]  # یا فرمت دیگری
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        این متد برای بارگذاری تصویر و لیبل در یک ایندکس خاص استفاده میشه
        :param idx: ایندکس تصویر مورد نظر
        :return: تصویر و لیبل‌های نرمال‌شده
        """
        # بارگذاری تصویر
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")  # تصویر رو باز می‌کنیم

        # بارگذاری لیبل
        label_name = image_name.replace(".jpg", ".txt")  # فرض کردیم پسوند txt برای لیبل‌ها داریم
        label_path = os.path.join(self.label_dir, label_name)

        # خواندن مختصات باندینگ باکس‌ها از فایل
        boxes = []
        classes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                class_id = int(line[0])
                x_center = float(line[1])
                y_center = float(line[2])
                width = float(line[3])
                height = float(line[4])
                boxes.append([x_center, y_center, width, height])
                classes.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, boxes, classes
