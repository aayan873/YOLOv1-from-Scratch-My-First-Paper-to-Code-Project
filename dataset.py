import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms


class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S = 7, B = 2, C = 20, transform = None ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S, self.B, self.C = S, B, C
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        image = Image.open(img_path).convert("RGB")

        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image, boxes = self.transform(image, boxes)
        else:
            image = transforms.ToTensor()(image)
        
        label_matrix = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for box in boxes:
            class_label, x, y, w, h = box

            i = int(min(self.S * y, self.S - 1e-6))
            j = int(min(self.S * x, self.S - 1e-6))
            x_cell = self.S * x - j
            y_cell = self.S * y - i
            width_cell, height_cell = w * self.S, h * self.S

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                label_matrix[i, j, self.C + 1:self.C+5] = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell], dtype=torch.float32
                )
                if int(class_label) < 20:
                    label_matrix[i, j, int(class_label)] = 1
                else:
                    print(f"Invalid class index {int(class_label)} found.")

        return image, label_matrix

