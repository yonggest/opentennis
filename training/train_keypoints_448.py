"""
重新训练球场关键点模型，输入分辨率从 224x224 提升到 448x448。
用法：python train_keypoints_448.py
输出：models/keypoints_model_448.pth
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import json
import cv2
import numpy as np
from pathlib import Path

TARGET_SIZE = 448
BATCH_SIZE  = 8
EPOCHS      = 10
LR          = 1e-4
DATA_DIR    = Path(__file__).parent / "data"
MODEL_OUT   = Path(__file__).parent.parent / "models" / "keypoints_model_448.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class KeypointsDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = json.load(f)
        self.img_dir = DATA_DIR / "images"
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img  = cv2.imread(str(self.img_dir / f"{item['id']}.png"))
        h, w = img.shape[:2]
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = self.transform(img)

        kps = np.array(item['kps']).flatten().astype(np.float32)
        kps[::2]  *= TARGET_SIZE / w   # x
        kps[1::2] *= TARGET_SIZE / h   # y
        return img, kps


train_loader = DataLoader(
    KeypointsDataset(DATA_DIR / "data_train.json"),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    KeypointsDataset(DATA_DIR / "data_val.json"),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# 从现有模型 fine-tune（而不是从头训练）
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)
existing = Path(__file__).parent.parent / "models" / "keypoints_model.pth"
model.load_state_dict(torch.load(existing, map_location="cpu"))
print(f"已加载现有模型: {existing}")
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # ── 训练 ──
    model.train()
    train_loss = 0
    for i, (imgs, kps) in enumerate(train_loader):
        imgs, kps = imgs.to(device), kps.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), kps)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  batch {i}/{len(train_loader)}  loss={loss.item():.4f}")

    # ── 验证 ──
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, kps in val_loader:
            imgs, kps = imgs.to(device), kps.to(device)
            val_loss += criterion(model(imgs), kps).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ✓ 保存最佳模型 → {MODEL_OUT}  (val_loss={val_loss:.4f})")

print(f"\n训练完成，最佳 val_loss={best_val_loss:.4f}")
