# iknet_weighted_training.py (修复版，防止 loss 为 NaN，并丢弃无效样本，自动拆分左右手输出)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ====== 可配置参数 ======
LEFT_CSV = "left_arm_ik_dataset.csv"
RIGHT_CSV = "right_arm_ik_dataset.csv"
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-4  # 更稳健
POSITION_WEIGHT = 1.0
ELBOW_WEIGHT = 0.8
WRIST_WEIGHT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 数据集定义 ======
class IKDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== 网络定义 ======
class IKNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IKNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ====== 主训练流程 ======
def main():
    # 读取数据
    df_left = pd.read_csv(LEFT_CSV)
    df_right = pd.read_csv(RIGHT_CSV)
    df_all = pd.concat([df_left, df_right], ignore_index=True)

    # 拆分左右手数据，并移除无效关节列
    left_joint_cols = [col for col in df_all.columns if col.startswith("left_")]
    right_joint_cols = [col for col in df_all.columns if col.startswith("right_")]

    df_left = df_all[df_all["arm_flag"] == 0].copy()
    df_left = df_left.drop(columns=right_joint_cols)

    df_right = df_all[df_all["arm_flag"] == 1].copy()
    df_right = df_right.drop(columns=left_joint_cols)

    # 重命名输出列为统一格式
    df_left.columns = list(df_left.columns[:8]) + [f"joint_{i}" for i in range(len(df_left.columns) - 8)]
    df_right.columns = list(df_right.columns[:8]) + [f"joint_{i}" for i in range(len(df_right.columns) - 8)]

    df = pd.concat([df_left, df_right], ignore_index=True).dropna().reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    input_cols = df.columns[0:8]  # arm_flag + xyz + quat
    output_cols = df.columns[8:]  # joint angles

    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    # 对输出进行归一化（防止爆炸）
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y = scaler_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_loader = DataLoader(IKDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(IKDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = IKNet(input_dim=8, output_dim=y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 构造 loss 权重
    n_joints = y.shape[1]
    if n_joints == 6:
        weights = torch.tensor([
            POSITION_WEIGHT, POSITION_WEIGHT, ELBOW_WEIGHT,
            WRIST_WEIGHT, WRIST_WEIGHT, WRIST_WEIGHT
        ], device=device)
    else:
        weights = torch.ones(n_joints, device=device)  # fallback

    # ====== 训练循环 ======
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = nn.SmoothL1Loss(reduction='none')(pred, y_batch)
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()

            total_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = nn.SmoothL1Loss(reduction='none')(pred, y_batch)
                loss = (loss * weights).mean()
                val_loss += loss.item()

        print(f"[Epoch {epoch+1:02d}] Train Loss: {total_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(val_loader):.5f}")

    torch.save({
        "model": model.state_dict(),
        "scaler_y": scaler_y,
    }, "iknet_model.pt")
    print("✅ 模型已保存到 iknet_model.pt")

if __name__ == "__main__":
    main()
