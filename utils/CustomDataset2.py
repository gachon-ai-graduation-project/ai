import numpy as np
import torch
from torch.utils.data import Dataset

# 1) 데이터셋 클래스
class SignLanguageDataset(Dataset):
    def __init__(self, file_list, max_seq_len=150, keypoint_dim=21):
        self.file_list = file_list
        self.max_seq_len = max_seq_len
        self.keypoint_dim = keypoint_dim
        self.data = []
        self.labels = []
        self.len = 0
        self.load_data()

    def load_data(self):
        for npz_path in self.file_list:
            data = np.load(npz_path, allow_pickle=True)
            keypoints = data['keypoints']    # (time_steps, feature_dim)
            label = data['label'].item()

            # 1) 빈 배열 또는 차원 오류 체크
            if keypoints.size == 0:
                print(f"⚠️Skipping {npz_path}: keypoints is empty.")
                continue
            if keypoints.ndim != 2:
                print(f"⚠️Skipping {npz_path}: expected 2D but got {keypoints.ndim}D, shape={keypoints.shape}")
                continue
            if keypoints.shape[0] < 10:
                print(f"⚠️Skipping {npz_path}: too short, shape={keypoints.shape}")
                continue
            
            # padding or trimming
            # 시퀀스 길이(time_steps)가 최대 길이보다 크면 자르고, 작으면 0으로 패딩해서 길이를 맞춤
            if keypoints.shape[0] > self.max_seq_len:
                keypoints = keypoints[:self.max_seq_len]
            else:
                # print(f"⚠️Padding {npz_path}: keypoints shape={keypoints.shape}")
                pad_len = self.max_seq_len - keypoints.shape[0]
                pad_array = np.zeros((pad_len, self.keypoint_dim), dtype=np.float32)
                keypoints = np.vstack([keypoints, pad_array])
            self.data.append(keypoints.astype(np.float32))
            self.labels.append(label)
            self.len = keypoints.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        len = self.len
        return torch.tensor(x), torch.tensor(y, dtype=torch.long), torch.tensor(len, dtype=torch.long)
