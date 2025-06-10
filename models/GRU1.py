import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,       # feature_dim, 예: 114
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3000,
        dropout: float = 0.0,        # inter-layer dropout
        recurrent_dropout: float = 0.5,  # 논문에서는 GRU 내부 recurrent dropout
    ):
        super().__init__()
        # PyTorch GRU는 recurrent_dropout을 직접 지원하지 않으므로,
        # 논문 재현을 위해 WeightDrop 같은 기법을 써야 하지만,
        # 여기서는 inter-layer dropout에만 dropout 파라미터를 사용합니다.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        # classifier
        self.fc = nn.Linear(hidden_dim, num_classes)

        # optional: recurrent dropout via Dropout on hidden state
        self.rec_dropout = nn.Dropout(recurrent_dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (batch, seq_len, input_dim)
        lengths: (batch,) 실제 시퀀스 길이 (zero-padding 전 길이)
        """
        # 1) pack (패딩 구간을 GRU가 보지 않도록)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # 2) GRU 통과
        packed_out, h_n = self.gru(packed)
        # h_n: (num_layers, batch, hidden_dim)

        # 3) recurrent dropout on final hidden layer
        #    논문의 recurrent dropout을 간단히 흉내
        last_h = h_n[-1]                    # (batch, hidden_dim)
        last_h = self.rec_dropout(last_h)

        # 4) fully-connected
        logits = self.fc(last_h)            # (batch, num_classes)
        return logits