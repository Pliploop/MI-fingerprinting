import torch.nn as nn


class SampleCNN(nn.Module):
    def __init__(self,
                 strides=[3, 3, 3, 3, 3, 3, 3, 3, 3]):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]
        self.embed_dim = 512
        self.sr = 22050
        self.n_samples = int(2.7 * self.sr)

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

    def forward(self, x):
        out = self.sequential(x)
        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        return out
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()