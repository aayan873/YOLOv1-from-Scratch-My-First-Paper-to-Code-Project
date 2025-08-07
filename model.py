import torch.nn as nn

# S = dimension of grid (S x S)
# B = No. of Bounding Boxes
# C = Labelled Classes

# architecture -
#     (7, 64, 2, 3),
#     "M",
#     (3, 192, 1, 1),
#     "M",
#     (1, 128, 1, 0),
#     (3, 256, 1, 1),
#     (1, 256, 1, 0),
#     (3, 512, 1, 1),
#     "M",
#     [(1, 256, 1, 0), (3, 512, 1, 1), 4],
#     (1, 512, 1, 0),
#     (3, 1024, 1, 1),
#     "M",
#     [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
#     (3, 1024, 1, 1),
#     (3, 1024, 2, 1),
#     (3, 1024, 1, 1),
#     (3, 1024, 1, 1),


class YOLOv1(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        
        super(YOLOv1, self).__init__()

        self.S, self.B, self.C = S, B, C

        def conv_block(ins, out, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(ins, out, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out),
                nn.LeakyReLU(0.1)
            )
        
        self.features = nn.Sequential(
            conv_block(3, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),

            conv_block(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(192, 128, 1, 1, 0),
            conv_block(128, 256, 3, 1, 1),
            conv_block(256, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            *[nn.Sequential(
                conv_block(512, 256, 1, 1, 0),
                conv_block(256, 512, 3, 1, 1)
            ) for _ in range(4)],

            conv_block(512, 512, 1, 1, 0),
            conv_block(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            *[nn.Sequential(
                conv_block(1024, 512, 1, 1, 0),
                conv_block(512, 1024, 3, 1, 1)
            ) for _ in range(2)],

            conv_block(1024, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 2, 1),
            
            conv_block(1024, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 1, 1)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, (S * S * (B * 5 + C)))
        )

    def forward(self, X):
        X = self.features(X)
        return self.fc_layers(X).reshape(-1, self.S, self.S, self.C + self.B * 5)
