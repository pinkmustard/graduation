import torch.nn as nn
import torchvision
import torchvision.models as models

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # 사전 훈련된 ResNet50 모델 불러오기
        self.cnn = models.resnet50(pretrained=True).cuda()

        # 모델의 모든 파라미터를 미분 가능하게 설정
        for param in self.cnn.parameters():
            param.requires_grad = True

        # ResNet의 마지막 선형 레이어(fc)를 사용자 정의 레이어로 교체
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 4)  # 최종 출력 클래스가 4개
        )

    def forward(self, img):
        output = self.cnn(img)
        return output
