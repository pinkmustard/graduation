
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from PIL import Image
from flask import Flask, request, jsonify

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.cnn = torchvision.models.efficientnet_v2_m(pretrained=True).cuda()
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.cnn.classifier = nn.Sequential(

        nn.Linear(self.cnn.classifier[1].in_features, 512),
        nn.Dropout(p=0.2),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
            nn.Linear(64, 4),
       )

    def forward(self, img):
        output = self.cnn(img)
        return output

def predict_classification():
    userId = request.form['userId']
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드 및 디바이스로 이동
    model = torch.load('/home/t24117/last/acne_classification/model/best.pt').to(device)
    model.eval()

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 이미지 로드 및 전처리
    img = Image.open('/home/t24117/last/acne_classification/exam.jpeg')
    img = transform(img)
    img = img.unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스로 이동

    # 예측
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # 예측된 레이블 출력

    return jsonify({"userId": userId, "acne_level": predicted.item()})