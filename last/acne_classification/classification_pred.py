
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from PIL import Image
from flask import Flask, request, jsonify
import requests
import os

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
    file = request.files['file']
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
    if file:
        img_path = f'/home/t24117/last/acne_classification/img/{userId}.jpg'
        file.save(img_path)
        # 이미지 로드 및 전처리
        img = Image.open(img_path)
        img = transform(img)
        img = img.unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스로 이동

        # 예측
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
        try:
            url = 'http://52.79.237.164:3000/user/skin/classification/save'
            files = {'photoFile': open(img_path, 'rb')}
            data = {
                'userId': userId,
                'aiType': "AI 트러블 분석",
                'troubleType': f'{predicted.item()}'
            }
            response = requests.post(url, files=files, data=data)
            json_data = response.json()
            os.remove(img_path)  # 입력 이미지 삭제
            record_id = json_data['recordId']
            return jsonify({'recordId': record_id,'acneLevel': predicted.item()})
        except requests.RequestException as e:
            return jsonify({'error': str(e)}), 500
    # 예측된 레이블 출력
    else:
        return jsonify({'message': 'No file provided'}), 400