from flask import Flask, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from PIL import Image
import requests

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
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 4),
        )

    def forward(self, img):
        output = self.cnn(img)
        return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'img/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기
app.secret_key = os.environ.get('SECRET_KEY', 'you-should-change-this')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/test', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return process_image(filename)
    return jsonify({'error': 'File not processed'}), 500

def process_image(filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('best.pt').to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = transform(img)
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    try:
        url = 'http://52.79.237.164:3000/user/skin/classification'
        files = {'photoFile': open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')}
        data = {
            'userId': 'park123123',
            'aiType': "트러블 유형 분석",
            'troubleType': f'{predicted.item()}'
        }
        response = requests.post(url, files=files, data=data)
        # 응답 데이터를 JSON 형식으로 파싱
        json_data = response.json()

        # 각 필드의 값을 출력
        record_id = json_data['recordId']
        success = json_data['success']
        message = json_data['message']
        property_value = json_data['property']

        return jsonify({'recordId': record_id,'acneLevel': predicted.item()})
        # return f'Record ID: {record_id}, Success: {success}, Message: {message}, Property: {property_value}'
    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60017)
