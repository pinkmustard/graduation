from flask import Flask, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from PIL import Image
from flask import Flask, request

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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'img/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB Max size
app.secret_key = 'supersecretkey'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/test', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # 사진 받는거 POST에 추가되면 연결 확인 코드 넣기
            # 디바이스 설정
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 모델 로드 및 디바이스로 이동
            model = torch.load('best.pt').to(device)
            model.eval()

            # 이미지 전처리
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 이미지 로드 및 전처리
            img = Image.open(f'img/{filename}')
            img = transform(img)
            img = img.unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스로 이동

            # 예측
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
            # return jsonify({"acne_level": predicted.item()})
            return f'acne_level: {predicted.item()}'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60017)
