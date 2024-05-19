from flask import Flask, request, jsonify
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras_cv
from werkzeug.utils import secure_filename
import os
import requests  # 추가: 외부 서버와 통신하기 위한 라이브러리

def load_and_preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 정규화
    img = tf.image.resize(img, [640, 640])  # 모델 입력 크기에 맞춤
    return img.numpy() # Eager Tensor를 NumPy 배열로 변환

def model_input(model, img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (640, 640))  # 모델에 맞는 크기로 조정
    img = tf.expand_dims(img, axis=0)  # 배치 차원 추가
    y_pred = model.predict(img, verbose=0)
    return y_pred

def visualize_predictions(img, result, save_path):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    boxes = result['boxes'][0]  # 첫 번째 배치의 박스
    confidences = result['confidence'][0]  # 첫 번째 배치의 신뢰도
    class_ids = result['classes'][0]  # 첫 번째 배치의 클래스 ID

    boxes_total = 0
    for box, score, class_id in zip(boxes, confidences, class_ids):
        if score == -1:
            continue  # 유효하지 않은 박스는 건너뜁니다.
        boxes_total += 1
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label = f'Score: {score:.2f}'
        ax.text(x1, y1, label, color='white', verticalalignment='top', bbox={'color': 'red', 'pad': 0})
    print("the number of acnes:", boxes_total)
    plt.axis('off')  # 축 표시 제거
    plt.savefig(save_path, bbox_inches='tight')  # 이미지 파일로 저장
    # plt.show()
    plt.close(fig)
    return boxes_total



def predict_detection():
    file = request.files['file']
    userId = request.form['userId']
    class_mapping = {0: 'Acne'} # 정수 인덱스르르 클래스 이름에 매핑, 딕셔너리 이용. 0 -> 여드름으로 인식
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone", include_rescaling = True)
    # yolov8 모델설정. 생성된 백본과 추가 설정을 사용해 객체 탐지 모델 초기화. 클래스 수는 class_mapping 딕셔너리 길이를 기반. 바운딩 박스 포맷은 xyxy
    YOLOV8_model = keras_cv.models.YOLOV8Detector(num_classes=len(class_mapping), bounding_box_format = "xyxy", backbone = backbone, fpn_depth = 5)
    YOLOV8_model.load_weights('/home/t24117/last/acne_detection/model/yolov8_acne_detection_large.h5')
    if file:
        img_path = secure_filename(f'{userId}.jpg')
        file.save(img_path)
        img = load_and_preprocess_image(img_path)
        y = model_input(YOLOV8_model, img_path)
        save_path = secure_filename(f'{userId}_output.jpg')
        boxes_total = visualize_predictions(img, y, save_path)
        try:
            url = 'http://52.79.237.164:3000/user/skin/detection/save'
            files = {'photoFile': open(save_path, 'rb')}
            data = {
                'userId': userId,
                'aiType': "AI 호전도 분석",
                'troubleTotal': boxes_total
            }
            response = requests.post(url, files=files, data=data)
            json_data = response.json()
            os.remove(img_path)  # 입력 이미지 삭제
            os.remove(save_path)  # 출력 이미지 삭제
            return jsonify(json_data)
        except requests.RequestException as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'message': 'No file provided'}), 400

