import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import *
import keras_cv

from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/detection_train', methods=['POST'])
def detection_pred():
    # 요청 데이터를 JSON 형태로 추출
    data = request.get_json()
    
    # 데이터에서 각 인자를 추출
    input_learning_rate = data.get('learningRate', None)
    input_weight_decay = data.get('weightDecay', None)
    input_epochs = data.get('Epochs', None)
    input_patience = data.get('Patience', None)
    
    BATCH_SIZE = 16
    AUTO = tf.data.AUTOTUNE # 데이터 로딩과 전처리 과정을 최적화하는데 사용하는 설정, 실행 환경에 맞게 자동으로 적절한 스레드 수나 버퍼 크기를 설정
    # 즉, 데이터 파이프라인의 성능을 자동으로 튜닝할 수 있어, 전체 모델의 훈련 속도와 효율성을 개선하는 데 도움을 줌.

    # 이미지 파일의 경로와 텍스트 파일의 경로를 입력받아 텍스트 파일에 있는 주석을 읽어 해당 이미지 내 객체들의 바운딩 박스와 클래스 라벨을 추출하는 역할
    def parse_txt_annot(img_path, txt_path): # 이미지 파일 경로, 텍스트 파일 경로
        img = cv2.imread(img_path) # 이미지 로드
        w = int(img.shape[0]) # 이미지 너비와 높이 추출
        h = int(img.shape[1])

        file_label = open(txt_path, "r") # 텍스트 파일을 읽기 모드로 켜고
        lines = file_label.read().split('\n')  # 파일 전체 내용을 읽고 개행 문자를 구분자로 사용하여 라인별로 분리

        boxes = [] # 객체의 위치를 나타내는 좌표 저장
        classes = [] # 각 객체의 클래스 저장

        if lines[0] == '': # 첫번째 줄이 비어 있으면 아무 객체가 없어 이미지 경로와 빈 클래스 및 박스 리스트 반환
            return img_path, classes, boxes
        else:
            for i in range(0, int(len(lines))):
                objbud=lines[i].split(' ')
                class_ = int(objbud[0])
                # 각 라인을 순회하며 객체 데이터를 처리, 각 라인에서 클래스 번호와 바운딩 박스 좌표, 너비, 높이 추출
                x1 = float(objbud[1])
                y1 = float(objbud[2])
                w1 = float(objbud[3])
                h1 = float(objbud[4])
                # 바운딩 박스 계산, 추출된 값을 사용해 바운딩 박스의 실제 픽셀 좌표 계산
                xmin = int((x1*w) - (w1*w)/2.0) # 좌측 상단 모서리
                ymin = int((y1*h) - (h1*h)/2.0)
                xmax = int((x1*w) + (w1*w)/2.0) # 우측 하단 모서리
                ymax = int((y1*h) + (h1*h)/2.0)

                boxes.append([xmin ,ymin ,xmax ,ymax])
                classes.append(class_)
        # 결과 반환, 이미지 경로, 클래스 리스트, 바운딩 박스 리스트 반환.
        return img_path, classes, boxes


    # a function for creating file paths list
    # 주어진 디렉터리 경로에 있는 모든 파일들의 전체 경로 목록을 생성
    def create_paths_list(path): # path 파일들이 저장된 디렉터리 경로
        full_path = [] # 디렉터리 내 모든 파일의 전체 경로를 저장
        images = sorted(os.listdir(path)) # 알파벳순으로 정렬, 주어진 경로에 있는 파일들의 목록을 가져옴

        for i in images:
            full_path.append(os.path.join(path, i)) # 전체 파일 경로를 생성, 리스트에 추가

        return full_path # 생성된 전체 경로를 반환


    class_ids = ['Acne'] # 클래스 식별자 리스트
    class_mapping = {0: 'Acne'} # 정수 인덱스르르 클래스 이름에 매핑, 딕셔너리 이용. 0 -> 여드름으로 인식

    # 이미지 파일 경로와 주석 파일 경로를 입력 받아 각 이미지 파일에 대한 주석 정보를 파싱하고 텐서플로의 형태로 변환하는 함수
    # 딕셔너리 형식의 파일 형태를 만드는 함수
    def creating_files(img_files_paths, annot_files_paths):

        img_files = create_paths_list(img_files_paths) # 이미지 파일들이 저장된 디렉터리 경로, 앞서 언급된 함수를 호출해 전체 경로 리스트 생성
        annot_files = create_paths_list(annot_files_paths) # 주석 파일이 저장된 디렉터리 경로

        image_paths = [] # 이미지 경로
        bbox = [] # 바운딩 박스
        classes = [] # 클래스 저장

        for i in range(0,len(img_files)): # 이미지 파일만큼 반복
            image_path_, classes_, bbox_ = parse_txt_annot(img_files[i], annot_files[i]) # 이미지 경로, 클래스, 바운딩 박스 정보를 추출(파싱)
            # 앞서 초기화한 리스트에 추가
            image_paths.append(image_path_)
            bbox.append(bbox_)
            classes.append(classes_)

        # 텐서플로우 ragged텐서로 변환. 앞서 저장된 리스트를 ragged텐서로 변환 -> 비정형 데이터를 다루기 위함
        image_paths = tf.ragged.constant(image_paths)
        bbox = tf.ragged.constant(bbox)
        classes = tf.ragged.constant(classes)

        # 변환된 ragged텐서 형태의 이미지 경로, 클래스, 바운딩 박스 정보 반환
        return image_paths, classes, bbox

    # applying functions
    # 앞서 선언한 creating_files 함수를 사용해 훈련, 검증, 테스트 데이터 셋 생성
    # 이미지 경로, 클래스, 바운딩 박스 정보 로드
    train_img_paths, train_classes, train_bboxes = creating_files('data/train/images',
                                                                'data/train/labels')

    valid_img_paths, valid_classes, valid_bboxes = creating_files('data/valid/images',
                                                                'data/valid/labels')

    test_img_paths, test_classes, test_bboxes = creating_files('data//test/images',
                                                            'data//test/labels')

    def img_preprocessing(img_path):
        img = tf.io.read_file(img_path) # 이미지 읽기
        img = tf.image.decode_jpeg(img, channels = 3) # jpeg파일로 디코딩하고 채널을 3개로 설정
        img = tf.cast(img, tf.float32)  # 이미지를 32비트 부동 소수점 형식으로 변환. 모델 입력전이라 이렇게 함

        return img

    # 이미지 크기를 변경하는데 사용됨. 이미지의 크기를 다양하게 조절할 수 있도록 함. 바운딩 박스도 조정
    resizing = keras_cv.layers.JitteredResize(
        target_size=(640, 640), # 640 * 640으로 설정
        scale_factor=(0.8, 1.25), # 0.8배에서 1.25배 범위 내에서 임의로 크기를 조정
        bounding_box_format="xyxy") # 바운딩 박스가 좌측 상단, 우측 하단의 x, y좌표로 정의 되어 있음

    # loading dataset
    # 이미지 파일의 경로, 해당 이미지에 대한 클래스 정보, 바운딩 박스 정보를 입력으로 해 이를 딕셔너리 형태로 묶어 반환
    def load_ds(img_paths, classes, bbox):
        img = img_preprocessing(img_paths) # 이미지를 로드하고 전처리 함

        # 바운딩 박스 딕셔너리 생성. 클래스와 바운딩 박스 정보를 각각 텐서로 변환해 딕셔너리에 저장
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32), # 클래스 정보를 부동 소수점 형태로 변환
            "boxes": bbox } # 바운딩 박스 정보를 그대로 사용

        return {"images": img, "bounding_boxes": bounding_boxes} # 전처리된 이미지와 바운딩 박스 정보를 포함한 딕셔너리 반환

    # 앞선 함수에서 생성된 딕셔너리를 입력받아, 이미지 데이터와 바운딩 박스 정보를 튜플 형식으로 변환
    def dict_to_tuple(inputs):
        return inputs["images"], inputs["bounding_boxes"]

    # 데이터 셋을 위한 로더와 데이터 파이프라인으로 구성

    # 학습 데이터 로더 구성
    # 각 데이터셋의 이미지 경로, 클래스, 바운딩 박스 정보로부터 텐서 슬라이스 데이터셋을 생성. 각 요소를 개별적으로 처리할 수 있음
    train_loader = tf.data.Dataset.from_tensor_slices((train_img_paths, train_classes, train_bboxes))
    # 데이터셋 파이프라인 구성
    # 1. 데이터 로딩 및 전처리: load_ds함수를 사용해 각 이미지를 로드하고, 해당 데이터를 딕셔너리 형태로 전처리 AUTO는 자동으로 적절한 수의 병렬 처리를 결정하도록 함
    # 2. 데이터 셔플링과 배치 처리: 데이터 순서를 무작위로 섞음(학습 데이터만). 16의 배치사이즈로 각 배치 내의 데이터 포인트가 다른 길이를 가질 수 있게 함. 마지막 배치가 배치크기보다 작으면 버림
    # 3. 이미지 재조정: 각 이미지 크기를 재조정하여 입력받는 이미지 크기를 일정하게 유지시킴
    # 4. 데이터 포맷변환: 데이터를 딕셔너리 형태에서 튜플 형태로 변환
    # 5. 데이터 프리페칭: 데이터 로딩을 비동기적으로 수행하여 모델 훈련 시 GPU가 다음 데이터 배치를 기다리지 않고 계속 훈련할 수 있도록 함.
    train_dataset = (train_loader
                    .map(load_ds, num_parallel_calls = AUTO)
                    .shuffle(BATCH_SIZE*10)
                    .ragged_batch(BATCH_SIZE, drop_remainder = True)
                    .map(resizing, num_parallel_calls = AUTO)
                    .map(dict_to_tuple, num_parallel_calls = AUTO)
                    .prefetch(AUTO))


    valid_loader = tf.data.Dataset.from_tensor_slices((valid_img_paths, valid_classes, valid_bboxes))
    valid_dataset = (valid_loader
                    .map(load_ds, num_parallel_calls = AUTO)
                    .ragged_batch(BATCH_SIZE, drop_remainder = True)
                    .map(resizing, num_parallel_calls = AUTO)
                    .map(dict_to_tuple, num_parallel_calls = AUTO)
                    .prefetch(AUTO))


    test_loader = tf.data.Dataset.from_tensor_slices((test_img_paths, test_classes, test_bboxes))
    test_dataset = (test_loader
                    .map(load_ds, num_parallel_calls = AUTO)
                    .ragged_batch(BATCH_SIZE, drop_remainder = True)
                    .map(resizing, num_parallel_calls = AUTO)
                    .map(dict_to_tuple, num_parallel_calls = AUTO)
                    .prefetch(AUTO))

    # creating mirrored strategy
    # mirroredstrategy초기화 싱글  머신 환경에서 모델 훈련을 가속화하기 위해 사용
    # 효율적으로 대규모 모델을 바르게 훈련할 수 있도록 함
    stg = tf.distribute.MirroredStrategy()

    # 욜로 백본 모델 생성
    with stg.scope(): # 모델 관련 컴포넌트가 mirroedstrategy를 사용해 여러 gpu에 걸쳐 동기화되게 설정
        # 모델 백본 생성 yolov8의 사전 설정된 작은 크기의 백본 설정, 입력 이미지의 자동 리스케일링 활성화
        backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone", include_rescaling = True)
        # yolov8 모델설정. 생성된 백본과 추가 설정을 사용해 객체 탐지 모델 초기화. 클래스 수는 class_mapping 딕셔너리 길이를 기반. 바운딩 박스 포맷은 xyxy
        YOLOV8_model = keras_cv.models.YOLOV8Detector(num_classes=len(class_mapping),
                                                    bounding_box_format = "xyxy", backbone = backbone, fpn_depth = 5)

        # 옵티마이저 및 콜백 설정: AdamW사용
        optimizer = AdamW(learning_rate=input_learning_rate, weight_decay=input_weight_decay, global_clipnorm = 10.0)
        # 모델 훈련과정에서 성능을 모니터링하고 조정하는 콜백 설정. 가중치 저장 및 학습률 조정, 조기 종료
        my_callbacks = [ModelCheckpoint('yolov8_acne_detection_large.h5', monitor = 'val_loss',save_best_only = True, save_weights_only = True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=8, verbose=0, min_delta=0.001),
                        EarlyStopping(monitor='val_loss', patience=input_patience)]

        # 모델 컴파일, 이진 크로스엔트로피 사용, 바운딩 박스 손실로는 CIoU사용
        YOLOV8_model.compile(optimizer = optimizer, classification_loss = 'binary_crossentropy', box_loss = 'ciou')
        
    # 모델 학습
    hist = YOLOV8_model.fit(train_dataset, validation_data = valid_dataset,  epochs = input_epochs, callbacks = my_callbacks)
    return 'finish!'
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60017)
