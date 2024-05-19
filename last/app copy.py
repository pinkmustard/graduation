from flask import Flask
from acne_classification.classification_pred import predict_classification
from acne_classification.classification_train import train_classification
from acne_detection.detection_pred import predict_detection
from acne_detection.detection_train import train_detection

app = Flask(__name__)

@app.route('/classification_pred', methods=['POST'])
def classification_prediction():
    return predict_classification()

@app.route('/classification_train', methods=['POST'])
def classification_training():
    return train_classification()

@app.route('/detection_pred', methods=['POST'])
def detection_prediction():
    return predict_detection()

@app.route('/detection_train', methods=['POST'])
def detection_training():
    return train_detection()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60017)
