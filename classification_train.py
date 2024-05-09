import pandas as pd
from PIL import Image
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics
import numpy
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from flask import Flask, request, jsonify

app = Flask(__name__)

# define a data class
class ClassificationDataset:
    def __init__(self, data, data_path, transform, training=True):
        """Define the dataset for classification problems

        Args:
            data ([dataframe]): [a dataframe that contain 2 columns: image name and label]
            data_path ([str]): [path/to/folder that contains image file]
            transform : [augmentation methods and transformation of images]
            training (bool, optional): []. Defaults to True.
        """
        self.data = data
        self.imgs = data["path"].unique().tolist()
        self.data_path = data_path
        self.training = training
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_path, self.data.iloc[idx, 0]))
        label = self.data.iloc[idx, 1]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    
def make_loader(dataset, train_batch_size, validation_split=0.2):
    """make dataloader for pytorch training

    Args:
        dataset ([object]): [the dataset object]
        train_batch_size ([int]): [training batch size]
        validation_split (float, optional): [validation ratio]. Defaults to 0.2.

    Returns:
        [type]: [description]
    """
    # number of samples in train and test set
    train_len = int(len(dataset) * (1 - validation_split))
    test_len = len(dataset) - train_len
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
    # create train_loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True,
    )
    # create test_loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,)
    return train_loader, test_loader


def data_split(data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(
        data, data["label"], test_size=test_size, stratify = data.iloc[:,1]
    )
    return x_train, x_test, y_train, y_test


class Metrics:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        # initialize a metric dictionary
        self.metric_dict = {metric_name: [0] for metric_name in self.metric_names}

    def step(self, labels, preds):
        for metric in self.metric_names:
            # get the metric function
            do_metric = getattr(
                skmetrics, metric, "The metric {} is not implemented".format(metric)
            )
            # check if metric require average method, if yes set to 'micro' or 'macro' or 'None'
            try:
                self.metric_dict[metric].append(
                    do_metric(labels, preds, average="macro")
                )
            except:
                self.metric_dict[metric].append(do_metric(labels, preds))

    def epoch(self):
        # calculate metrics for an entire epoch
        avg = [sum(metric) / (len(metric) - 1) for metric in self.metric_dict.values()]
        metric_as_dict = dict(zip(self.metric_names, avg))
        return metric_as_dict

    def last_step_metrics(self):
        # return metrics of last steps
        values = [self.metric_dict[metric][-1] for metric in self.metric_names]
        metric_as_dict = dict(zip(self.metric_names, values))
        return metric_as_dict

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        target = F.one_hot(target, num_classes=pred.size(-1))
        target = target.float()
        target = (1 - self.smoothing) * target + self.smoothing / pred.size(-1)
        log_pred = F.log_softmax(pred, dim=self.dim)
        loss = nn.KLDivLoss(reduction='batchmean')(log_pred, target)
        return loss

def train_one_epoch(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    criterion,
    train_metrics,
    val_metrics,
):

    # training-the-model
    train_loss = 0
    valid_loss = 0
    all_labels = []
    all_preds = []
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.type(torch.FloatTensor).to(device)
#         day = day.view(-1,1).type(torch.FloatTensor).to(device)
        # target=torch.Tensor(target)
        target = target.float().to(device)
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        preds = torch.argmax(output, axis=1).cpu().detach().numpy()
        labels = target.cpu().numpy()
        # calculate-the-batch-loss
        loss = criterion(output.type(torch.FloatTensor), target.type(torch.LongTensor))
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        # calculate training metrics
        all_labels.extend(labels)
        all_preds.extend(preds)

    train_metrics.step(all_labels, all_preds)

    # validate-the-model
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.type(torch.FloatTensor).to(device)
#             day = day.view(-1,1).type(torch.FloatTensor).to(device)
            target = target.to(device)
            output = model(data)
            preds = torch.argmax(output, axis=1).tolist()
            labels = target.tolist()
            all_labels.extend(labels)
            all_preds.extend(preds)
            loss = criterion(output, target)

            # update-average-validation-loss
            valid_loss += loss.item() * data.size(0)

    val_metrics.step(all_labels, all_preds)
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)

    return (
        train_loss,
        valid_loss,
        train_metrics.last_step_metrics(),
        val_metrics.last_step_metrics(),
    )

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

def test_result(model, test_loader, device,name='no_tta_prob.npy'):
    # testing the model by turning model "Eval" mode
    model.eval()
    preds = []
    aprobs = []
    labels = []
    with torch.no_grad():
        for data,target in test_loader:
            # move-tensors-to-GPU
            data = data.to(device)
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            prob = nn.Softmax(dim=1)
            # applying Softmax to results
            probs = prob(output)
            aprobs.append(probs.cpu())
            labels.append(target.cpu().numpy())
            preds.extend(torch.argmax(probs, axis=1).tolist())
    aprobs = np.array(aprobs)
    np.save(name,aprobs)
    return preds, np.array(labels)

def test_result_roc_auc(model, test_loader, device, num_classes, name='no_tta_prob.npy'):
    # testing the model by turning model "Eval" mode
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            # move tensors to GPU
            data = data.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            prob = nn.Softmax(dim=1)
            # applying Softmax to results
            prob_result = prob(output)
            probs.extend(prob_result.cpu().numpy())
            labels.extend(target.cpu().numpy())

    probs = np.array(probs)
    labels = label_binarize(labels, classes=range(num_classes))
    np.save(name, probs)
    return labels, probs
@app.route('/classification_train', methods=['POST'])
def upload_model_train():
    data = request.get_json()
    
    # 데이터에서 각 인자를 추출
    input_learning_rate = data.get('learningRate', None)
    input_epochs = data.get('Epochs', None)
    input_patience = data.get('Patience', None)
    
    train_files = ['data/NNEW_trainval_0.txt','data/NNEW_trainval_1.txt',
            'data/NNEW_trainval_2.txt','data/NNEW_trainval_3.txt',
            'data/NNEW_trainval_4.txt']

    test_files = ['data/NNEW_test_0.txt','data/NNEW_test_1.txt',
            'data/NNEW_test_2.txt','data/NNEW_test_3.txt',
            'data/NNEW_test_4.txt']
    path = 'data/JPEGImagesv'

    #batch size
    bs = 16

    criterion = LabelSmoothingLoss(smoothing=0.12)
        
    train_df = pd.read_csv(train_files[0],names=['path','label','leisons'],sep='  ')
    x_train, x_val, y_train, y_val = data_split(train_df,0.2)
    test_df = pd.read_csv(test_files[0],names=['path','label','leisons'],sep='  ')

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images left-right
                                torchvision.transforms.RandomVerticalFlip(p=0.5),
                                torchvision.transforms.RandomRotation(degrees=15),
                                                torchvision.transforms.ElasticTransform(),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean, std)])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean, std)])

    train_dataset = ClassificationDataset(x_train,data_path = "data/JPEGImages",transform=transform,training=True)
    val_dataset = ClassificationDataset(x_val,data_path = "data/JPEGImages",transform=test_transform,training=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
        )
        # create test_loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    testset = ClassificationDataset(test_df,data_path = "data/JPEGImages",transform=test_transform,training=True)
    test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False,
        )

    train_metrics = Metrics(["accuracy_score","f1_score"])
    val_metrics = Metrics(["accuracy_score","f1_score"])
        
    model = MyNet().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=input_learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=input_patience, factor=0.5
        )

    device = torch.device("cuda")


    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)
    num_epoch = input_epochs
    best_val_acc = 0.0

    print("begin training process")
    for i in tqdm(range(0, num_epoch)):
        loss, val_loss, train_result, val_result = train_one_epoch(
            model,
            train_loader,
            val_loader,
            device,
            optimizer,
            criterion,
            train_metrics,
            val_metrics,
        )

        scheduler.step(val_loss)
    #     scheduler.step()
        print(
            "Epoch {} / {} \n Training loss: {} - Other training metrics: ".format(
                i + 1, num_epoch, loss
            )
        )
        print(train_result)
        print(
            " \n Validation loss : {} - Other validation metrics:".format(val_loss)
        )
        print(val_result)
        print("\n")
        # saving epoch with best validation accuracy
        if (loss<0.04):
            # no saving
            continue
        if best_val_acc < float(val_result["accuracy_score"]):
            print(
                "Validation accuracy= "+
                str(val_result["accuracy_score"])+
                "===> Save best epoch"
            )
            best_val_acc = val_result["accuracy_score"]
            torch.save(
                model,
                "./model/" +  "best.pt"
            )
        else:
            print(
                "Validation accuracy= "+ str(val_result["accuracy_score"])+ "===> No saving"
            )
            continue

    preds,labels =test_result(model, test_loader, device)


    # Assuming preds and labels are defined earlier
    f1 = f1_score(labels, preds, average='macro')  # 클래스별로 동일한 가중치를 적용하여 계산
    f1_micro = f1_score(labels, preds, average='micro')  # 전체 데이터에 대해 계산
    f1_weighted = f1_score(labels, preds, average='weighted')  # 각 클래스의 지지도(실제 샘플 수)에 따라 가중 평균
    num_classes = 4  # Number of classes
    labels, probs = test_result_roc_auc(model, test_loader, device, num_classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    average_auc = np.mean([roc_auc[i] for i in range(num_classes)])
    return jsonify({'f1Score': f1, 'f1ScoreM': f1_micro, 'f1ScoreW':f1_weighted, 'Auc': average_auc})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60017)