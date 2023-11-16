import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
import flwr as fl
import math
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import roc_auc_score, mean_squared_error 
from models import ginet_finetune
from sklearn.model_selection import train_test_split
from dataset.dataset_test import MolTestDatasetWrapper

files = ['ms_mouse.csv', 'trimmed_CHEMBL2367369.csv', 'trimmed_CHEMBL613373.csv', 'trimmed_CHEMBL2367379.csv',
 'trimmed_CHEMBL613580.csv', 'trimmed_CHEMBL2367428.csv', 'trimmed_CHEMBL613694.csv', 'trimmed_CHEMBL612558.csv']

BATCH_SIZE = 128
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:3') # 해당 조의 GPU 번호로 변경 ex) 1조 : cuda:1
else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

# %%
def train(model, epochs, train_loader, optimizer, log_interval, loss_fn):
    model.train()
    for epoch in range(epochs):
        for batch_idx, sample in enumerate(train_loader):
            image = sample
            label = sample['y']
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            h, output = model(image)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                                                        epoch, batch_idx * len(image), 
                                                        len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                                                        loss.item()))
            
''' 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate2(model, test_loader, loss_fn):
    model.eval()
    predictions = []
    labels = []
    test_loss = 0.0

    with torch.no_grad():
        for sample in test_loader:
            image = sample
            true_label = sample.y
            image = image.to(DEVICE)
            true_label = true_label.to(DEVICE)
            h, output = model(image)
            label = true_label.cpu().flatten().numpy()
            prediction = output.cpu().detach().numpy()

            # print(label.cpu().flatten().numpy(), output[0][0].item())
            test_loss += loss_fn(output, true_label).item()
            predictions.extend(prediction)
            labels.extend(label)
            # prediction = output.max(1, keepdim = True)[1]
            # correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    rmse = mean_squared_error(labels, predictions, squared=False).item()
    # print(labels, predictions)
    # print('rmse', rmse)
    return test_loss, rmse

# %%
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, dataset, opt, loss_fn):
        self.model = model
        # self.train_loader, _, self.test_loader = dataset.get_data_loaders()
        self.train_loader, self.validation_loader, _ = dataset.get_data_loaders()
        # self.test_loader = dataset.get_data_loaders()
        self.optimizer = opt
        self.loss_fn = loss_fn

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters): # pytorch 모델에 파라미터를 적용하는 코드가 복잡하여 함수로 정의
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters) # 위에서 정의한 set_parameters함수를 사용
        train(self.model, 50, self.train_loader, self.optimizer, 200, self.loss_fn)
        # ''' MLP 학습 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
        # EPOCHS = 30
        # for epoch in range(1, EPOCHS + 1):
        #     train(self.model, epoch, self.train_loader, self.optimizer, 200, self.loss_fn)
        #     test_loss, test_accuracy = evaluate(torch_model_cen, test_loader_n, criterion)
        #     print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        #         epoch, test_loss, test_accuracy))
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, rmse = evaluate2(self.model, self.validation_loader, self.loss_fn)
        print(f'rmse: {rmse}')
        return loss, len(self.validation_loader.dataset), {"rmse": rmse}


# %%
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
config['dataset']['task'] = 'regression'
config['dataset']['data_path'] = 'laidd_source_data/' + files[int(sys.argv[1])]
# config['dataset']['data_path'] = 'ms_mouse.csv'
config['dataset']['target'] = 'MLM'

model_fl = ginet_finetune.GINet(config['dataset']['task'], **config["model"]).to(DEVICE)
criterion_fl = nn.MSELoss().to(DEVICE)
optimizer_fl = torch.optim.Adam(model_fl.parameters())

# %%

# split data into train and validation datasets by using train_test_split

# train_loader_fl = DataLoader(train_list, batch_size=BATCH_SIZE)
# test_loader_fl = DataLoader(test_list, batch_size=BATCH_SIZE)
dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

flwr_client = FlowerClient(model_fl, dataset, optimizer_fl, criterion_fl)

fl.client.start_numpy_client(server_address="127.0.0.1:8090", client=flwr_client)

# %%



