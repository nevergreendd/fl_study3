import torch
from dataset.dataset_test import MolTestDatasetWrapper
from dataset import dataset
from dataset.dataset_test import MolTestDataset
import yaml
from torch import nn
from models import ginet_finetune


device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
config['dataset']['task'] = 'regression'
config['dataset']['data_path'] = 'ms_mouse.csv'
#config['dataset']['data_path'] = 'ms_human.csv'

model = ginet_finetune.GINet(config['dataset']['task'], **config["model"]).to(device)

# mouse
model_state_dict_mlm = torch.load("finetune/Sep24_02-48-50_dacon_MLM/checkpoints/model.pth", map_location=device)

# human
model_state_dict_hlm = torch.load("finetune/Sep24_02-54-17_dacon_HLM/checkpoints/model.pth", map_location=device)


template = open('sample_submission.csv').readlines()[1:]
template = [x.strip().split(',') for x in template]


outputs = [[],[]]
for i, model_state_dict in enumerate([model_state_dict_mlm, model_state_dict_hlm]):
    model.load_state_dict(model_state_dict, strict=False)

    with torch.no_grad():
        model.eval()
        # predict test.csv
        inputs = MolTestDataset('test.csv', "y", "regression")

        for j in range(len(inputs)):
            res = model(inputs[j])[1][0][0].item()
            if res < 0:
                res = 0.0
            elif res > 100:
                res = 100.0
            outputs[i].append(res)

with open('sub.csv', 'w') as f:
    f.write('id,MLM,HLM\n')
    for j in range(len(template)):
        id = template[j][0]
        #print(f'{id},{outputs[0][j]},{outputs[1][j]}\n')
        f.write(f'{id},{outputs[0][j]},{outputs[1][j]}\n')
        
