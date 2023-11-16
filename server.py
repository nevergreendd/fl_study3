import flwr as fl
import os
import numpy as np

N=8
fraction_fit=1
fraction_eval=1
min_fit_clients=N
min_eval_clients=N
min_available_clients=N
num_rounds=3

def evaluate_metrics_aggregation_fn(eval_metrics):
    data_len = sum([num for num, met in eval_metrics])
    rmse = sum([num*met['rmse'] for num, met in eval_metrics])/data_len
    return {'rmse' : rmse}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=fraction_fit,                    # 훈련을 위해서 사용 가능한 클라이언트의 100% 이용
    fraction_evaluate=fraction_eval,              # 평가를 위해서 사용 가능한 클라이언트의 100% 이용
    min_fit_clients=min_fit_clients,              # 훈련을 위해서는 적어도 5개 이상의 클라이언트가 필요
    min_evaluate_clients=min_eval_clients,        # 평가를 위해서는 적어도 5개 이상의 클라이언트가 필요
    min_available_clients=min_available_clients,  # 사용 가능한 클라이언트의 수가 5 될 때까지 대기
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)

print('server start!')
output = fl.server.start_server(server_address='[::]:8090',config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy)
output
