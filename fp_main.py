import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.neighbors import DistanceMetric

import fp_util
import model as Model
import wandb


def eval_retrieval(hash_code, raw_data, category, hamming_dist, output_size):
    distance = DistanceMetric.get_metric("hamming").pairwise(hash_code)
    np.fill_diagonal(distance, np.Infinity)
    distance = np.where(distance > hamming_dist / output_size, np.Infinity, distance)
    sort_dis = np.sort(distance, axis=-1)
    sort_idx = np.argsort(distance, axis=-1)
    sort_idx = np.where(sort_dis != np.Infinity, sort_idx, -1)

    # 比較
    precision = []
    recall = []
    raw_count = 0
    pred_count = 0
    for i, predict in enumerate(sort_idx):
        truth = set(category[i])
        truth.discard(-1)
        predict = set(predict)
        predict.discard(-1)
        tp = len(truth & predict)
        
        raw_count += len(truth)
        pred_count += len(predict)

        if len(truth) != 0:
            recall.append(tp / len(truth))
        if len(predict) != 0:
            precision.append(tp / len(predict))
    if len(precision) != 0:
        precision = round(np.mean(np.sum(precision) / len(precision)), 2)
    else:
        precision = 0
    if len(recall) != 0:
        recall = round(np.mean(np.sum(recall) / len(recall)), 2)
    else:
        recall = 0
    return precision, recall, pred_count, raw_count


def encode(model, feature, batch_size=512, use_cuda=True) :
    vector = []
    model.eval()
    for i in range(0, feature.shape[0], batch_size) :
        x = feature[i:i+batch_size]
        x = torch.from_numpy(x).float()
        if use_cuda :
            x = x.cuda()
        with torch.no_grad() :
            vector.append(
                model.encode(x).detach().cpu().numpy().astype("uint8")
            )
    vector = np.vstack(vector)
    model.train()
    return vector


def main(config, wandb) :
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)

    data = fp_util.load_fp(feature_type=config.feature_type, data_path=config.data_path, data_hamming_dist=config.data_hamming_dist)
    raw_db = data["raw_data"]
    feature = data["feature"]
    category = data["category"]
    input_size = feature.shape[-1]

    dataloader = DataLoader(TensorDataset(
         torch.from_numpy(feature).float()
    ), batch_size=config.batch_size, shuffle=True, drop_last=True)

    model = Model.NASH(config, input_size)
    wandb.watch(model)
    if config.use_cuda :
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_rate)

    max_precision = 0
    best_weight = None
    for e in range(config.epoch) :
        avg_loss = 0
        for batch in tqdm(dataloader) :
            x = batch[0]
            if config.use_cuda :
                x = x.cuda()
            tot_loss = model(x)
            avg_loss += tot_loss.item()
            tot_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        print("train epoch {} : {}".format(e+1, avg_loss/len(dataloader)))
        wandb.log({"train epoch": e+1, "loss": avg_loss/len(dataloader)})
        hash_code = encode(model, feature, use_cuda=config.use_cuda)
        precision, recall, pred_count, raw_count = eval_retrieval(hash_code, raw_db, category, config.hamming_dist, config.output_size)
        print("Number of Predict / Raw_Data: {} / {}".format(pred_count, raw_count))
        wandb.log({"num_of_predict": pred_count, "num_of_rawd_ata": raw_count})
        print("Precision: {}, Recall: {}".format(precision, recall))
        wandb.log({"precision": precision, "recall": recall})
        if precision > max_precision :
            max_precision = precision
            best_weight = model.state_dict()
        print("")
        print("")
    torch.save(best_weight, "best.w")
    wandb.save("best.w")


if __name__=="__main__" :
    wandb.init(project="nash-db", entity="mu-lab")
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", default=True, type=bool, help="use cuda or not")
    parser.add_argument("--feature_type" , default="onehot", type=str, help="tfidf | onehot")
    parser.add_argument("--data_path" , default="data/fp/chess.txt", type=str)
    parser.add_argument("--hamming_dist" , default=3, type=int)
    parser.add_argument("--data_hamming_dist" , default=3, type=int)
    parser.add_argument("--epoch" , default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--hidden_size" , default=500, type=int)
    parser.add_argument("--output_size" , default=64, type=int)
    parser.add_argument("--dropout" , default=0.1, type=float)
    parser.add_argument("--deterministic" , default=True, type=bool, help="use deterministic binarization or not")
    parser.add_argument("--lr_decay_step" , default=1e4, type=int)
    parser.add_argument("--lr_decay_rate" , default=0.96, type=float)
    parser.add_argument("--top_n" , default=100, type=int, help="number of top n retrieved number")

    config = parser.parse_args()
    wandb.config.update(config)
    main(config, wandb)
