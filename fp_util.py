import numpy as np
from sklearn.neighbors import DistanceMetric

def load_fp(feature_type="onehot", data_path="data/fp/contextPasquier99.txt", data_hamming_dist=3):
    with open(data_path) as f:
        transaction = f.readlines()
    transaction = list([list(map(int, t.split())) for t in transaction])

    items = set()
    for t in transaction:
        items = items | set(t)

    # One-hot vector生成
    features = np.zeros([len(transaction), len(items)], dtype=np.float64)
    items = sorted(list(items))
    for i, tran in enumerate(transaction):
        for j, item in enumerate(items):
            if item in tran:
                features[i][j] = 1

    distance = DistanceMetric.get_metric("hamming").pairwise(features)
    np.fill_diagonal(distance, np.Infinity)
    distance = np.where(distance > data_hamming_dist / len(items), np.Infinity, distance)
    sort_dis = np.sort(distance, axis=-1)
    sort_idx = np.argsort(distance, axis=-1)
    sort_idx = np.where(sort_dis != np.Infinity, sort_idx, -1)

    # all_subset = []
    # for i, tran1 in enumerate(transaction):
    #     tmp = set()
    #     for j, tran2 in enumerate(transaction):
    #         if i == j:
    #             continue
    #         if tran1 <= tran2:
    #             tmp.add(j)
    #     all_subset.append(tmp)
        
    data = {"feature": features, "raw_data": transaction, "category": sort_idx}
    return data