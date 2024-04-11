import os
from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision, RetrievalAUROC, RetrievalMAP, RetrievalHitRate
import torch

def build_ground_truth(query_fingerprints, database_fingerprints):
    ground_truth = {}

    # Iterate over the query fingerprints
    for query_file, query_fingerprint in query_fingerprints.items():
        query_name = query_file.replace('.wav','')
        ground_truth[query_file] = {}

        # Iterate over the database fingerprints
        for db_file, db_fingerprint in database_fingerprints.items():
              # Extract the database item name
            db_name = db_file.replace('.wav','')
            ground_truth[query_file][db_file] = 0  # Initialize the boolean value

            # TODO: Add your logic here to determine whether the fingerprints match
            if query_name.split('-')[0] == db_name:
                ground_truth[query_file][db_file] = 1

    return ground_truth


def get_metrics(top_k, preds, ground_truth):
    
    # join the dictionaries of preds and ground truth on the query key. For each key, make a dict [preds, ground_truth]
    # then pass to the metric
    
    k = top_k
    
    metrics = {
        f'Recall@{k}': RetrievalRecall(top_k=top_k),
        f'Precision@{k}': RetrievalPrecision(top_k=top_k),
        'AUROC': RetrievalAUROC(),
        'MAP': RetrievalMAP(),
        f'HitRate@{k}': RetrievalHitRate(top_k=top_k)
    }
    
    metric_results = {
        f'Recall@{k}': {
            'preds': [],
            'ground_truth': [],
            'index': []},
        f'Precision@{k}': {
            'preds': [],
            'ground_truth': [],
            'index': []},
        'AUROC': {
            'preds': [],
            'ground_truth': [],
            'index': []},
        'MAP': {
            'preds': [],
            'ground_truth': [],
            'index': []},
        f'HitRate@{k}': {
            'preds': [],
            'ground_truth': [],
            'index': []}
    }

    index = 0
    for query_item, db_items in ground_truth.items():
        preds_item = preds[query_item]
        ground_truth_item = [ground_truth[query_item][db_item] for db_item in db_items]
        preds_item = torch.tensor([x[-1] for x in list(preds_item.values())])
        ground_truth_item = torch.tensor(ground_truth_item)
        index +=1
        index_item = torch.full_like(ground_truth_item, index)
        
        for metric_name, metric in metrics.items():
            
            metric_results[metric_name]['preds'].append(preds_item)
            metric_results[metric_name]['ground_truth'].append(ground_truth_item)
            metric_results[metric_name]['index'].append(index_item)
    
            
    metric_results = {metric_name : {
        'preds': torch.cat(metric_results[metric_name]['preds']),
        'ground_truth': torch.cat(metric_results[metric_name]['ground_truth']),
        'index': torch.cat(metric_results[metric_name]['index'])
    } for metric_name in metric_results.keys()}
    
    
    results = {metric_name: metric(metric_results[metric_name]['preds'], metric_results[metric_name]['ground_truth'], metric_results[metric_name]['index']) for metric_name, metric in metrics.items()}

    return results
    
    
    