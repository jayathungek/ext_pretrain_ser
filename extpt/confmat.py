import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 5)

import torch
from torchmetrics.classification import MulticlassRecall, MulticlassConfusionMatrix

import numpy as np
from pprint import pprint

def calculate_metrics_from_confusion_matrix(conf_matrix):
    num_classes = conf_matrix.shape[0]
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1_score

# # Example confusion matrix
# conf_matrix = np.array([
#     [50, 2, 1],
#     [10, 40, 5],
#     [5, 2, 45]
# ])





def dataline2tensor(dataline):
    rows = dataline.split("|")
    tensor_rows = []
    for row in rows:
        items = [i for i in row.split(" ") if i.isdigit()]
        items = [int(i.strip()) for i in items]
        tensor_rows.append(items)

    data = torch.tensor(tensor_rows)
    return data


def make_confmat_from_txt(data_filename):
    with open(data_filename, "r") as fh:
        lines = fh.readlines()

    for line in lines:
        line = line.split(":")
        name, N, human, data = line
        cfm = MulticlassConfusionMatrix(int(N), normalize='true')
        human = human.split(" ")
        data = dataline2tensor(data)
        for i in range(int(N)):
            cfm.metric_state['confmat'][i] = data[i]
        # fig, ax = cfm.plot(labels=human)
        # ax.tick_params(axis='both', labelsize=15)
        # fig.set_size_inches(8, 8)
        # fig.savefig(f"{name}_confusion_matrix.png")
        print(f"{name} stats:")
        precision, recall, f1_score = calculate_metrics_from_confusion_matrix(cfm.metric_state['confmat'])
        print("Precision:")
        pprint([f'{i:.3f}' for i in precision.tolist()])
        pprint(f'{precision.mean():.3f}')
        print("Recall:")
        pprint([f'{i:.3f}' for i in recall.tolist()])
        pprint(f'{recall.mean():.3f}')
        print("F1:")
        pprint([f'{i:.3f}' for i in f1_score.tolist()])
        pprint(f'{f1_score.mean():.3f}')

def make_confmat_from_chkpt(data_filename, name, human_labels):
    import ast
    chkpt = torch.load(data_filename)
    N = len(human_labels)
    cfm = MulticlassConfusionMatrix(N, normalize='true')
    data = torch.tensor(ast.literal_eval(chkpt["best"]["VAL"]["conf_matrix"]))
    for i in range(int(N)):
        cfm.metric_state['confmat'][i] = data[i]
    fig, ax = cfm.plot(labels=human_labels)
    ax.tick_params(axis='both', labelsize=15)
    fig.set_size_inches(8, 8)
    fig.savefig(f"{name}_confusion_matrix.png")
    print(f"{name} stats:")
    precision, recall, f1_score = calculate_metrics_from_confusion_matrix(cfm.metric_state['confmat'])
    print("Precision:")
    pprint([f'{i:.3f}' for i in precision.tolist()])
    pprint(f'{precision.mean():.3f}')
    print("Recall:")
    pprint([f'{i:.3f}' for i in recall.tolist()])
    pprint(f'{recall.mean():.3f}')
    print("F1:")
    pprint([f'{i:.3f}' for i in f1_score.tolist()])
    pprint(f'{f1_score.mean():.3f}')

if __name__ == "__main__":
    filename = Path(sys.argv[1])
    if filename.suffix == ".pth":
        name = sys.argv[2]
        labels = sys.argv[3].split(",")
        make_confmat_from_chkpt(filename, name, labels)
    else:
        make_confmat_from_txt(filename)
        

