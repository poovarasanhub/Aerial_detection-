if True:
    from reset_random import reset_random

    reset_random()

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import prettytable
from sklearn.preprocessing import StandardScaler

from kelm import KELMClassifier
from performance_evaluator.metrics import evaluate
from performance_evaluator.plots import confusion_matrix, precision_recall_curve, roc_curve
from utils import DATASET


def get_data(dataset):
    print('[INFO] Loading {0} Dataset'.format(dataset))
    fe = np.load('Data/fused_features/{0}/features.npy'.format(dataset))
    print('[INFO] Features Shape :: {0}'.format(fe.shape))
    lb = np.load('Data/fused_features/{0}/labels.npy'.format(dataset))
    print('[INFO] Labels Shape :: {0}'.format(lb.shape))
    return fe, lb, dataset


def classify(features, labels, dataset, bwo=False):
    classifier_dir = 'classifiers/{0}'.format(dataset)
    os.makedirs(classifier_dir, exist_ok=True)
    params = {
        'UCM': {'batch_size': 32, 'n_neurons': 2000},
        'AID': {'batch_size': 8, 'n_neurons': 2500},
    }
    plot_params = {
        'UCM': {'figsize': (6.5, 6.5)},
        'AID': {'figsize': (8.5, 8.5)}
    }
    if bwo:
        ss = StandardScaler()
        ss.fit(features, labels)
        features = ss.transform(features)
        with open(os.path.join(classifier_dir, 'ss.pkl'), 'wb') as f:
            pickle.dump(ss, f)
    print('[INFO] Building KELM With Params :: {0}'.format(params[dataset]))
    classifier = KELMClassifier(**params[dataset])
    classifier.fit(features, labels)
    with open(os.path.join(classifier_dir, 'classifier{0}.pkl'.format('' if bwo else '')), 'wb') as f:
        pickle.dump(classifier, f)
    pred = classifier.predict(features)
    prob = classifier.predict_proba(features)

    results_dir = 'results/{0}'.format(dataset)
    os.makedirs(results_dir, exist_ok=True)
    m = evaluate(labels, pred, prob, DATASET[dataset])
    m.overall_metrics.to_csv(os.path.join(results_dir, 'metrics{0}.csv'.format('' if bwo else '')), index=False)
    p_table = prettytable.PrettyTable(field_names=list(m.overall_metrics['Metrics'].values))
    p_table.hrules = True
    p_table.add_row(list(m.overall_metrics['Values'].values))
    print(p_table)

    plt.figure(figsize=plot_params[dataset]['figsize'])
    ax = plt.gca()
    confusion_matrix(labels, pred, list(range(len(DATASET[dataset]))), ax=ax)
    plt.savefig(os.path.join(results_dir, 'conf_mat{0}.png'.format('' if bwo else '')))
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=plot_params[dataset]['figsize'])
    ax = plt.gca()
    precision_recall_curve(labels, prob, list(range(len(DATASET[dataset]))), ax=ax)
    plt.savefig(os.path.join(results_dir, 'pr_curve{0}.png'.format('' if bwo else '')))
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=plot_params[dataset]['figsize'])
    ax = plt.gca()
    roc_curve(labels, prob, list(range(len(DATASET[dataset]))), ax=ax)
    plt.savefig(os.path.join(results_dir, 'roc_curve{0}.png'.format('' if bwo else '')))
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    fes, lbs, dt = get_data('UCM')
    classify(fes, lbs, dt, bwo=True)
    fes, lbs, dt = get_data('AID')
    classify(fes, lbs, dt, bwo=True)
