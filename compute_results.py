import numpy as np
from scipy import stats
import pandas as pd
import json
import matplotlib.pyplot as plt
import statistics
from math import isnan, sqrt
from sklearn.metrics import mean_squared_error

__author__ = "Sasi Kiran Gaddipati"
__credits__ = ["Tim Metzler"]
__license__ = ""
__version__ = "1.0.1"
__email__ = "sasi-kiran.gaddipati@smail.inf.h-brs.de"
__last_modified__ = "04.04.2021"
__status__ = "Prototype"

def center_of_measure(count_tuple):
    data = []
    for tuple in count_tuple:
        value = tuple[0]
        count = tuple[1]
        value_list = [value] * count
        data.extend(value_list)
    return statistics.mean(data), statistics.median(data), statistics.mode(data), statistics.stdev(data)

def pearson_correlation(x, y):
    return np.corrcoef(np.asarray(x), np.asarray(y))

def spearman_r(x, y):
    return stats.spearmanr(np.asarray(x), np.asarray(y))

def individual_results(df):

    id_list = df.id.to_list()
    rho_dict = {}
    rs_dict = {}
    rmse_dict = {}

    for id in id_list:
        actual = list(df.loc[df["id"] == id, "score_avg"])
        predicted = list(df.loc[df["id"] == id, "score_afar"])

        # actual = list(df.loc[df["id"] == id, "score_me"])
        # predicted = list(df.loc[df["id"] == id, "score_other"])

        rho = pearson_correlation(actual, predicted)[0][1]
        rs = spearman_r(actual, predicted)[0]
        rmse = sqrt(mean_squared_error(actual, predicted))
        rho_dict[id] = float(rho)
        rs_dict[id] = float(rs)
        rmse_dict[id] = float(rmse)

    return rho_dict, rs_dict, rmse_dict

def plot_correlations(id_dict):
    plt.rc('font', family='serif', weight='bold')
    plt.rc('lines', lw=2)
    # plt.rc('xtick', labelsize='large')
    # plt.rc('ytick', labelsize='large')
    plt.rc('axes', linewidth=2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.5
    # pos = np.arange(len(id_dict))

    ax.grid(linestyle='--')
    # ax.set_xticks(pos + (width/2))
    # ax.set_xticklabels(id_dict.keys())

    plt.bar(id_dict.keys(), id_dict.values(), width, color='b')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.ylabel('Spearman correlation', fontsize=12, weight='bold')
    plt.xlabel('id', fontsize=12, weight='bold')

    plt.show()


# def change_range(x):
#
#     scores_range = {0:0,1:0,2:1,3:1,4:2,5:2}
#
#     for i in range(0, len(x)):
#         x[i] = scores_range[round(x[i],0)]
#
#     return x

PATH = "results/nn/"
#
df = pd.read_csv(PATH+"feedback.csv", delimiter=",")
# actual = df.score_other.to_list()
# predicted = df.score_me.to_list()
#
# print(pearson_correlation(actual, predicted))
# print(spearman_r(actual, predicted))
id_rhos, id_rss, id_rmse = individual_results(df)
print(statistics.mean(list(id_rmse.values())))
print(statistics.median(list(id_rmse.values())))
print(sorted(list(id_rmse.values())))
id_rhos = {k: id_rhos[k] for k in id_rhos if not isnan(id_rhos[k])}
id_rss = {k: id_rss[k] for k in id_rss if not isnan(id_rss[k])}
print(statistics.mean(list(id_rhos.values())))
print(statistics.mean(list(id_rss.values())))
# print(statistics.median(list(id_rmse.values())))
# save_file = "results/mohler/rmse_scores.json"
#
# with open(save_file, 'w+') as json_file:
#     json.dump(id_rmse, json_file)

#print(sum(values)/len(values))
print("Done")

# print(pearson_correlation(actual, predicted))
# print(spearman_r(actual, predicted))

# print(center_of_measure([(1, 6), (2, 6), (3, 15), (4, 25), (5, 52)]))
# print(center_of_measure([(1, 15), (2, 8), (3, 10), (4, 15), (5, 56)]))
# print(center_of_measure([(1, 17), (2, 7), (3, 5), (4, 19), (5, 56)]))
# print(center_of_measure([(1, 5), (2, 5), (3, 4), (4, 17), (5, 73)]))
# print(center_of_measure([(1, 12), (2, 19), (3, 72)]))