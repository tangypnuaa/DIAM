import os
import numpy as np
from matplotlib import pyplot as plt


def plot_lc(dataset_name, method_arr: 'list', legend=True, fold=10, model_num=12):
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize

    batch_size = 1500

    fig, ax = plt.subplots()
    for mth_id, mth in enumerate(method_arr):
        x_all = []
        y_all = []
        err_bar_all = []
        for fo in range(fold):
            results_saving_dir = f"./extracted_results/{model_num}/{dataset_name}/{mth}/{fo}"
            acc_file = os.path.join(results_saving_dir, "performances.txt")
            acc = np.loadtxt(acc_file, dtype=float)
            mx = np.mean(acc[:, 0])
            err_bar = np.std(acc[:, 0])
            y_all.append(mx)
            x_all.append(batch_size * fo)
            err_bar_all.append(err_bar)

        ax.plot(x_all, y_all, label=methods_label[mth_id],
                linewidth=methods_linewodth[mth_id],
                color=methods_color[mth_id],
                linestyle=methods_lstyle[mth_id],
                marker=methods_marker[mth_id],
                markersize=4
                )
        ax.errorbar(x_all, y_all, yerr=err_bar_all, fmt='none', ecolor=methods_color[mth_id],
                    alpha=0.4, lolims=[True] * len(err_bar_all), uplims=[False] * len(err_bar_all))
        ax.errorbar(x_all, y_all, yerr=err_bar_all, fmt='none', ecolor=methods_color[mth_id],
                    alpha=0.4, lolims=[False] * len(err_bar_all), uplims=[True] * len(err_bar_all))

        if legend:
            ax.legend(loc='lower right')
    plt.xlabel("number of queries")
    plt.ylabel("mean accuracy")
    if dataset_name == "mnist":
        plt.ylim(90, 99.5)
    else:
        plt.ylim(59, 97)
    plt.xticks(np.arange(0, 15001, 3000))
    # plt.title(str(dataset_name))
    # ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')

    fig.tight_layout()
    fig.savefig(f'fig/lc_{dataset_name}_{model_num}.pdf', dpi=200, bbox_inches='tight')
    fig.show()


datasets = ['mnist', 'kmnist']
methods = ['DIAM', 'CAL', 'entropy', 'margin', 'least_conf', 'coreset', 'random']
methods_label = ['DIAM', 'CAL', 'Entropy', 'Margin', 'Least conf.', 'Coreset', 'Random']
methods_linewodth = [2.8, 1.7, 1.7, 1.7, 1.7, 1.5,
                     1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.7]
methods_lstyle = ['-', '--', '--',
                  '--', '--',
                  '--', '--', '--', '--', '--']
methods_color = ['#F71E35', '#274c5e', '#0080ff',
                 '#bf209f', '#79bd9a', 'gray', 'black', '#679b00', 'black']
methods_marker = ["D", "o", "^", "^",
                  "^", "o", "^", "^"]
os.makedirs("./fig", exist_ok=True)
for did, da in enumerate(datasets):
    plot_lc(dataset_name=da, method_arr=methods, legend=True, fold=11, model_num=12)
