import os
import numpy as np


def tex_table(dataset_name, method_arr: 'list',
              fold=10, model_num=(2, 4, 6, 8, 12)):
    mean_mat = np.zeros([7, 5])
    std_mat = np.zeros([7, 5])
    best_mat = np.zeros([7, 5])
    for mn_id, mn in enumerate(model_num):
        cur_mth = 0
        for mth_id, mth in enumerate(method_arr):
            y_all = []
            std_all = []
            for fo in range(fold):
                results_saving_dir = f"./extracted_results/{mn}/{dataset_name}/{mth}/{fo}"
                acc_file = os.path.join(results_saving_dir, "performances.txt")
                acc = np.loadtxt(acc_file, dtype=float)
                mx = np.mean(acc[:, 0])
                mx2 = np.std(acc[:, 0])
                y_all.append(mx)
                std_all.append(mx2)

            mean_mat[cur_mth, mn_id] = np.mean(y_all)
            std_mat[cur_mth, mn_id] = np.mean(std_all)

            cur_mth += 1

        best_idx = np.argmax(mean_mat[:, mn_id])
        # print(mean_mat[:, mn_id])
        # print(best_idx)
        best_mat[best_idx, mn_id] = 1

    # to tex
    for mth_id, mth in enumerate(method_arr):
        cur_mth = mth_id
        print(f"{methods_label[mth_id]} ", end='')
        for mn_id, mn in enumerate(model_num):
            if best_mat[cur_mth, mn_id] == 1:
                print(f"& $\\bm{'{'}{mean_mat[cur_mth, mn_id]:.2f} \\pm {std_mat[cur_mth, mn_id]:.2f}{'}'}$ ", end='')
            else:
                print(f"& ${mean_mat[cur_mth, mn_id]:.2f} \\pm {std_mat[cur_mth, mn_id]:.2f}$ ", end='')
        print(" \\\\")


datasets = ['mnist', 'kmnist']
methods = ['DIAM', 'CAL', 'entropy', 'margin', 'least_conf', 'coreset', 'random']
methods_label = ['DIAM', 'CAL', 'Entropy', 'Margin', 'Least conf.', 'Coreset', 'Random']
for did, da in enumerate(datasets):
    print(da)
    tex_table(dataset_name=da, method_arr=methods,
              fold=11, model_num=(2, 4, 6, 8, 12))
