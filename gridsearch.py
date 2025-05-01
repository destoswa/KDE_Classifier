import os
import time
import csv
import pandas as pd
from train import training
from src.visualization import show_log_train, show_grid_search


def main():
    lst_kernel_sizes = range(1, 3)
    lst_repeat_kernel = range(1, 3)
    args_training = {
        'do_update_caching': True,
        'do_preprocess': False,
        'frac_train': .8,
        'do_continue_from_existing_model': False,
        'num_class': 3,
        'num_epoch': 1,
        'batch_size': 12,
        'num_workers': 12,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'kernel_size': 1,
        'num_repeat_kernel': 1,
        'grid_size': 64,
        'frac_training': .01,
        'frac_testing': .01,
    }

    # create folder of logs:
    version = 0
    while os.path.exists('./log/grid_train_' + str(version)):
        version += 1
    log_gridsearch_root = './log/grid_train_' + str(version) + '/'
    os.mkdir(log_gridsearch_root)

    # create file for results:
    df_res = pd.DataFrame(columns=['kernel_size', 'number_repeat', 'overall_acc', 'average_acc'])
    total_time = time.time()

    # grid-search
    for kernel_size in lst_kernel_sizes:
        for repeat_kernel in lst_repeat_kernel:
            print("========================================================")
            print(f"========== TRAINING WITH {repeat_kernel} KERNELS OF SIZE {kernel_size} ==========")
            print("========================================================")
            args_training['kernel_size'] = kernel_size
            args_training['num_repeat_kernel'] = repeat_kernel

            # create csv of logs:
            version = f"ks={kernel_size}_rk={repeat_kernel}"
            log_file_root = log_gridsearch_root + version + '/'
            os.mkdir(log_file_root)
            log_file = log_file_root + '/logs.csv'
            with open(log_file, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(['train_acc', 'train_loss', 'test_acc', 'test_class_acc', 'test_loss'])

            # training
            start_time = time.time()
            OA, AA = training(log_file_root, args_training)
            end_time = time.time()

            # saving results
            show_log_train(log_file, log_file_root, do_save=True, do_show=False)
            df_res.loc[len(df_res)] = [kernel_size, repeat_kernel, OA, AA]

            # print time of training
            delta_time = end_time - start_time
            n_hours = int(delta_time / 3600)
            n_min = int((delta_time % 3600) / 60)
            n_sec = int(delta_time - n_hours * 3600 - n_min * 60)
            print("\n==============\n")
            print(f"TIME TO TRAIN: {n_hours}:{n_min}:{n_sec}")

    # print total time of training
    delta_time = time.time() - total_time
    n_hours = int(delta_time / 3600)
    n_min = int((delta_time % 3600) / 60)
    n_sec = int(delta_time - n_hours * 3600 - n_min * 60)
    print("\n==============\n")
    print(f"TIME TO GRID SEARCH: {n_hours}:{n_min}:{n_sec}")

    # save results of grid search
    df_res.to_csv(log_gridsearch_root + 'log_grid_search.csv', sep=';', index=False)
    show_grid_search(
        log_gridsearch_root,
        df_res[['kernel_size', 'number_repeat', 'overall_acc']],
        'Overall Accuracy',
        do_save=True,
        do_show=False,
    )
    show_grid_search(
        log_gridsearch_root,
        df_res[['kernel_size', 'number_repeat', 'average_acc']],
        'Average Accuracy',
        do_save=True,
        do_show=False,
    )


if __name__ == '__main__':
    main()
