import numpy as np
from os.path import join, exists
import pandas as pd


def histoty_to_csv(history, epochs, tuned_hyper_pattern,
                   n_filters, bn_before_act, use_conv_bias,
                   history_dir_path, global_random_seed,
                   init_random_seed,
                   initial_epoch=0):

    keys = dict()
    for key in history.history.keys():
        if key[-1].isdigit():
            keys[key[:key.rfind('_')]] = key
        else:
            keys[key] = key
    print(keys)


    y_loss = np.array(history.history[keys["loss"]])[None, :]
    y_val_loss = np.array(history.history[keys["val_loss"]])[None, :]
    y_binary_mean_iou = np.array(history.history[keys["binary_mean_iou"]])[None, :]
    y_val_binary_mean_iou = np.array(history.history[keys["val_binary_mean_iou"]])[None, :]
    x = list(range(initial_epoch, y_loss.shape[1] + initial_epoch))

    data_loss_pd = pd.DataFrame(data=y_loss, columns=x, index=[tuned_hyper_pattern])
    data_val_loss_pd = pd.DataFrame(data=y_val_loss, columns=x, index=[f"{tuned_hyper_pattern} (val)"])

    data_metric_pd = pd.DataFrame(data=y_binary_mean_iou, columns=x, index=[tuned_hyper_pattern])
    data_val_metric_pd = pd.DataFrame(data=y_val_binary_mean_iou, columns=x,
                                      index=[f"{tuned_hyper_pattern} (val)"])

    '''
    write_columns = not exists(
        join(history_dir_path,
             f'loss__n_f={n_filters}__{"conv_bias__" if use_conv_bias else ""}{"bn_bef__" if bn_before_act else ""}{global_random_seed}grs_{init_random_seed}irs.csv')
    )
    '''
    write_columns = True

    data_loss_pd.to_csv(
        join(
            history_dir_path,
            f'loss__n_f={n_filters}__{"conv_bias__" if use_conv_bias else ""}{"bn_bef__" if bn_before_act else ""}{global_random_seed}grs_{init_random_seed}irs.csv'),
        index=True, mode='a', header=write_columns)
    data_val_loss_pd.to_csv(
        join(
            history_dir_path,
            f'val_loss__n_f={n_filters}__{"conv_bias__" if use_conv_bias else ""}{"bn_bef__" if bn_before_act else ""}{global_random_seed}grs_{init_random_seed}irs.csv'),
        index=True, mode='a', header=write_columns)
    data_metric_pd.to_csv(
        join(
            history_dir_path,
            f'binary_mean_iou__n_f={n_filters}__{"conv_bias__" if use_conv_bias else ""}{"bn_bef__" if bn_before_act else ""}{global_random_seed}grs_{init_random_seed}irs.csv'),
        index=True, mode='a', header=write_columns)
    data_val_metric_pd.to_csv(
        join(
            history_dir_path,
            f'val_binary_mean_iou__n_f={n_filters}__{"conv_bias__" if use_conv_bias else ""}{"bn_bef__" if bn_before_act else ""}{global_random_seed}grs_{init_random_seed}irs.csv'),
        index=True, mode='a', header=write_columns)
