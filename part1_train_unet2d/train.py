import random
import os
from os import makedirs
from os.path import join, exists, isdir
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import sys, importlib
from pathlib import Path



# https://qastack.ru/programming/16981921/relative-imports-in-python-3
def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    # Additionally remove the current file's directory from sys.path
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)  # won't be needed after that


# Инициализация проекта и библиотеки tf должны происходить после настройки ус-ва
def setup_seed(seed):
    random.seed(seed)  # Set random seed for Python
    np.random.seed(seed)  # Set random seed for numpy
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

def main(
        device, global_random_seed, init_random_seed,
        n_filters, separable_conv, shortcut_connection,
        conv_bias, bn_before_act, decoder_method, upconv_bias,
        pretrained_model_dir, trained_model_dir, epochs, early_loss_val,
        retrain_decoder, base_decoder, base_upconv_bias,
        data_dir_patch, train_dir_name, validate_dir_name,
        train_batch_size, validate_batch_size,
        tuned_hyper_pattern, save_history, history_dir_path
):
    print(
        f'\n\t\tdevice = {device}',
        f'global_random_seed={global_random_seed}',
        f'init_random_seed={init_random_seed}',
        f'n_filters={n_filters}',
        f'separable_conv={separable_conv}',
        f'shortcut_connection={shortcut_connection}',
        f'conv_bias={conv_bias}',
        f'bn_before_act={bn_before_act}',
        f'decoder_method={decoder_method}\n',
        f'upconv_bias={upconv_bias}',
        f'pretrained_model_dir={pretrained_model_dir}',
        f'trained_model_dir={trained_model_dir}',
        f'epochs={epochs}',
        f'early_loss_val={early_loss_val}',
        f'retrain_decoder={retrain_decoder}',
        f'base_decoder={base_decoder}',
        f'base_upconv_bias={base_upconv_bias}',
        f'data_dir_patch={data_dir_patch}',
        f'train_dir_name={train_dir_name}',
        f'validate_dir_name={validate_dir_name}',
        f'train_batch_size={train_batch_size}',
        f'validate_batch_size={validate_batch_size}',
        f'tuned_hyper_pattern={tuned_hyper_pattern}',
        f'save_history={save_history}',
        f'history_dir_path={history_dir_path}\n', sep='\n\t\t'
    )

    print()

    setup_seed(global_random_seed)

    print("---------------------------------------------------")
    print("Loading datasets")
    aug_dict = dict()

    train_gene = get_image_mask_generator2d(
        join(data_dir_patch, train_dir_name), 'imgs', 'masks',
        color_mode="grayscale", target_size=(512, 512), aug_dict=aug_dict,
        batch_size=train_batch_size, seed=607
    )
    validate_gene = get_image_mask_generator2d(
        join(data_dir_patch, validate_dir_name), 'imgs', 'masks',
        color_mode="grayscale", target_size=(512, 512), aug_dict=aug_dict,
        batch_size=validate_batch_size, seed=607
    )

    model_name = get_model_name(
            n_filters, separable_conv, shortcut_connection,
            conv_bias, bn_before_act, decoder_method, upconv_bias,
            global_random_seed, init_random_seed
    )
    print(model_name)

    pretrained_model_path = None
    if pretrained_model_dir:
        if not isdir(pretrained_model_dir):
            if retrain_decoder:
                base_model_name = get_model_name(
                    n_filters, separable_conv, shortcut_connection,
                    conv_bias, bn_before_act, base_decoder, base_upconv_bias,
                    global_random_seed, init_random_seed
                )
                pretrained_model_path = join(pretrained_model_dir, f'{base_model_name}.hdf5')
            else:
                pretrained_model_path = join(pretrained_model_dir, f'{model_name}.hdf5')
        else:
            pretrained_model_path = pretrained_model_dir

    loss = 'binary_crossentropy'
    metrics = [BinaryMeanIOU(2, name=None, dtype=None)]
    lr = 1e-4 # Максимальная скорость, пока loss не достигнет значения early_loss
    # Затем максимальная скорость будет понижена до lr = 1e-5
    print("\n-------------------------------------------")
    print(f"Train full model with Adam lr = {lr} for {epochs} epochs")
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model = UNet2D(
        optimizer=optimizer, loss=loss, metrics=metrics,
        retrain_decoder=retrain_decoder,
        pretrained_weights=pretrained_model_dir,
        input_size=(None, None, 1), n_filters=n_filters,
        separable_conv=separable_conv, shortcut_connection=shortcut_connection,
        bn_before_act=bn_before_act, use_conv_bias=conv_bias,
        decoder_method=decoder_method, use_upconv_bias=upconv_bias,
        pretrained_decoder_method=base_decoder, pretrained_use_upconv_bias=base_upconv_bias,
        init_seed=init_random_seed
    )
    model.summary()
    print("(LOG) Model count params:", model.count_params())

    if retrain_decoder:
        print("\n-------------------------------------------")
        print(f"Train decoder part by {epochs // 3} epochs with Adam lr = 1e-4")
        freezeEncoder(model, bn_before_act=bn_before_act)

        callbacks = [
            ModelCheckpoint(join(
                trained_model_dir, 'retrained_decoder', f'{model_name}.hdf5'), monitor='val_loss', verbose=1,
                save_best_only=True, mode='min'
            )
        ]
        if device == 'gpu':
            history = model.fit(
                x=train_gene,
                steps_per_epoch=304 // train_batch_size,
                validation_data=validate_gene,
                validation_steps=140 // validate_batch_size,
                epochs=epochs // 3,
                shuffle=True,
                verbose=1,
                initial_epoch=0,
                callbacks=callbacks,
                use_multiprocessing=False,
                workers=1)
        else:
            history = model.fit(
                x=train_gene,
                steps_per_epoch=304 // train_batch_size,
                validation_data=validate_gene,
                validation_steps=140 // validate_batch_size,
                epochs=epochs // 3,
                shuffle=True,
                verbose=1,
                initial_epoch=0,
                callbacks=callbacks,
                use_multiprocessing=False,
                workers=1)
        if save_history:
            histoty_to_csv(
                history, epochs // 3, tuned_hyper_pattern,
                n_filters, bn_before_act, conv_bias,
                join(history_dir_path, "retrained_decoder"), global_random_seed,
                init_random_seed
            )
        unfreezeEncoder(model, bn_before_act=bn_before_act)
        epochs -= epochs // 3

    if epochs > 0:
        callbacks = [
            ModelCheckpoint(join(trained_model_dir, f'{model_name}.hdf5'), monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min'),
            EarlyStoppingByLossVal(monitor='val_loss', value=early_loss_val, verbose=1),
        ]
        if device == 'gpu':
            history = model.fit(
                x=train_gene,
                steps_per_epoch=304 // train_batch_size,
                validation_data=validate_gene,
                validation_steps=140 // validate_batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                initial_epoch=0,
                callbacks=callbacks,
                use_multiprocessing=False,
                workers=1)
        else:
            history = model.fit(
                x=train_gene,
                steps_per_epoch=304 // train_batch_size,
                validation_data=validate_gene,
                validation_steps=140 // validate_batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                initial_epoch=0,
                callbacks=callbacks,
                use_multiprocessing=False,
                workers=1)
        if save_history:
            histoty_to_csv(
                history, epochs, tuned_hyper_pattern,
                n_filters, bn_before_act, conv_bias,
                history_dir_path, global_random_seed,
                init_random_seed
            )

        print("\n-------------------------------------------")
        initial_epoch = len(history.history["loss"])
        print(f"Train full model with Adam lr = {lr / 10} for {epochs - initial_epoch} epochs")
        if epochs - initial_epoch > 0:
            K.set_value(model.optimizer.learning_rate, lr / 10)
            callbacks = [
                ModelCheckpoint(join(trained_model_dir, f'{model_name}.hdf5'), monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min'),
                EarlyStopping(monitor='val_loss', verbose=1, patience=20),
            ]

            if device == 'gpu':
                history = model.fit(
                    x=train_gene,
                    steps_per_epoch=304 // train_batch_size,
                    validation_data=validate_gene,
                    validation_steps=140 // validate_batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=1,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                    use_multiprocessing=False,
                    workers=1)
            else:
                history = model.fit(
                    x=train_gene,
                    steps_per_epoch=304 // train_batch_size,
                    validation_data=validate_gene,
                    validation_steps=140 // validate_batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=1,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                    use_multiprocessing=False,
                    workers=1)
            if save_history:
                histoty_to_csv(
                    history, epochs, tuned_hyper_pattern,
                    n_filters, bn_before_act, conv_bias,
                    history_dir_path, global_random_seed,
                    init_random_seed,
                    initial_epoch=initial_epoch
                )


if __name__ == '__main__':
    import_parents(level=1)
    import_parents(level=2)
    from ..decoder_methods import *
    build_deconv_methods(case2d=True) # Теперь доступные методы декодирования перечислены в списке DECODER_METHODS
    # print(f"LOG: DECODER METHODS for 2D-models: {DECODER_METHODS}")
    parser = argparse.ArgumentParser()
    # -----------------------------------------------------------------------------------------
    # Настройка окружения и инициализации обучаемых параметров
    parser.add_argument('-cpu', '--cpu', default=1, type=int, choices=[0, 1], required=False)
    parser.add_argument('-global_r_s', '--global_random_seed', default=156, type=int, required=False)
    parser.add_argument('-init_r_s', '--init_random_seed', default=75, type=int, required=False)
    # -----------------------------------------------------------------------------------------
    # Параметры гипернастройки
    parser.add_argument('-n_f', '--n_filters', type=int, required=True)
    parser.add_argument('-sep_conv', '--separable_conv', default=0, type=int, choices=[0, 1], required=False)
    parser.add_argument('-short_con', '--shortcut_connection', default=0, type=int, choices=[0, 1], required=False)
    parser.add_argument('-conv_bias', '--conv_bias', default=0, type=int, choices=[0, 1], required=False)
    parser.add_argument('-bn_bef', '--bn_before_act', default=0, type=int, choices=[0, 1], required=False)
    parser.add_argument('-dec', '--decoder', default='nearest', type=str,
                        choices=DECODER_METHODS, required=False
                        )
    parser.add_argument('-upconv_bias', '--upconv_bias', default=0, type=int, choices=[0, 1], required=False)
    # -----------------------------------------------------------------------------------------
    # Настройка процесса обучения модели
    # ToDo: По умолчанию - дообучение на предобученных весах (после переобучения декодера) - плохо, скорее всего
    # ToDo: Требуется добавить предупреждение, если путь сохранения совпадает с предобученной моделью \
    #       и предупреждение, если предобученных весов нет
    # Можно указывать полный путь к чекпоинту
    # Если задать -pre_mod_dir=None, тогда модель будет обучена заново
    parser.add_argument('-pre_mod_dir', '--pretrained_model_dir', default=None, type=str, required=False)
    parser.add_argument('-mod_dir', '--trained_model_dir', default=join("..", "checkpoints"), type=str, required=False)
    parser.add_argument('-epochs', '--epochs', type=int, required=True)
    parser.add_argument('-early_loss', '--early_loss', type=float, default=0.0799, required=False)
    # -----------------------------------------------------------------------------------------
    parser.add_argument('-retrain_dec', '--retrain_decoder', default=0, type=int, required=False)
    parser.add_argument('-base_dec', '--base_decoder', default='nearest', type=str,
                        choices=DECODER_METHODS, required=False
                        )
    parser.add_argument('-base_upconv_bias', '--base_upconv_bias', default=0, type=int, choices=[0, 1], required=False)
    # -----------------------------------------------------------------------------------------
    # Наборы данных и сохранение результата
    parser.add_argument('-data_dir', '--data_dir_patch', default=join('..', '..', '..', f'samples1'), type=str, required=False)
    parser.add_argument('-train_dir', '--train_dir_name', default='train', type=str, required=False) # ToDo Условие - not None?
    parser.add_argument('-val_dir', '--validate_dir_name', default='validate', type=str, required=False)
    parser.add_argument('-trn_batch', '--train_batch_size', default=16, type=int, required=False)
    parser.add_argument('-val_batch', '--validate_batch_size', default=10, type=int, required=False)
    # ToDo - добавить тип выгрузки результата в pd (Дозапись, новый файл)
    parser.add_argument('-save_history', type=int, default=1, choices=[0, 1], required=False)
    parser.add_argument('-history_dir', '--history_dir_path', default=join('.', 'histories'), type=str, required=False)

    args = parser.parse_args()
    if args.cpu:
        device = 'cpu'
    else:
        device = 'gpu'

    if args.bn_before_act and args.conv_bias:
        args.conv_bias = 0
        print("LOG INFO: conv_bias is skipped because there is an bias in bn_before_act")

    tuned_hyper_pattern = f"n_f={args.n_filters}"
    if args.separable_conv:
        tuned_hyper_pattern += "xsep_conv2D"
    else:
        tuned_hyper_pattern += "xconv2D"
    if args.shortcut_connection:
        tuned_hyper_pattern += "xshort_con"

    if args.bn_before_act:
        tuned_hyper_pattern += "xbn_bef"
    elif args.conv_bias:
        tuned_hyper_pattern += f"xconv_bias"

    tuned_hyper_pattern += f"x{args.decoder}"

    if args.upconv_bias:
        tuned_hyper_pattern += "xupconv_bias"

    print(f"tuned_hyper_pattern is '{tuned_hyper_pattern}'")
    # ToDo - вынести в метод
    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print("Training on GPU...")
        import tensorflow as tf

        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import EarlyStopping
    from ..models2d_utils.unet2d_model import *
    from ..models2d_utils.generator2d import *
    from ..models_utils.make_history import *
    from ..models_utils.metrics import *
    from ..models_utils.utils import *
    from tensorflow.keras import backend as K

    if not exists(args.trained_model_dir):
        makedirs(args.trained_model_dir)

    # ToDo - exist and not empty?
    assert exists(args.data_dir_patch), f"Argument Error: There is no directory {args.data_dir} for train/val datasets"
    assert exists(join(args.data_dir_patch, args.train_dir_name)), \
        f"Argument Error: There is no directory {join(args.data_dir_patch, args.train_dir_name)} for train dataset"
    assert exists(join(args.data_dir_patch, args.validate_dir_name)), \
        f"Argument Error: There is no directory {join(args.data_dir_patch, args.validate_dir_name)} for validate dataset"

    if not exists(args.history_dir_path):
        makedirs(args.history_dir_path)

    #ToDo - mkdir retrained_decoder

    main(
        device, args.global_random_seed, args.init_random_seed,
        args.n_filters, args.separable_conv, args.shortcut_connection,
        args.conv_bias, args.bn_before_act, args.decoder, args.upconv_bias,
        args.pretrained_model_dir, args.trained_model_dir, args.epochs, args.early_loss,
        args.retrain_decoder, args.base_decoder, args.base_upconv_bias,
        args.data_dir_patch, args.train_dir_name, args.validate_dir_name,
        args.train_batch_size, args.validate_batch_size,
        tuned_hyper_pattern, args.save_history, args.history_dir_path
    )