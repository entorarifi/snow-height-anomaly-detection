# %%
import os
import re

import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils import format_with_border


class MonteCarloDropout(keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def get_features_and_targets(df, split_index, feature_columns, target_column, mean=None, std=None, scale=True):
    if (mean is None) ^ (std is None):
        raise Exception('mean and std must both be set or unset')

    features = df[feature_columns].values
    targets = None if target_column is None else df[target_column].values

    if not scale:
        return features, targets, None, None

    if mean is None:
        mean = features[:split_index].mean(axis=0)
        features -= mean
        std = features[:split_index].std(axis=0)
        features /= std
    else:
        features = (features - mean) / std

    return features, targets, mean, std


def sequence_debugging():
    int_sequence = np.arange(30)

    train_end_index = int(0.5 * len(int_sequence))
    val_end_index = int(0.75 * len(int_sequence))

    sequence_length = 3
    target_start_idx = sequence_length - 1

    train_dataset = keras.utils.timeseries_dataset_from_array(
        data=int_sequence,
        targets=int_sequence[target_start_idx:],
        sequence_length=sequence_length,
        start_index=0,
        end_index=train_end_index
    )

    val_dataset = keras.utils.timeseries_dataset_from_array(
        data=int_sequence,
        targets=int_sequence[target_start_idx:],
        sequence_length=sequence_length,
        start_index=train_end_index,
        end_index=val_end_index
    )

    test_dataset = keras.utils.timeseries_dataset_from_array(
        data=int_sequence,
        targets=int_sequence[target_start_idx:],
        sequence_length=sequence_length,
        start_index=val_end_index
    )

    print("Train:")
    for inputs, targets in train_dataset:
        for i in range(inputs.shape[0]):
            print([int(x) for x in inputs[i]], int(targets[i]))

    print("\nVal:")
    for inputs, targets in val_dataset:
        for i in range(inputs.shape[0]):
            print([int(x) for x in inputs[i]], int(targets[i]))

    print("\nTest:")
    for inputs, targets in test_dataset:
        for i in range(inputs.shape[0]):
            print([int(x) for x in inputs[i]], int(targets[i]))


def create_dataset(
        features,
        targets,
        sequence_length,
        target_start_index,
        batch_size,
        shuffle=False,
        start_index=None,
        end_index=None
):
    return keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=None if targets is None else targets[target_start_index:],
        sequence_length=sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=shuffle,
        start_index=start_index,
        end_index=end_index
    )


def create_model(architecture, input_shape, dropout_rate, logging=None, summary=True):
    arch_split = architecture.split('-')
    dense = True
    bidirectional = False
    layers = []

    digits_pattern = re.compile(r"\d+")

    if 'l' in arch_split[0]:
        rnn_layer = 'LSTM'
    elif 'g' in arch_split[0]:
        rnn_layer = 'GRU'
    else:
        raise Exception('rnn_layers must be one of [LSTM, GRU]')

    if 'b' in arch_split[0]:
        bidirectional = True

    for i, layer in enumerate(reversed(arch_split)):
        no_units = int(digits_pattern.findall(layer)[0])

        if dense:
            activation = 'sigmoid' if i == 0 else 'relu'
            layers.append(keras.layers.Dense(no_units, activation=activation))
        else:
            args = {
                'units': no_units,
            }

            if '(d)' not in arch_split[-i]:
                args['return_sequences'] = True

            if i == len(arch_split) - 1:
                args['input_shape'] = input_shape

            current_layer = keras.layers.LSTM(**args) if rnn_layer == 'LSTM' else keras.layers.GRU(**args)

            if bidirectional:
                current_layer = keras.layers.Bidirectional(current_layer)

            layers.extend([
                MonteCarloDropout(dropout_rate),
                current_layer
            ])

        if '(d)' in layer:
            dense = False

    layers.reverse()

    if summary:
        logging.info(format_with_border('Model Summary'))
        for layer in layers:
            layer_type = str(type(layer))

            if 'Dense' in str(type(layer)):
                logging.info(type(layer), layer.units, layer.activation)
            elif 'Bidirectional' in str(type(layer)):
                logging.info(type(layer), layer.layer, layer.layer.units, layer.layer.activation,
                             layer.layer.return_sequences)
            elif 'LSTM' in str(type(layer)):
                logging.info(type(layer), layer.units, layer.activation, layer.return_sequences)
            elif 'GRU' in str(type(layer)):
                logging.info(type(layer), layer.units, layer.activation, layer.return_sequences)
            elif 'Dropout' in layer_type:
                logging.info(type(layer), layer.rate)

    return keras.Sequential(layers)


def plot_keras_history(history):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_data(dfs, y='HS', target='no_snow', predictions=[]):
    if not isinstance(dfs, list):
        dfs = [dfs]

    if not isinstance(predictions, list):
        predictions = [predictions] if predictions is not None else []

    if not isinstance(y, list):
        y = [y] * len(dfs)

    if not isinstance(target, list):
        target = [target] * len(dfs)

    if len(predictions):
        rows = len(dfs) * 2
    else:
        rows = len(dfs)

    fig, axes = plt.subplots(rows, 1, figsize=(20, 5 * rows))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, df in enumerate(dfs):
        plot_df = df.copy()
        plot_df.index = pd.to_datetime(plot_df['measure_date'])
        station_code = plot_df['station_code'].iloc[0]

        # Plot original data
        ax = axes[2 * i] if len(predictions) else axes[i]
        ax.set_xlabel('Date')
        ax.set_ylabel(y[i])
        ax.set_title(f'Station: {station_code} | Original Data with Anomalies Highlighted')
        ax.plot(plot_df.index, plot_df[y[i]], label='Data', marker='o', linestyle='-', ms=4)
        ax.scatter(plot_df[plot_df[target[i]]].index, plot_df[plot_df[target[i]]][y[i]], color='red', label='Anomalies',
                   zorder=5, s=20)
        ax.legend()

        # Plot predictions
        if len(predictions):
            ax = axes[2 * i + 1]
            ax.set_xlabel('Date')
            ax.set_ylabel(y[i])
            ax.set_title(f'Station: {station_code} | Predicted Data with Anomalies Highlighted')
            ax.plot(plot_df.index, plot_df[y[i]], label='Data', marker='o', linestyle='-', ms=4)
            ax.scatter(plot_df[predictions[i]].index, plot_df[predictions[i]][y[i]], color='green',
                       label='Predicted Anomalies', zorder=5, s=20)
            ax.legend()

    plt.tight_layout()
    plt.show()


def predict_with_uncertainty(model, x, n_iter=100):
    predictions = np.array([model.predict(x, verbose=0) for _ in range(n_iter)])
    uncertainty = np.std(predictions, axis=0)
    scaled_uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    return predictions.mean(axis=0).reshape(-1), scaled_uncertainty.mean()


def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def preprocces(df):
    df['measure_date'] = pd.to_datetime(df['measure_date'])
    df['year'] = df['measure_date'].dt.year
    df['month'] = df['measure_date'].dt.month
    df['day'] = df['measure_date'].dt.day
    df['weekday'] = df['measure_date'].dt.weekday
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    return df


def create_train_val_datasets(
        dfs,
        split_percentage,
        feature_columns,
        target_column,
        sequence_length,
        target_start_index,
        batch_size,
        logging=None,
        mlflow=None
):
    train_df, val_df = pd.DataFrame(), pd.DataFrame()

    for df in dfs:
        split_index = int(len(df) * split_percentage)

        train_df = pd.concat([train_df, df[:split_index]])
        val_df = pd.concat([val_df, df[split_index:]])

    combined_df = preprocces(pd.concat([train_df, val_df]))
    split_index = len(train_df)

    # TODO: Move this to calling function
    logging.info(f'Training samples: {len(train_df)}')
    logging.info(f'Validation samples: {len(val_df)}')

    mlflow.log_param('al_training_samples', len(train_df))
    mlflow.log_param('al_validation_samples', len(val_df))

    features, targets, mean, std = get_features_and_targets(combined_df, split_index, feature_columns, target_column)

    train_dataset = create_dataset(
        features, targets, sequence_length, target_start_index, batch_size, end_index=split_index, shuffle=True
    )

    val_dataset = create_dataset(
        features, targets, sequence_length, target_start_index, batch_size, start_index=split_index, shuffle=True
    )

    return train_dataset, val_dataset, mean, std


def create_test_dataset(
        df, feature_columns, target_column, sequence_length, target_start_index, batch_size, mean, std
):
    df = preprocces(df)
    features, targets, _, _ = get_features_and_targets(df, None, feature_columns, target_column, mean, std)
    test_dataset = create_dataset(features, targets, sequence_length, target_start_index, batch_size)
    return test_dataset


def create_test_datasets(
        dfs, feature_columns, target_column, sequence_length, target_start_index, batch_size, mean, std
):
    return [
        create_test_dataset(
            df, feature_columns, target_column, sequence_length, target_start_index, batch_size, mean, std
        )
        for df in dfs
    ]

def load_stations_from_path(path):
    files = os.listdir(path)

    return [pd.read_csv(os.path.join(path, f), index_col=False) for f in files]
