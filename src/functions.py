import io
import os
import re

import keras
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.src.layers import Dropout
from sklearn.metrics import confusion_matrix

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


def create_model(architecture, input_shape, dropout_rate, logging=None, summary=True, dropout_layer='monte_carlo'):
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
                MonteCarloDropout(dropout_rate) if dropout_layer == 'monte_carlo' else Dropout(dropout_rate),
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
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    return fig


def plot_data(dfs, y='HS', target='no_snow', predictions=[], show=True):
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

    if show:
        plt.show()

    return fig


def predict_with_uncertainty(model, x, n_iter=5):
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
        batch_size
):
    total_sequences = sum(len(df) - sequence_length for df in dfs)

    seq_array = np.zeros((total_sequences, sequence_length, len(feature_columns)))
    target_array = np.zeros(total_sequences)

    insert_idx = 0

    for df in dfs:
        df = preprocces(df)
        features = df[feature_columns].values
        targets = df[target_column].values if target_column else np.zeros(len(df))

        max_index = len(df) - sequence_length

        for i in range(max_index):
            seq_array[insert_idx] = features[i:i + sequence_length]
            target_array[insert_idx] = targets[i + sequence_length - 1]
            insert_idx += 1

    indices = np.arange(seq_array.shape[0])
    np.random.shuffle(indices)
    shuffled_seq_array = seq_array[indices]
    shuffled_target_array = target_array[indices]

    split_index = int(split_percentage * len(shuffled_seq_array))

    train_seq = shuffled_seq_array[:split_index]
    val_seq = shuffled_seq_array[split_index:]
    train_target = shuffled_target_array[:split_index]
    val_target = shuffled_target_array[split_index:]

    mean = train_seq.mean(axis=(0, 1))
    std = train_seq.std(axis=(0, 1))

    train_seq = (train_seq - mean) / std
    val_seq = (val_seq - mean) / std

    train_dataset = tf.data.Dataset.from_tensor_slices((train_seq, train_target)).cache().shuffle(
        buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_seq, val_target)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, mean, std, len(train_seq), len(val_seq), None


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


def replicate_seasonal_pattern(df, column_name):
    yearly_data = df.resample('Y').count()
    max_count = yearly_data.max().iloc[0]
    first_full_year = yearly_data[yearly_data[column_name] == max_count].index[0].year

    if max_count < 365:
        return df

    period_arrays = []

    for year in range(first_full_year, first_full_year + 3):
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        if start_date in df.index and end_date in df.index:
            typical_period = df.loc[start_date:end_date, column_name]
            period_arrays.append(typical_period.values)

    if len(period_arrays) < 3:
        return df

    min_length = min(len(arr) for arr in period_arrays)
    period_arrays = [arr[:min_length] for arr in period_arrays]
    combined_mean = np.mean(period_arrays, axis=0)

    repeat_times = len(df) // len(combined_mean) + 1
    fill_pattern = np.tile(combined_mean, repeat_times)[:len(df)]

    df[column_name].fillna(pd.Series(fill_pattern, index=df.index), inplace=True)

    return df


def mlflow_log_np_as_file(data, name):
    buffer = io.StringIO()
    np.savetxt(buffer, data, delimiter=',')
    mlflow.log_text(buffer.getvalue(), artifact_file=f'{name}.txt')
    buffer.close()


def generate_confusion_matrix(testing_stations, predictions, target_tart_index):
    y_true = [ts.loc[target_tart_index:, 'no_snow'].values for ts in testing_stations.values()]
    y_pred = predictions
    n_matrices = len(y_true)
    n_cols = 3
    n_rows = (n_matrices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5))
    fig.tight_layout(pad=5.0)

    for i in range(n_matrices):
        ax = axes[i // n_cols, i % n_cols]
        cm = confusion_matrix(y_true[i], y_pred[i])
        sns.heatmap(cm, annot=True, fmt='d', cmap="crest", ax=ax)
        ax.set_title(list(testing_stations.keys())[i])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    plt.show()

    return fig
