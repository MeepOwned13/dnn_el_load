import numpy as np
import torch
from torch import nn
from models.params import PARAMS
from math import sqrt as math_sqrt
import models.trainer_lib as tl
from timeit import default_timer as timer
import os
import pandas as pd


def calc_metrics(p, t):
    """
    Calculates the metrics for predictions and targets
    :param p: predictions
    :param t: targets
    :return: The metrics (mae, mse, rmse, mape, mpe)
    """
    loss = nn.MSELoss()(torch.tensor(p), torch.tensor(t)).item()
    mae = round(np.mean(np.abs(p - t)), 4)
    mse = round(loss, 4)
    rmse = round(math_sqrt(loss), 4)
    mape = round(tl.mape(p, t) * 100, 4)
    mpe = round(tl.mpe(p, t) * 100, 4)

    return mae, mse, rmse, mape, mpe


def load_data(until):
    """
    Loads the data from the csv file and returns it as a numpy array
    :param until: The last date to include in the dataset
    :return: X and y as numpy arrays
    """
    dataset = tl.load_country_wide_dataset('data/country_data.csv', until=until)
    x = dataset.to_numpy(dtype=np.float32)
    y = dataset['el_load'].to_numpy(dtype=np.float32)

    return x, y


def eval_model(params_key: str, seq_len: int, pred_len: int, repeat: int,
               write_file: str = None, model_file: str = None) -> pd.DataFrame:
    """
    Evaluates the model with the given params_key
    :param params_key: the params_key to use from params.py
    :param seq_len: the sequence length to use
    :param pred_len: the prediction length to use
    :param repeat: the number of times to repeat the evaluation
    :param write_file: what file to write the results to, append if exists
    :param model_file: what file to save the best model to, can't be used unless write_file is also used
    :return: evaluation results as Pandas DataFrame
    """
    if model_file and not write_file:
        raise ValueError('Cannot save model without writing results')

    if params_key not in PARAMS:
        raise ValueError(f'Invalid params_key: {params_key}')
    if write_file and not os.path.exists(os.path.dirname(write_file)):
        raise ValueError(f'Invalid write_file directory: {os.path.dirname(write_file)}')
    if model_file and not os.path.exists(os.path.dirname(model_file)):
        raise ValueError(f'Invalid model_file directory: {os.path.dirname(model_file)}')

    df = pd.DataFrame(columns=['Prediction length', 'Sequence length', 'Model',
                               'MAE', 'MSE', 'RMSE', 'MAPE', 'MPE', 'Train Time', 'Pred Time'])
    full_df = df
    if write_file:
        if not os.path.exists(write_file):
            df.to_csv(write_file, index=False)
        else:
            full_df = pd.read_csv(write_file)

    params = PARAMS[params_key]
    n_splits = 2
    epochs = params['epochs']
    lr = params['lr']
    batch_size = params['batch_size']
    es_p = params['es_p']

    params['seq_len'] = seq_len
    params['pred_len'] = pred_len
    params['model_params']['pred_len'] = pred_len

    wrapper = tl.S2STSWrapper(
        model=params['model'](**params['model_params']),
        seq_len=params['seq_len'], pred_len=params['pred_len'],
        teacher_forcing_decay=params['teacher_forcing_decay'],
    )

    x, y = load_data(until='2019-12-31 23:00:00')

    split_len = len(x) // (n_splits+1)

    train_val_sp: int = split_len * n_splits - split_len // 8
    val_test_sp: int = split_len * n_splits

    x_train, x_val, x_test = x[:train_val_sp], x[train_val_sp:val_test_sp], x[val_test_sp:]
    y_train, y_val, y_test = y[:train_val_sp], y[train_val_sp:val_test_sp], y[val_test_sp:]

    res_df = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'MPE', 'Train Time', 'Pred Time'])

    for i in range(args.repeat):
        print(f'Iteration {i + 1}/{args.repeat}')
        st_time = timer()

        # train
        wrapper.init_strategy()
        _ = wrapper.train_strategy(x_train, y_train, x_val, y_val, x_test, y_test,
                                   epochs=epochs, lr=lr, batch_size=batch_size, loss_fn=nn.MSELoss(),
                                   es_p=es_p, es_d=0, verbose=1, cp=True)

        # timing
        train_finish = timer()
        train_time = train_finish - st_time

        # prediction and timing
        y_pred, y_true = wrapper.predict(x_test, y_test)

        pred_finish = timer()
        pred_time = pred_finish - train_finish

        # calculate metrics
        mae, mse, rmse, mape, mpe = calc_metrics(y_pred, y_true)

        # set up dataframe
        df = pd.DataFrame({'Prediction length': pred_len, 'Sequence length': seq_len, 'Model': params_key,
                           'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MPE': mpe,
                           'Train Time': train_time, 'Pred Time': pred_time}, index=[0])

        # save model?
        if model_file:
            best_score = full_df.loc[(full_df['Prediction length'] == pred_len) &
                                     (full_df['Sequence length'] == seq_len) &
                                     (full_df['Model'] == params_key), 'RMSE'].min()
            best_score = best_score if not np.isnan(best_score) else np.inf

            if rmse < best_score:
                wrapper.save_state(model_file)

        # write results
        if write_file:
            df.to_csv(write_file, mode='a', header=False, index=False)

        # concat results to variables
        full_df = df if full_df.empty else pd.concat([full_df, df], axis='rows', ignore_index=True)
        if res_df.empty:
            res_df = df.drop(['Prediction length', 'Sequence length', 'Model'], axis=1)
        else:
            res_df = pd.concat([res_df, df.drop(['Prediction length', 'Sequence length', 'Model'], axis=1)],
                               axis='rows', ignore_index=True)

        # print info
        print(f"RMSE loss: {rmse:.3f} - "
              f"Train time: {train_time / 60:.2f} min - "
              f"Pred time: {pred_time:.3f} sec")

    return res_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate seq2seq approaches')
    parser.add_argument('-p', '--params', choices=list(PARAMS.keys()),
                        required=True, help='Model params to evaluate')
    parser.add_argument('-sl', '--seq_len', type=int, required=True,
                        help='Sequence length to use')
    parser.add_argument('-pl', '--pred_len', type=int, required=True,
                        help='Prediction length to use')
    parser.add_argument('-r', '--repeat', type=int, default=1)
    parser.add_argument('-sw', '--skip_write', action='store_true',
                        help="Don't write results to file")
    parser.add_argument('-sm', '--save_model', action='store_true',
                        default=False, help="Save best model for each fold")
    args = parser.parse_args()

    if args.skip_write and args.save_model:
        raise ValueError('Cannot save model without writing results')

    write_file = None
    if not args.skip_write:
        if not os.path.exists('final_eval_results'):
            os.makedirs('final_eval_results')
        write_file = f'final_eval_results/results.csv'

    model_file = None
    if args.save_model:
        if not os.path.exists('final_eval_results'):
            os.makedirs('final_eval_results')
        model_file = f'final_eval_results/{args.seq_len}_{args.pred_len}_{args.params}.pt'

    print(eval_model(args.params, args.seq_len, args.pred_len, args.repeat, write_file, model_file))
