from eval_model import eval_model
from models.params import PARAMS
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all seq2seq approaches')
    parser.add_argument('-r', '--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("Repeat must be at least 1")

    if not os.path.exists('final_eval_results'):
        os.makedirs('final_eval_results')

    write_file = f'final_eval_results/results.csv'

    repeat = args.repeat
    seq_lens = [24, 48]  # 1, 2 days
    pred_lens = [6, 12, 24, 48]  # 6, 12, 24, 48 hours
    param_keys = list(PARAMS.keys())
    # r * 2 * 4 * 4 = r * 32 runs

    for _ in range(repeat):
        for sl in seq_lens:
            for pl in pred_lens:
                for key in param_keys:
                    print(f"Seq_len: {sl}, Pred_len: {pl}, Config: {key}")
                    custom_embedding_size = PARAMS[key]['model_params']['embedding_size']
                    if sl != 24:
                        custom_embedding_size += 2
                    eval_model(key, sl, pl, 1, write_file, f"final_eval_results/{sl}_{pl}_{key}.pt",
                               custom_embedding_size)
