from eval_model import eval_model
from models.params import PARAMS
import os

if __name__ == '__main__':
    if not os.path.exists('final_eval_results'):
        os.makedirs('final_eval_results')

    write_file = f'final_eval_results/results.csv'

    seq_lens = [24, 48]  # 1, 2 days
    pred_lens = [6, 12, 24, 48]  # 6, 12, 24, 48 hours
    param_keys = list(PARAMS.keys())
    # 2 * 4 * 3 = 24 runs

    for sl in seq_lens:
        for pl in pred_lens:
            for key in param_keys:
                print(f"Seq_len: {sl}, Pred_len: {pl}, Config: {key}")
                custom_embedding_size = 10 if sl == 24 else 12
                eval_model(key, sl, pl, 1, write_file, f"final_eval_results/{sl}_{pl}_{key}.pt",
                           custom_embedding_size)
