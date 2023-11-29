import models.trainer_lib as tl
import models.torch_model_definitions as tmd

# seq_len, pred_len and model_params pred_len are need to be set after loading params
PARAMS = {
    'reg_s2s': {
        'epochs': 2000,
        'lr': 0.001,
        'batch_size': 2048,
        'es_p': 20,
        'wrapper': tl.S2STSWRAPPER,
        'model': tmd.Seq2seq,
        'model_params': {
            'features': 11,
            'pred_len': None,
            'embedding_size': 10,
            'num_layers': 1,
            'bidirectional': True,
            'dropout': 0.5,
            'noise': 0.05
        },
        'seq_len': None,
        'pred_len': None,
        'teacher_forcing_decay': 0.01,
    },
    'att_s2s': {
        'epochs': 2000,
        'lr': 0.001,
        'batch_size': 2048,
        'es_p': 20,
        'wrapper': tl.S2STSWRAPPER,
        'model': tmd.AttentionSeq2seq,
        'model_params': {
            'features': 11,
            'pred_len': None,
            'embedding_size': 10,
            'bidirectional': True,
            'dropout': 0.0,
            'noise': 0.00
        },
        'seq_len': None,
        'pred_len': None,
        'teacher_forcing_decay': 0.01,
    },
    'pos_att_s2s': {
        'epochs': 2000,
        'lr': 0.001,
        'batch_size': 2048,
        'es_p': 20,
        'wrapper': tl.S2STSWRAPPER,
        'model': tmd.PosAttSeq2seq,
        'model_params': {
            'features': 11,
            'pred_len': None,
            'embedding_size': 10,
            'bidirectional': True,
            'dropout': 0.0,
            'noise': 0.00
        },
        'seq_len': None,
        'pred_len': None,
        'teacher_forcing_decay': 0.01,
    },
}
