import logging
import os
import pickle

from torch.utils.data import DataLoader

from train import eval_prism_scores_multi_sets
from utils import TestParameters

LOG_ENABLED = True
log = print
if LOG_ENABLED:
    # noinspection PyRedeclaration
    log = logging.info


def run_eval_on_model(batch_size, ds_list, model):
    test_params = TestParameters()
    test_params.model = model
    test_params.loaders_list = [DataLoader(x, batch_size=batch_size, shuffle=True) for x in ds_list]
    # test_params.loaders_list = [DataLoader(x, batch_size=batch_size, shuffle=False) for x in ds_list]
    log(f'Sum of loaders={sum([len(x) for x in test_params.loaders_list])}')
    test_res = eval_prism_scores_multi_sets(test_params, log)
    test_res.model_name = model.file_name
    log(test_res)
    return test_res


def pickle_test_result(report_path, run_type, test_res):
    """
    Writes test result to pickled file
    @param report_path: folder to save result object there
    @param run_type: result run type
    @param test_res: result object
    """
    file_name = f'{run_type}_result.pkl'
    dump_path = os.path.join(report_path, file_name)
    with open(dump_path, "wb") as f:
        pickle.dump(test_res, f)
    log(f'Saved test result: {dump_path}')


if __name__ == '__main__':
    pass
