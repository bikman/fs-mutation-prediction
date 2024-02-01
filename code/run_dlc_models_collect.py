"""
Passes over all runs for models 1, 2, 3, .., 7 and saves model PT files into 'models' folder
"""
import os
import shutil
from utils import get_result_folder_from_log
from utils import MODELS_FOLDER, LOG_FOLDER, RESULTS_PATH
from result_lists import *
import sys
from sys import platform


def create_destination_folder(model_num, protein_num):
    """
    Create folder for pre-trained model
    @param model_num: model num 1..8
    @param protein_num: protein num 1..39
    @return: folder full path
    """
    dest_folder_path = os.path.join(MODELS_FOLDER, f'model_{model_num}_pretrain', str(protein_num))

    if os.path.exists(dest_folder_path) and os.path.isdir(dest_folder_path):
        shutil.rmtree(dest_folder_path)

    os.mkdir(dest_folder_path)
    assert os.path.exists(dest_folder_path)
    assert os.path.isdir(dest_folder_path)
    return dest_folder_path


def get_log_path(run_id):
    """
    Create full path for run log file
    @param run_id: example '12345'
    @return: full path to a LOG file
    """
    log_path = os.path.join(LOG_FOLDER, f'mbikman_{run_id}.log')
    return log_path


def copy_model_from_result_folder(result_folder, dest_folder):
    """
    Copy model PT file from run result folder
    @param result_folder: run result folder
    @param dest_folder: destination folder
    """
    model_file = [f for f in os.listdir(result_folder) if f.endswith('.pt')][0]
    model_src_path = os.path.join(result_folder, model_file)
    model_dst_path = os.path.join(dest_folder, model_file)
    shutil.copyfile(model_src_path, model_dst_path)
    assert os.path.exists(model_dst_path)
    print(f'copied MODEL: {run_id}, model {model_num}, prot. {protein_num}')
    res_file = [f for f in os.listdir(result_folder) if f.endswith('eval_result.pkl')][0]
    res_src_path = os.path.join(result_folder, res_file)
    res_dst_path = os.path.join(dest_folder, res_file)
    shutil.copyfile(res_src_path, res_dst_path)
    assert os.path.exists(res_dst_path)
    print(f'copied EVAL_RES: {run_id}, model {model_num}, prot. {protein_num}')


def copy_eval_res_from_result_folder(result_folder, dest_folder):
    """

    @param result_folder:
    @param dest_folder:
    @return:
    """
    res_file = [f for f in os.listdir(result_folder) if f.endswith('eval_result.pkl')][0]
    res_src_path = os.path.join(result_folder, res_file)
    res_dst_path = os.path.join(dest_folder, res_file)
    shutil.copyfile(res_src_path, res_dst_path)
    assert os.path.exists(res_dst_path)


def copy_model_by_run(run_id, model_num, protein_num):
    """
    Copy PT file of the model into dest. folder, according to a protein number
    @param run_id: example '12345'
    @param model_num: model number = 1 for full, 2,3,...,7 for partial ablation models
    @param protein_num: from 1 to 39
    """
    dest_folder = create_destination_folder(model_num, protein_num)
    if platform == "linux" or platform == "linux2":
        log_path = get_log_path(run_id)
        run_folder = get_result_folder_from_log(log_path)
        result_folder = os.path.join(RESULTS_PATH, run_folder)
    else:
        result_folder = os.path.join(RESULTS_PATH, run_id)
    copy_model_from_result_folder(result_folder, dest_folder)





if __name__ == '__main__':
    print('@' * 100)
    print('ver:26.1.24')
    print('@' * 100)
    runs_per_model = {
        1: MODEL1_RUNS,
        2: MODEL2_RUNS,
        3: MODEL3_RUNS,
        4: MODEL4_RUNS,
        5: MODEL5_RUNS,
        6: MODEL6_RUNS,
        7: MODEL7_RUNS,
    }
    # for model_num in range(1, 8):
    for model_num in range(1, 2):  # only 1
        protein_num = 1
        for run_id in runs_per_model[model_num]:
            copy_model_by_run(run_id, model_num, protein_num)
            protein_num += 1
    print('-- OK! --')
