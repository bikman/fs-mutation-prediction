"""
Package used for collecting run results
0. Clear 'temp' output folder
1. Update run ids into list
2. Run container in DLC
3. Run this PY file with python
4. Copy collected folders to local machine manually via SSH
"""
import os.path
import shutil
from utils import get_result_folder_from_log, LOG_FOLDER


RESULTS_FOLDER = r'/root/code/reports'
OUTPUT_FOLDER = r'/root/temp'
# ============= Update list of folders here ===================
LIST_OF_RESULTS_IDS = [
    # TODO: fill list of ids here
]


# =============================================================

def get_result_folders(list_of_ids):
    """
    Result list of tuples with id for run folder
    @param list_of_ids: list of run ids
    @return: list of tuples (id, result folder name)
    """
    res = []
    for id in list_of_ids:
        print(f'Current id: {id}')
        log_file = f'mbikman_{id}.log'
        log_path = os.path.join(LOG_FOLDER, log_file)
        print(f'{log_path=}')
        res_folder = get_result_folder_from_log(log_path)
        print(f'{res_folder=}')
        res.append((str(id), res_folder))
    return res


def copy_results(id_res_folders):
    i = 0
    for t in id_res_folders:
        id = t[0]
        res_folder = t[1]
        run_folder_path = os.path.join(RESULTS_FOLDER, res_folder)
        res_folder_path = os.path.join(OUTPUT_FOLDER, id)
        shutil.copytree(run_folder_path, res_folder_path)
        print(f'Copied "{run_folder_path}" ==> "{res_folder_path}"')
        i = i + 1
    print('=' * 20)
    print(f'Totally copied: {i} folders')


def validate_folder_empty(path):
    d = os.listdir(path)
    if len(d) > 0:
        raise Exception(f'Cannot run: {path} is not empty!')


if __name__ == '__main__':
    try:
        validate_folder_empty(OUTPUT_FOLDER)
    except Exception as e:
        print(f'FAIL: {e}')
        exit(1)
    id_res_folders = get_result_folders(LIST_OF_RESULTS_IDS)
    assert len(id_res_folders) == len(LIST_OF_RESULTS_IDS)
    print(id_res_folders)
    copy_results(id_res_folders)
    print('OK!')
