"""
Module used to fetch data from result folders
This data is put to CSV to be moved to Excel and plots
"""
import os
from results import BINS_RESULTS, NO_BINS_RESULTS
from utils import RESULTS_PATH
from result_lists import *
import re


def fetch_ft_results_from_folder(all_res_folder):
    res_ids = [res_id for res_id in os.listdir(all_res_folder)]
    list_of_tuples = []
    for res_id in res_ids:
        res_dir = os.path.join(all_res_folder, res_id)
        res_path = os.path.join(res_dir, 'result.txt')
        with open(res_path) as file:
            # print(res_path)
            while line := file.readline():
                if " mae:" in line.strip():
                    mae = line.split(':')[1].strip()
                if "mape:" in line.strip():
                    mape = line.split(':')[1].strip()
                if "nn_mae:" in line.strip():
                    nn_mae = line.split(':')[1].strip()
                if "spearman:" in line.strip():
                    sp = line.split(':')[1].strip()
        list_of_tuples.append((res_id, mae, mape, nn_mae, sp))
    for t in list_of_tuples:
        print(','.join(t))
        pass
    print('ok')


def fetch_results_from_folder(all_res_folder):
    """
    Allows to fetch result from folder where all 39 result folders are located
    Data is collected from 'result.txt' file
    @return: prints collected data
    """
    res_ids = [res_id for res_id in os.listdir(all_res_folder)]
    list_of_tuples = []
    for res_id in res_ids:
        res_dir = os.path.join(all_res_folder, res_id)
        res_path = os.path.join(res_dir, 'result.txt')
        with open(res_path) as file:
            print(res_path)
            ft_dict = {}
            while line := file.readline().strip():
                # if line.startswith("mae:"):
                #     mae = line.split(':')[1].strip()
                #     continue
                # if line.startswith("nn_mae:"):
                #     nn_mae = line.split(':')[1].strip().strip(']').strip('[').split(',')
                #     continue
                # if line.startswith("spearman:"):
                #     sp = line.split(':')[1].strip()
                #     continue
                # if line.startswith("ft") and " mae" in line:
                #     mut_count = re.search(r'^ft[\d]+', line).group(0)[2:]
                #     value = line.split(':')[1]
                #     if mut_count in ft_dict:
                #         ft_dict[mut_count].append(value)
                #     else:
                #         ft_dict[mut_count] = [value]
                #     continue
                if line.startswith("ft") and "nn_mae" in line:
                    mut_count = re.search(r'^ft[\d]+', line).group(0)[2:]
                    value = line.split(':')[1]
                    if mut_count in ft_dict:
                        ft_dict[mut_count].append(value)
                    else:
                        ft_dict[mut_count] = [value]
                    continue
                # if line.startswith("ft") and "sp." in line:
                #     mut_count = re.search(r'^ft[\d]+', line).group(0)[2:]
                #     value = line.split(':')[1]
                #     if mut_count in ft_dict:
                #         ft_dict[mut_count].append(value)
                #     else:
                #         ft_dict[mut_count] = [value]
                #     continue

        # csv_row = [res_id, mae] + nn_mae + [sp]
        csv_row = [res_id]
        # all
        for mut_count in ft_dict:
            csv_row += ft_dict[mut_count]
        list_of_tuples.append(csv_row)
    for t in list_of_tuples:
        print(','.join(t))  # all
        pass
    print('ok')


def fetch_result_from_list_of_folders(runs_list):
    '''
    TBD
    @param runs_list:
    @return:
    '''
    list_of_tuples = []
    for res_id in runs_list:
        run_folder_path = os.path.join(RESULTS_PATH, res_id)
        log_file = [f for f in os.listdir(run_folder_path) if f.endswith('.log')][0]
        log_file_path = os.path.join(run_folder_path, log_file)
        with open(log_file_path) as file:
            print(res_id, log_file_path)
            is_in_mut_count = False
            is_in_result = False
            while line := file.readline():
                # if line.startswith("mut_count='16'"):
                if line.startswith("mut_count='4'"):
                    is_in_mut_count = True
                    continue
                if line.startswith("is_destructive='0'") and is_in_mut_count:
                    is_in_result = True
                    continue
                if is_in_result and line.startswith("MAE"):
                    mae = line.split(':')[1].strip()
                    continue
                if is_in_result and line.startswith("Spearman"):
                    sp = line.split(':')[1].strip()
                    continue
                if is_in_result and line.startswith("R2"):
                    is_in_mut_count = False
                    is_in_result = False
        list_of_tuples.append((res_id, mae, sp))
    for t in list_of_tuples:
        print(f'{t[0]},{t[1]},{t[2]}')  # res_id, mae, sp
        pass
    print('ok')


def cleanup_folder(all_res_folder):
    """
    Remove all *.pt files (big model saves) from given folder
    @param all_res_folder: folder to clean-up
    """
    res_ids = [res_id for res_id in os.listdir(all_res_folder)]
    for res_id in res_ids:
        res_dir = os.path.join(all_res_folder, res_id)
        print(f'Cleaning {res_dir}')
        victims_list = [f for f in os.listdir(res_dir) if f.endswith('.pt')]
        for v in victims_list:
            victim_path = os.path.join(res_dir, v)
            print(f'Removing file {victim_path}')
            os.remove(victim_path)
            assert not os.path.isfile(victim_path)
    print('OK!')


if __name__ == '__main__':
    # fetch_ft_results_from_folder(r'D:\Temp')
    cleanup_folder(r'D:\Temp')
    fetch_results_from_folder(r'D:\Temp')
    pass
