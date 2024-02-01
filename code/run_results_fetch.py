"""
Module used to fetch data from result folders
This data is put to CSV to be moved to Excel and plots
"""
import os
import re

from utils import RESULTS_PATH


def fetch_ft_results_from_folder(all_res_folder):
    res_ids = [res_id for res_id in os.listdir(all_res_folder)]
    list_of_tuples = []
    for res_id in res_ids:
        res_dir = os.path.join(all_res_folder, res_id)
        res_path = os.path.join(res_dir, 'result.txt')
        with open(res_path) as file:
            # print(res_path)
            while line := file.readline():
                if "nn_mae:" in line.strip():
                    nn_mae = line.split(':')[1].strip()
                if "spearman:" in line.strip():
                    sp = line.split(':')[1].strip()
        list_of_tuples.append((nn_mae, sp))
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
                if line.startswith("ft") and "nn_mae" in line:
                    mut_count = re.search(r'^ft[\d]+', line).group(0)[2:]
                    value = line.split(':')[1]
                    if mut_count in ft_dict:
                        ft_dict[mut_count].append(value)
                    else:
                        ft_dict[mut_count] = [value]
                    continue
        csv_row = [res_id]
        # all
        for mut_count in ft_dict:
            csv_row += ft_dict[mut_count]
        list_of_tuples.append(csv_row)
    for t in list_of_tuples:
        print(','.join(t))  # all
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
