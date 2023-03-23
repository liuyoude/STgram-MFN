# directly copy from https://github.com/y-kawagu/dcase2020_task2_evaluator
########################################################################
# import default python-library
########################################################################
import os
import sys
import csv
import glob
import re
from operator import itemgetter
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
from sklearn import metrics
########################################################################


########################################################################
# constant values
########################################################################
# Number of columns in lines indicating machine types in the ground truth CSV + 1
CHK_MACHINE_TYPE_LINE = 2

# Column index in the ground truth CSV
FILENAME_COL = 0
MACHINE_TYPE_COL = 0
Y_TRUE_COL = 2

# Column index in anomaly score CSVs
EXTRACTION_ID_COL = 0
SCORE_COL = 1
########################################################################


########################################################################
# parameters
########################################################################
# FPR threshold for pAUC
MAX_FPR = 0.1

# Path of the ground truth
EVAL_DATA_LIST_PATH = "./eval_data_list.csv"

# Directory in which each team's subdirectory containing the anomaly scores
TEAMS_ROOT_DIR = "./teams/"

# Output directory
RESULT_DIR = "./teams/"
########################################################################


########################################################################
# data save in CSV file
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)
########################################################################


########################################################################
# result data output
########################################################################
def output_result(team_dir, machine_types):

    dir_name = os.path.basename(team_dir)

    result_name = "result_" + dir_name + ".csv"

    result_file = "{result_dir}/{result_name}".format(result_dir=RESULT_DIR, result_name=result_name)

    csv_lines = []

    averaged_result = []
    nums = 0
    for machine_type in machine_types:

        anomaly_score_path_list = sorted(
            glob.glob("{dir}/anomaly_score_{machine_type}_id*".format(dir=team_dir, machine_type=machine_type)))

        csv_lines.append([machine_type])
        csv_lines.append(["id", "AUC", "pAUC"])

        performance = []
        print("=============================================")
        print("MACHINE TYPE IS [{}]".format(machine_type))
        print("---------------------------------------------")

        for anomaly_score_path in anomaly_score_path_list:

            with open(anomaly_score_path) as fp:
                anomaly_score_list = list(csv.reader(fp))

                anomaly_score_list_sort = sorted(anomaly_score_list, key=itemgetter(0))

            machine_id = re.findall('id_[0-9][0-9]', anomaly_score_path)[EXTRACTION_ID_COL]
            print(machine_id)

            y_true = []

            for eval_data in eval_data_list:
                if len(eval_data) < CHK_MACHINE_TYPE_LINE:
                    flag = True if eval_data[MACHINE_TYPE_COL] == machine_type else False
                else:
                    if flag and machine_id in str(eval_data[FILENAME_COL]):
                        y_true.append(float(eval_data[Y_TRUE_COL]))

            y_pred = [float(anomaly_score[SCORE_COL]) for anomaly_score in anomaly_score_list_sort]

            if len(y_true) != len(y_pred):
                print("Err:anomaly_score may be missing")
                sys.exit(1)

            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=MAX_FPR)
            csv_lines.append([machine_id.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])
            print("AUC :", auc)
            print("pAUC :", p_auc)

        averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
        print("\nAUC Average :", averaged_performance[0])
        print("pAUC Average :", averaged_performance[1])
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])

        if nums == 0:
            averaged_result = averaged_performance
        else:
            averaged_result += averaged_performance
        nums += 1

    averaged_result /= nums
    print("\nAUC Average :", averaged_result[0])
    print("pAUC Average :", averaged_result[1])
    csv_lines.append(["Average"] + list(averaged_result))
    csv_lines.append([])
    print("=============================================")
    print("AUC and pAUC results -> {}".format(result_file))
    save_csv(save_file_path=result_file, save_data=csv_lines)
########################################################################


########################################################################
# main evaluator.py
########################################################################
if __name__ == "__main__":

    teams_dirs = glob.glob("{root_dir}/*".format(root_dir=TEAMS_ROOT_DIR))

    if os.path.exists(EVAL_DATA_LIST_PATH):
        with open(EVAL_DATA_LIST_PATH) as fp:
            eval_data_list = list(csv.reader(fp))
    else:
        print("Err:eval_data_list.csv not found")
        sys.exit(1)

    machine_types = []

    for idx in eval_data_list:
        if len(idx) < CHK_MACHINE_TYPE_LINE:
            machine_types.append(idx[MACHINE_TYPE_COL])

    for team_dir in teams_dirs:
        if os.path.isdir(team_dir):
            print(team_dir)
            output_result(team_dir, machine_types)
        else:
            print("{} is not directory.".format(team_dir))

