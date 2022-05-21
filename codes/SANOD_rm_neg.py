from codes.SANOD import SANODCalc

import os
import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
import numpy as np
import matplotlib.pyplot as plt
from marccd import MarCCD
from codes.mccd2dat import load_n_azi_intg
from codes.SVDCalc2022 import SVDCalc


folder_name = "I2_Benzene"
# run_num_list = [85]
run_num_list = list(range(45, 64)) + list(range(73, 79)) + list(range(80, 86))
bad_run_num_list = [49, 55, 56, 58, 59, 61, 62, 63, 81, 82, 83]
run_num_list = [run_num for run_num in run_num_list if run_num not in bad_run_num_list]
#run_num_list = list(range(64, 73))
show_hist = True
show_static = False

neg_used_delay_idx = 0
svd_delay_idx = 1

diff_intg_cut = 0.01

norm_start_q = 4
norm_end_q = 8

raw_waxs_intg_cut = 1.5e6

print("source run list : ", run_num_list)

def calc_q_cut_idx(data, q_start, q_end):
    start_idx = int(np.where(data > q_start)[0][0])
    end_idx = int(np.where(data > q_end)[0][0])
    return start_idx, end_idx

def make_remainder_to_delay_idx_dict(delay_list):
    now_delay_len = len(delay_list)
    divider = now_delay_len * 2
    convert_dict = {}
    for remainder_val in range(divider):
        delay_idx = remainder_val
        if remainder_val >=  now_delay_len:
            delay_idx = (divider - remainder_val) - 1
        convert_dict[remainder_val] = delay_idx
    print(convert_dict)
    return divider, convert_dict


preprocess_file_dir = "../results/shot_azi_intg/"
rawdata_file_dir = "../data/"
common_delay_list = []
common_q = []

norm_diff_int_list = []
each_delay_shot_idx_list = []
on_waxs_intg_list_before_cut = []
off_waxs_intg_list_before_cut = []
diff_waxs_intg_list_before_cut = []



on_norm_int_list = []
off_norm_int_list = []
on_waxs_intg_list = []
off_waxs_intg_list = []
diff_int_list = []
diff_waxs_intg_list = []

if show_static == True:
    neg_delay_int = []


for idx_delay in range(len(run_num_list)):
    norm_diff_int_list.append([])
    each_delay_shot_idx_list.append([])

    on_waxs_intg_list_before_cut.append([])
    off_waxs_intg_list_before_cut.append([])
    diff_waxs_intg_list_before_cut.append([])

    on_norm_int_list.append([])
    off_norm_int_list.append([])
    on_waxs_intg_list.append([])
    off_waxs_intg_list.append([])
    diff_int_list.append([])
    diff_waxs_intg_list.append([])

intg_start_idx = None
intg_end_idx = None

num_col_in_one_fig = 3
num_row_in_one_fig = 2
num_suplot_in_one_fig = num_col_in_one_fig * num_row_in_one_fig
idx_col = 0
idx_row = 0
# outlier_fig, outlier_axis = plt.subplots(nrows=num_row_in_one_fig, ncols=num_col_in_one_fig, figsize=(16, 10))
outlier_idx = 0
in_fig_idx = outlier_idx
# now_axs = outlier_axis[idx_row][idx_col]


for idx_run, run_num in enumerate(run_num_list):
    now_run_name = f"run{run_num:02d}"
    now_run_azi_int_dir = f"{preprocess_file_dir}{now_run_name}/"
    delay_info_file = f"{rawdata_file_dir}{folder_name}/{now_run_name}/{now_run_name}_delayinfo.log"
    if len(common_delay_list) == 0:
        common_delay_list = np.loadtxt(delay_info_file)
        divider_for_delay, remainder2idx = make_remainder_to_delay_idx_dict(common_delay_list)
    dir_file_list = os.listdir(now_run_azi_int_dir)
    dir_file_list.sort()

    idx_shot = 0
    for each_file_name in dir_file_list:
        if each_file_name.endswith("_on.dat"):
            on_file_path = f"{now_run_azi_int_dir}{each_file_name}"
            on_rawdata = np.loadtxt(on_file_path)
            shot_num = int(each_file_name[-10:-7])
            if len(common_q) == 0:
                common_q = on_rawdata[:, 0]
                intg_start_idx, intg_end_idx = calc_q_cut_idx(common_q, norm_start_q, norm_end_q)
            now_on_int = on_rawdata[:, 1]
            norm_on_val = np.sum(now_on_int[intg_start_idx:intg_end_idx])
            now_norm_on_int = now_on_int / norm_on_val
            now_time_delay_idx = remainder2idx[(shot_num % divider_for_delay)]
            if now_time_delay_idx == neg_used_delay_idx:
                off_waxs_intg_list_before_cut[idx_run].append(norm_on_val)
                if norm_on_val < raw_waxs_intg_cut:
                    continue
                off_norm_int_list[idx_run].append(now_norm_on_int)
                off_waxs_intg_list[idx_run].append(now_norm_on_int)
            elif now_time_delay_idx == svd_delay_idx:
                on_waxs_intg_list_before_cut[idx_run].append(norm_on_val)
                if norm_on_val < raw_waxs_intg_cut:
                    continue
                on_norm_int_list[idx_run].append(now_norm_on_int)
                on_waxs_intg_list[idx_run].append(now_norm_on_int)


each_run_avg_diff = []
for idx_run, each_run_on_int_list in enumerate(on_norm_int_list):
    for idx_data, on_norm_int in enumerate(each_run_on_int_list):
        try:
            now_off_pair_int = off_norm_int_list[idx_run][idx_data]
        except IndexError as e:
            print(e)
            now_off_pair_int = off_norm_int_list[idx_run][-1]
            print("use last off int")
        now_norm_diff = on_norm_int - now_off_pair_int
        norm_diff_intg = np.sum(np.abs(now_norm_diff))
        diff_waxs_intg_list_before_cut[idx_run].append(norm_diff_intg)
        if norm_diff_intg > diff_intg_cut:
            continue
        diff_waxs_intg_list[idx_run].append(norm_diff_intg)
        norm_diff_int_list[idx_run].append(now_norm_diff)
    now_run_avg_diff = np.average(norm_diff_int_list[idx_run], axis=0)
    each_run_avg_diff.append(now_run_avg_diff)
    plt.plot(common_q, now_run_avg_diff, label=f"run{run_num_list[idx_run]}")
    if (idx_run % 5) == 4:
        plt.legend()
        plt.show()
if (idx_run % 5) != 4:
    plt.legend()
    plt.show()
each_run_avg_diff = np.array(each_run_avg_diff)
whole_diff = np.concatenate(norm_diff_int_list)
whole_diff_avg = np.average(whole_diff, axis=0)
whole_on = np.concatenate(on_norm_int_list)
whole_off = np.concatenate(off_norm_int_list)

delay_info_text = f"{svd_delay_idx}-th delay {common_delay_list[svd_delay_idx]}us"
plt.title(f"{delay_info_text} whole run avg")
plt.plot(common_q, whole_diff_avg)
plt.show()

# diffSVD = SVDCalc(np.transpose(each_run_avg_diff))
# diffSVD.calc_svd()
# diffSVD.pick_meaningful_data(3)
# diffSVD.plot_singular_val()
# diffSVD.plot_left_vec(plot_x_val=common_q, title_data_name=delay_info_text)
# diffSVD.plot_right_vec(plot_x_val=run_num_list, x_val_as_text=True, title_data_name=delay_info_text)
#
# eachSVD = SVDCalc(np.transpose(whole_diff))
# eachSVD.calc_svd()
# eachSVD.pick_meaningful_data(3)
# eachSVD.plot_singular_val()
# eachSVD.plot_left_vec(plot_x_val=common_q, title_data_name=delay_info_text)
pos_shot_len = len(whole_on)
nowSANOD = SANODCalc()
nowSANOD.set_file_save(False)
nowSANOD.set_file_family_name(f"{svd_delay_idx}-th_delay")
nowSANOD.set_data_for_sanod(pos_data=np.array(whole_on), neg_data=np.array(whole_off), q_val=common_q)
nowSANOD.advanced_svd_neg_and_all_signal(sing_val_show_num=10, num_neg_comp=3, num_whole_comp=3)
nowSANOD.plot_non_ortho_comps()
sanod_after_data = nowSANOD.artifact_filtering()
avg_after_sanod = np.average(sanod_after_data[-pos_shot_len:], axis=0)
plt.plot(common_q, avg_after_sanod)
plt.title("sanod after data average " + delay_info_text)
plt.show()