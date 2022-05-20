import os
import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
import numpy as np
import matplotlib.pyplot as plt
from marccd import MarCCD
from codes.mccd2dat import load_n_azi_intg

folder_name = "I2_Benzene"
run_num = 14

# shot number is 0 based !
shot_num_start = 0
# shot_num_end = 24
shot_num_end = 24

norm_start_q = 4
norm_end_q = 8
diff_cut_start_q = 8
diff_cut_end_q = 10

shot_num_list = list(range(shot_num_start, shot_num_end + 1))
print("plot range : run ", shot_num_list)

preprocess_file_dir = "../results/shot_azi_intg/"
rawdata_file_dir = "../data/"
now_run_name = f"run{run_num:02d}"
now_run_azi_int_dir = f"{preprocess_file_dir}{now_run_name}/"

# load_n_azi_intg(rawdata_file_dir, preprocess_file_dir, folder_name, run_num, do_overwrite_file=False)

# dir_file_list = os.listdir(now_run_azi_int_dir)
# print(dir_file_list)

delay_info_file = f"{rawdata_file_dir}{folder_name}/{now_run_name}/{now_run_name}_delayinfo.log"
delay_list = np.loadtxt(delay_info_file)
print(delay_info_file)
print("now run delay list : ", delay_list)

# on_file_name_list = []
# neg_file_name_list = []
# for each_file_name in dir_file_list:
#     if each_file_name.endswith("_on.dat"):
#         on_file_name_list.append(each_file_name)
#     elif each_file_name.endswith("_off.dat"):
#         neg_file_name_list.append(each_file_name)
#
# on_file_name_list.sort()
# neg_file_name_list.sort()
# on_shot_len = len(on_file_name_list)

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

divider_for_delay, remainder2idx = make_remainder_to_delay_idx_dict(delay_list)

common_q = []
on_int_each_delay = []
diff_int_list = []
on_norm_int_list = []
off_norm_int_list = []
norm_paired_diff_int_list = []
on_each_delay_shot_idx_list = []
off_each_delay_shot_idx_list = []

on_waxs_intg_list = []
off_waxs_intg_list = []
diff_all_intg_list = []

for idx_delay in range(len(delay_list)):
    diff_int_list.append([])
    norm_paired_diff_int_list.append([])
    on_each_delay_shot_idx_list.append([])
    off_each_delay_shot_idx_list.append([])

    on_norm_int_list.append([])
    off_norm_int_list.append([])
    on_waxs_intg_list.append([])
    off_waxs_intg_list.append([])
    diff_all_intg_list.append([])
intg_start_idx = None
intg_end_idx = None

for shot_num in shot_num_list:
    on_file_path = f"{now_run_azi_int_dir}{now_run_name}_shot{shot_num:03d}_on.dat"
    off_file_path = f"{now_run_azi_int_dir}{now_run_name}_shot{shot_num:03d}_off.dat"
    on_rawdata = np.loadtxt(on_file_path)
    off_rawdata = np.loadtxt(off_file_path)
    if len(common_q) == 0:
        common_q = on_rawdata[:, 0]
        intg_start_idx, intg_end_idx = calc_q_cut_idx(common_q, norm_start_q, norm_end_q)
        cut_intg_start_idx, cut_intg_end_idx = calc_q_cut_idx(common_q, diff_cut_start_q, diff_cut_end_q)
    now_on_int = on_rawdata[:, 1]
    now_off_int = off_rawdata[:, 1]

    norm_off_val = np.sum(now_off_int[intg_start_idx:intg_end_idx])
    norm_on_val = np.sum(now_on_int[intg_start_idx:intg_end_idx])

    norm_on_int = now_on_int / norm_on_val
    norm_off_int = now_off_int / norm_off_val
    now_time_delay_idx = remainder2idx[(shot_num % divider_for_delay)]

    raw_waxs_intg_cut = 4.6e5

    if norm_on_val < raw_waxs_intg_cut:
        print(f"shot{shot_num}_on")
        # plt.plot(common_q, now_on_int, label=f"shot{shot_num}_on")
        # plt.legend()
        # plt.show()
    else:
        on_each_delay_shot_idx_list[now_time_delay_idx].append(shot_num)
        on_norm_int_list[now_time_delay_idx].append(norm_on_int)
        on_waxs_intg_list[now_time_delay_idx].append(norm_on_val)

    if norm_off_val < raw_waxs_intg_cut:
        print(f"shot{shot_num}_off")
        # plt.plot(common_q, now_off_int, label=f"shot{shot_num}_off")
        # plt.legend()
        # plt.show()
    else:
        off_each_delay_shot_idx_list[now_time_delay_idx].append(shot_num)
        off_norm_int_list[now_time_delay_idx].append(norm_off_int)
        off_waxs_intg_list[now_time_delay_idx].append(norm_off_val)


    # diff_int_list[now_time_delay_idx].append(now_diff)
    # norm_diff_int_list[now_time_delay_idx].append(now_norm_diff)
    # diff_waxs_intg_list[now_time_delay_idx].append(norm_diff_intg)

    print_per_shot = 20
    if (shot_num % print_per_shot) == (print_per_shot - 1):
        print(f"load dat done until shot{shot_num + 1}")

# num_plot_in_one_fig = 10
# for idx_delay, delay_val in enumerate(delay_list):
#     # plt.title(f"diff hist of each delay (delay={delay_val})")
#     # plt.hist(diff_waxs_intg_list[idx_delay])
#     # plt.show()
#
#     plt.title(f"on hist (delay={delay_val})")
#     plt.hist(on_waxs_intg_list[idx_delay])
#     plt.show()
#
#     plt.title(f"off hist (delay={delay_val})")
#     plt.hist(off_waxs_intg_list[idx_delay])
#     plt.show()

def find_neareast_pair(compare_arr, item):
    temp_arr = np.asarray(compare_arr)
    pair_idx = (np.abs(temp_arr - item)).argmin()
    return compare_arr[pair_idx], pair_idx


for idx_delay, delay_on_intg_list in enumerate(on_waxs_intg_list):
    title = f"[{now_run_name}] {idx_delay + 1}-th delay : {delay_list[idx_delay]}ns paired diff"
    plt.title(title)
    for idx_data_in_delay, on_intg_val in enumerate(delay_on_intg_list):
        _, nearest_int_pair_idx = find_neareast_pair(off_waxs_intg_list[idx_delay], on_intg_val)
        now_on_int = on_norm_int_list[idx_delay][idx_data_in_delay]
        paired_off_int = off_norm_int_list[idx_delay][nearest_int_pair_idx]
        print(f"now pair: on={on_each_delay_shot_idx_list[idx_delay][idx_data_in_delay]} with off={off_each_delay_shot_idx_list[idx_delay][nearest_int_pair_idx]}")

        paired_diff = now_on_int - paired_off_int
        original_off_int = off_norm_int_list[idx_delay][idx_data_in_delay]
        original_diff = now_on_int - original_off_int
        plt.plot(common_q, paired_diff, label=f"paired_dff")
        plt.plot(common_q, original_diff, label=f"ordered_diff")
        plt.legend()
        plt.show()
        # plt.plot(common_q, paired_diff, label=f"on{idx_data_in_delay}_paired_diff")
        paired_diff_intg = np.sum(np.abs(paired_diff))
        if paired_diff_intg < 0.02:
            norm_paired_diff_int_list[idx_delay].append(paired_diff)
            diff_all_intg_list[idx_delay].append(paired_diff_intg)

    # plt.show()

# for idx_delay, delay_val in enumerate(delay_list):
#     plt.title(f"diff hist of each delay (delay={delay_val})")
#     plt.hist(diff_all_intg_list[idx_delay])
#     plt.show()
#
# for idx_delay, delay_paired_diff in enumerate(norm_paired_diff_int_list):
#     title = f"[{now_run_name}] {idx_delay + 1}-th delay : {delay_list[idx_delay]}ns paired diff"
#     plt.title(title)
#     for each_diff in delay_paired_diff:
#         plt.plot(common_q, each_diff)
#     plt.show()
#
# each_delay_diff = []
# for delay_dat_list in norm_paired_diff_int_list:
#     delay_dat_list = np.array(delay_dat_list)
#     now_delay_avg_diff = np.average(delay_dat_list, axis=0)
#     each_delay_diff.append(now_delay_avg_diff)
#
# plt.title(f"{now_run_name} each delay diff (on - neg) avg")
# for idx_delay, each_avg in enumerate(each_delay_diff):
#     plt.plot(common_q, each_avg, label=f"{idx_delay}-th_{delay_list[idx_delay]}ns")
# plt.legend()
# plt.show()