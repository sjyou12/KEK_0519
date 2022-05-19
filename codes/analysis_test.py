import os
import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
import numpy as np
import matplotlib.pyplot as plt

folder_name = "I2_Benzene"
run_num = 14

preprocess_file_dir = "../results/shot_azi_intg/"
rawdata_file_dir = "../data/"
now_run_name = f"run{run_num:02d}"
now_run_azi_int_dir = f"{preprocess_file_dir}/{now_run_name}/"

dir_file_list = os.listdir(now_run_azi_int_dir)
# print(dir_file_list)

delay_info_file = f"{rawdata_file_dir}{folder_name}/{now_run_name}/{now_run_name}_delayinfo.log"
delay_list = np.loadtxt(delay_info_file)
print("now run delay list : ", delay_list)

on_file_name_list = []
neg_file_name_list = []
for each_file_name in dir_file_list:
    if each_file_name.endswith("_on.dat"):
        on_file_name_list.append(each_file_name)
    elif each_file_name.endswith("_off.dat"):
        neg_file_name_list.append(each_file_name)

on_file_name_list.sort()
neg_file_name_list.sort()
on_shot_len = len(on_file_name_list)

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

norm_start_q = 4
norm_end_q = 6


common_q = []
on_int_each_delay = []
diff_int_list = []
on_int_list = []
off_int_list = []
norm_diff_int_list = []
each_delay_shot_idx_list = []
on_waxs_intg_list = []
off_waxs_intg_list = []
for idx_delay in range(len(delay_list)):
    diff_int_list.append([])
    norm_diff_int_list.append([])
    each_delay_shot_idx_list.append([])
    on_int_list.append([])
    off_int_list.append([])
    on_waxs_intg_list.append([])
    off_waxs_intg_list.append([])
intg_start_idx = None
intg_end_idx = None
for shot_idx, now_file_name in enumerate(on_file_name_list):
    on_file_path = f"{now_run_azi_int_dir}{now_file_name}"
    off_file_path = f"{now_run_azi_int_dir}{neg_file_name_list[shot_idx]}"
    on_rawdata = np.loadtxt(on_file_path)
    off_rawdata = np.loadtxt(off_file_path)

    if len(common_q) == 0:
        common_q = on_rawdata[:, 0]
        intg_start_idx, intg_end_idx = calc_q_cut_idx(common_q, norm_start_q, norm_end_q)
    now_on_int = on_rawdata[:, 1]
    now_off_int = off_rawdata[:, 1]

    norm_off_val = np.sum(now_off_int[intg_start_idx:intg_end_idx])
    norm_on_val = np.sum(now_on_int[intg_start_idx:intg_end_idx])
    norm_on_int = now_on_int / norm_on_val
    norm_off_int = now_off_int / norm_off_val
    # if shot_idx < 10:
    #     plt.plot(common_q, now_on_int, label=f"shot{shot_idx}_on")
    #     plt.plot(common_q, now_off_int, label=f"shot{shot_idx}_off")
    #     plt.legend()
    #     plt.show()
    now_diff = now_on_int - now_off_int
    now_norm_diff = norm_on_int - norm_off_int
    now_time_delay_idx = remainder2idx[(shot_idx % divider_for_delay)]
    diff_int_list[now_time_delay_idx].append(now_diff)
    norm_diff_int_list[now_time_delay_idx].append(now_norm_diff)
    each_delay_shot_idx_list[now_time_delay_idx].append(shot_idx)
    on_int_list[now_time_delay_idx].append(now_on_int)
    off_int_list[now_time_delay_idx].append(now_off_int)

    intg_start_idx, intg_end_idx = calc_q_cut_idx(common_q, 1, 2)
    on_intg = np.sum(now_on_int[intg_start_idx:intg_end_idx])
    on_waxs_intg_list[now_time_delay_idx].append(on_intg)
    off_intg = np.sum(now_off_int[intg_start_idx:intg_end_idx])
    off_waxs_intg_list[now_time_delay_idx].append(off_intg)

    print_per_shot = 20
    if (shot_idx % print_per_shot) == (print_per_shot - 1):
        print(f"load dat done until shot{shot_idx + 1}")


num_plot_in_one_fig = 10
# for idx_delay, delay_diff_list in enumerate(norm_diff_int_list):
#     title = f"[{now_run_name}] {idx_delay + 1}-th delay : {delay_list[idx_delay]}ns"
#     for idx_data, each_shot_diff in enumerate(delay_diff_list):
#         plt.plot(common_q, each_shot_diff, label=f"shot{each_delay_shot_idx_list[idx_delay][idx_data]}_norm_diff")
#         if (idx_data % num_plot_in_one_fig) == (num_plot_in_one_fig - 1):
#             plt.title(title)
#             plt.legend(fontsize=8)
#             plt.show()
#     if (idx_data % num_plot_in_one_fig) != (num_plot_in_one_fig - 1):
#         plt.title(title)
#         plt.legend(fontsize=8)
#         plt.show()
for idx_delay, delay_dat_list in enumerate(on_int_list):
    plt.title(f"{idx_delay + 1}-th delay : {delay_list[idx_delay]}ns")
    for idx_data, each_diff_dat in enumerate(delay_dat_list):
        now_shot_num = each_delay_shot_idx_list[idx_delay][idx_data]
        off_int = off_int_list[idx_delay][idx_data]
        # plt.plot(common_q, each_diff_dat, label=f"shot{now_shot_num+1}_off")
        plt.plot(common_q, off_int, label=f"shot{now_shot_num+1}_on")
        if (idx_data % num_plot_in_one_fig) == (num_plot_in_one_fig - 1):
            # plt.title(title)
            plt.legend(fontsize=8)
            plt.show()
    if (idx_data % num_plot_in_one_fig) != (num_plot_in_one_fig - 1):
        # plt.title(title)
        plt.legend(fontsize=8)
        plt.show()
        plt.legend()
        plt.show()


# for idx_delay, delay_dat_list in enumerate(diff_int_list):
#     plt.title(f"{idx_delay + 1}-th delay : {delay_list[idx_delay]}ns")
#     for idx_data, each_diff_dat in enumerate(delay_dat_list):
#         now_shot_num = each_delay_shot_idx_list[idx_delay][idx_data]
#         plt.plot(common_q, each_diff_dat, label=f"shot{now_shot_num+1}_diff")
#     plt.legend()
#     plt.show()
#

plt.title("on q 4 ~ 6 intg hist")
plt.hist(on_waxs_intg_list)
plt.show()
plt.title("off q 4 ~ 6 intg hist")
plt.hist(off_waxs_intg_list)
plt.show()

for idx_delay, delay_diff_intg in enumerate(on_waxs_intg_list):
    plt.title(f"{idx_delay + 1}-th delay : {delay_list[idx_delay]}ns on q 4 ~ 6 intg hist")
    plt.hist(delay_diff_intg, bins=50)
    plt.show()
    plt.title(f"{idx_delay + 1}-th delay : {delay_list[idx_delay]}ns off q 4 ~ 6 intg hist")
    plt.hist(off_waxs_intg_list[idx_delay], bins=50)
    plt.show()

for idx_delay, delay_dat_list in enumerate(norm_diff_int_list):
    plt.title(f"{idx_delay + 1}-th delay : {delay_list[idx_delay]}ns")
    for idx_data, each_diff_dat in enumerate(delay_dat_list):
        now_shot_num = each_delay_shot_idx_list[idx_delay][idx_data]
        plt.plot(common_q, each_diff_dat, label=f"shot{now_shot_num+1}_norm_diff")
    # plt.legend()
    plt.show()

each_delay_diff = []
for delay_dat_list in norm_diff_int_list:
    delay_dat_list = np.array(delay_dat_list)
    now_delay_avg_diff = np.average(delay_dat_list, axis=0)
    each_delay_diff.append(now_delay_avg_diff)

plt.title(f"{now_run_name} each delay diff (on - neg) avg")
for idx_delay, each_avg in enumerate(each_delay_diff):
    plt.plot(common_q, each_avg, label=f"{idx_delay}-th_{delay_list[idx_delay]}ns")
plt.legend()
plt.show()

