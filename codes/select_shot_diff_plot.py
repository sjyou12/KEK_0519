import os
import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
import numpy as np
import matplotlib.pyplot as plt
from marccd import MarCCD
from codes.mccd2dat import load_n_azi_intg

folder_name = "KI_Methanol"
run_num = 41

# shot number is 0 based !
shot_num_start = 0
# shot_num_end = 24
shot_num_end = 10

norm_start_q = 6
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
on_int_list = []
off_int_list = []
norm_diff_int_list = []
each_delay_shot_idx_list = []
on_waxs_intg_list = []
off_waxs_intg_list = []
diff_waxs_intg_list = []
num_outlier = 0
for idx_delay in range(len(delay_list)):
    diff_int_list.append([])
    norm_diff_int_list.append([])
    each_delay_shot_idx_list.append([])
    on_int_list.append([])
    off_int_list.append([])
    on_waxs_intg_list.append([])
    off_waxs_intg_list.append([])
    diff_waxs_intg_list.append([])
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
    raw_waxs_intg_cut = 1.0e6
    if norm_off_val < raw_waxs_intg_cut or norm_on_val < raw_waxs_intg_cut:
        print("shot num : ", shot_num)
        plt.plot(common_q, now_on_int, label="on")
        plt.plot(common_q, now_off_int, label="off")
        plt.legend()
        plt.show()
        num_outlier += 1
        continue
    norm_on_int = now_on_int / norm_on_val
    norm_off_int = now_off_int / norm_off_val
    now_diff = now_on_int - now_off_int
    now_norm_diff = norm_on_int - norm_off_int
    norm_diff_intg = np.sum(np.abs(now_norm_diff))
    # if (norm_diff_intg > 0.013):
    #     continue
    now_time_delay_idx = remainder2idx[(shot_num % divider_for_delay)]
    diff_int_list[now_time_delay_idx].append(now_diff)
    norm_diff_int_list[now_time_delay_idx].append(now_norm_diff)
    each_delay_shot_idx_list[now_time_delay_idx].append(shot_num)
    on_int_list[now_time_delay_idx].append(now_on_int)
    off_int_list[now_time_delay_idx].append(now_off_int)
    diff_waxs_intg_list[now_time_delay_idx].append(norm_diff_intg)
    on_waxs_intg_list[now_time_delay_idx].append(norm_on_val)
    off_waxs_intg_list[now_time_delay_idx].append(norm_off_val)

    print_per_shot = 20
    if (shot_num % print_per_shot) == (print_per_shot - 1):
        print(f"load dat done until shot{shot_num + 1}")

num_plot_in_one_fig = 10
for idx_delay, delay_diff_list in enumerate(norm_diff_int_list):
    plt.title("diff hist of each delay")
    plt.hist(diff_waxs_intg_list[idx_delay])
    plt.show()

    plt.title("on hist")
    plt.hist(on_waxs_intg_list[idx_delay])
    plt.show()

    plt.title("off hist")
    plt.hist(off_waxs_intg_list[idx_delay])
    plt.show()

    title = f"[{now_run_name}] {idx_delay + 1}-th delay : {delay_list[idx_delay]}ns"
    for idx_data, each_shot_diff in enumerate(delay_diff_list):
        plt.plot(common_q, each_shot_diff, label=f"shot{each_delay_shot_idx_list[idx_delay][idx_data]}_norm_diff")
        if (idx_data % num_plot_in_one_fig) == (num_plot_in_one_fig - 1):
            plt.title(title)
            plt.legend(fontsize=8)
            plt.show()
    if (idx_data % num_plot_in_one_fig) != (num_plot_in_one_fig - 1):
        plt.title(title)
        plt.legend(fontsize=8)
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
print("number of outlier : {}".format(num_outlier))
