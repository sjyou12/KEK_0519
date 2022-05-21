import os
import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
import numpy as np
import matplotlib.pyplot as plt
from marccd import MarCCD
from codes.mccd2dat import load_n_azi_intg

# folder_name = "KI_Methanol"
# run_num = 45
folder_name = "I2_Benzene"
run_num = 84
show_hist = True
show_static = False

# shot number is 0 based !
shot_num_start = 0
# shot_num_end = 24
shot_num_end = 60

neg_delay_criteria = -1
#neg_delay_criteria = 0
ref_delay_idx = 0

diff_intg_cut = 0.01
# diff_intg_cut = 0.005s
# raw_waxs_intg_cut = 2.4e6
norm_start_q = 4
norm_end_q = 8
# raw_waxs_intg_cut = 2.4e6

raw_waxs_intg_cut = 1.5e6
# norm_start_q = 6
# norm_end_q = 8


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
on_waxs_intg_list_before_cut = []
off_waxs_intg_list_before_cut = []
diff_waxs_intg_list_before_cut = []
on_waxs_intg_list = []
off_waxs_intg_list = []
diff_waxs_intg_list = []

if show_static == True:
    neg_delay_int = []


for idx_delay in range(len(delay_list)):
    diff_int_list.append([])
    norm_diff_int_list.append([])
    each_delay_shot_idx_list.append([])
    on_int_list.append([])
    off_int_list.append([])
    on_waxs_intg_list.append([])
    off_waxs_intg_list.append([])
    diff_waxs_intg_list.append([])
    on_waxs_intg_list_before_cut.append([])
    off_waxs_intg_list_before_cut.append([])
    diff_waxs_intg_list_before_cut.append([])
intg_start_idx = None
intg_end_idx = None

num_col_in_one_fig = 3
num_row_in_one_fig = 2
num_suplot_in_one_fig = num_col_in_one_fig * num_row_in_one_fig
idx_col = 0
idx_row = 0
outlier_fig, outlier_axis = plt.subplots(nrows=num_row_in_one_fig, ncols=num_col_in_one_fig, figsize=(16, 10))
outlier_idx = 0
in_fig_idx = outlier_idx
now_axs = outlier_axis[idx_row][idx_col]

for shot_num in shot_num_list:
    on_file_path = f"{now_run_azi_int_dir}{now_run_name}_shot{shot_num:03d}_on.dat"
    # on_file_path = f"{now_run_azi_int_dir}{now_run_name}_shot{shot_num:03d}_on.dat"
    on_rawdata = np.loadtxt(on_file_path)
    if len(common_q) == 0:
        common_q = on_rawdata[:, 0]
        intg_start_idx, intg_end_idx = calc_q_cut_idx(common_q, norm_start_q, norm_end_q)
    now_on_int = on_rawdata[:, 1]

    norm_on_val = np.sum(now_on_int[intg_start_idx:intg_end_idx])
    now_time_delay_idx = remainder2idx[(shot_num % divider_for_delay)]
    on_waxs_intg_list_before_cut[now_time_delay_idx].append(norm_on_val)

    if norm_on_val < raw_waxs_intg_cut:
        title = f"{outlier_idx}-th outlier = shot {shot_num}"
        now_axs.set_title(title)
        now_axs.plot(common_q, now_on_int, label="on")
        now_axs.legend()
        outlier_idx += 1
        in_fig_idx = outlier_idx % num_suplot_in_one_fig
        idx_row = in_fig_idx // num_col_in_one_fig
        idx_col = in_fig_idx % num_col_in_one_fig
        now_axs = outlier_axis[idx_row][idx_col]
        if in_fig_idx == 0:
            outlier_fig.set_tight_layout(True)
            outlier_fig.show()
            outlier_fig, outlier_axis = plt.subplots(nrows=num_row_in_one_fig, ncols=num_col_in_one_fig,
                                                          figsize=(16, 10))
        continue
    norm_on_int = now_on_int / norm_on_val
    each_delay_shot_idx_list[now_time_delay_idx].append(shot_num)
    on_int_list[now_time_delay_idx].append(norm_on_int)
    on_waxs_intg_list[now_time_delay_idx].append(norm_on_val)

    if show_static == True and now_time_delay_idx == 0:
        neg_delay_int.append(now_on_int)

    print_per_shot = 20
    if (shot_num % print_per_shot) == (print_per_shot - 1):
        print(f"load dat done until shot{shot_num + 1}")

if show_static == True:
    static = np.average(neg_delay_int, axis=0)
    plt.plot(common_q, static)
    plt.title(f"{now_run_name} statics (neg delay average)")
    plt.show()

if in_fig_idx != 0:
    outlier_fig.set_tight_layout(True)
    outlier_fig.show()

on_hist_fig, on_axs = plt.subplots(nrows=2, ncols=num_col_in_one_fig, figsize=(16,10))
num_bin = 50
if show_hist:
    for idx_delay, delay_diff_list in enumerate(norm_diff_int_list):
        now_col_idx = idx_delay % num_col_in_one_fig
        now_delay_text = f" {idx_delay}-th delay ({delay_list[idx_delay]}ns)"

        on_axs[0][now_col_idx].set_title("on hist before cut" + now_delay_text)
        on_axs[0][now_col_idx].hist(on_waxs_intg_list_before_cut[idx_delay], bins=num_bin)
        on_axs[0][now_col_idx].axvline(x=raw_waxs_intg_cut, c='r')

        on_axs[1][now_col_idx].set_title("on hist after cut" + now_delay_text)
        on_axs[1][now_col_idx].hist(on_waxs_intg_list[idx_delay], bins=num_bin)

        if now_col_idx == (num_col_in_one_fig - 1):
            on_hist_fig.suptitle("on hist - cut below criteria")
            on_hist_fig.set_tight_layout(True)
            on_hist_fig.show()
            on_hist_fig, on_axs = plt.subplots(nrows=2, ncols=num_col_in_one_fig, figsize=(16, 10))
    if now_col_idx != (num_col_in_one_fig - 1):
        on_hist_fig.suptitle("on hist - cut below criteria")
        on_hist_fig.set_tight_layout(True)
        on_hist_fig.show()

neg_delay_idx = []
pos_delay_idx = []
for idx_delay, delay_val in enumerate(delay_list):
    if delay_val < neg_delay_criteria:
        neg_delay_idx.append(idx_delay)
    else:
        pos_delay_idx.append(idx_delay)
neg_delay_len = len(neg_delay_idx)
# now_use_neg_delay_idx = neg_delay_idx[0]
now_use_neg_delay_idx = ref_delay_idx
print(f"now use neg img : {now_use_neg_delay_idx}-th delay {delay_list[now_use_neg_delay_idx]}ns")

for idx_delay, delay_on_list in enumerate(on_int_list):
    for idx_data, on_norm_int in enumerate(delay_on_list):
        now_norm_diff = None
        if idx_delay in neg_delay_idx:
            now_pair_other_off = None
            if idx_data == (len(delay_on_list) - 1):
                now_pair_other_off = on_int_list[idx_delay][0]
            else:
                now_pair_other_off = on_int_list[idx_delay][idx_data + 1]
            now_norm_diff = on_norm_int - now_pair_other_off
        else:
            try:
                now_off_pair_int = on_int_list[now_use_neg_delay_idx][idx_data]
            except IndexError as e:
                print(e)
                now_off_pair_int = on_int_list[now_use_neg_delay_idx][-1]
                print("use last off int")
            now_norm_diff = on_norm_int - now_off_pair_int
        norm_diff_intg = np.sum(np.abs(now_norm_diff))
        diff_waxs_intg_list_before_cut[idx_delay].append(norm_diff_intg)
        if norm_diff_intg > diff_intg_cut:
            continue
        diff_waxs_intg_list[idx_delay].append(norm_diff_intg)
        norm_diff_int_list[idx_delay].append(now_norm_diff)

diff_hist_fig, diff_axs = plt.subplots(nrows=2, ncols=num_col_in_one_fig, figsize=(16,10))

if show_hist:
    for idx_delay, delay_diff_list in enumerate(norm_diff_int_list):
        now_col_idx = idx_delay % num_col_in_one_fig
        now_delay_text = f" {idx_delay}-th delay ({delay_list[idx_delay]}ns)"

        diff_axs[0][now_col_idx].set_title("diff hist before cut" + now_delay_text)
        diff_axs[0][now_col_idx].hist(diff_waxs_intg_list_before_cut[idx_delay], bins=num_bin)
        diff_axs[0][now_col_idx].axvline(x=diff_intg_cut, c='r')

        diff_axs[1][now_col_idx].set_title("diff hist after cut" + now_delay_text)
        diff_axs[1][now_col_idx].hist(diff_waxs_intg_list[idx_delay], bins=num_bin)

        if now_col_idx == (num_col_in_one_fig - 1):
            diff_hist_fig.suptitle("diff hist - cut above criteria")
            diff_hist_fig.set_tight_layout(True)
            diff_hist_fig.show()
            diff_hist_fig, diff_axs = plt.subplots(nrows=2, ncols=num_col_in_one_fig, figsize=(16, 10))
    if now_col_idx != (num_col_in_one_fig - 1):
        diff_hist_fig.suptitle("diff hist - cut above criteria")
        diff_hist_fig.set_tight_layout(True)
        diff_hist_fig.show()

num_plot_in_one_fig = 5
for idx_delay, delay_diff_list in enumerate(norm_diff_int_list):
    diff_indiv_fig, diff_indiv_axs = plt.subplots(nrows=num_row_in_one_fig, ncols=num_col_in_one_fig, figsize=(16, 10))
    title = f"[{now_run_name}] {idx_delay}-th delay : {delay_list[idx_delay]}ns"
    in_fig_idx = 0
    idx_col = 0
    idx_row = 0
    now_axs = diff_indiv_axs[idx_row][idx_col]
    idx_data = 0
    for idx_data, each_shot_diff in enumerate(delay_diff_list):
        now_axs.plot(common_q, each_shot_diff, label=f"shot{each_delay_shot_idx_list[idx_delay][idx_data]}_norm_diff")
        if (idx_data % num_plot_in_one_fig) == (num_plot_in_one_fig - 1):
            now_axs.set_title(title)
            now_axs.legend()
            in_fig_idx = (in_fig_idx + 1) % num_suplot_in_one_fig
            if in_fig_idx == 0:
                diff_indiv_fig.set_tight_layout(True)
                diff_indiv_fig.show()
                diff_indiv_fig, diff_indiv_axs = plt.subplots(nrows=num_row_in_one_fig, ncols=num_col_in_one_fig,
                                                              figsize=(16, 10))
            idx_row = in_fig_idx // num_col_in_one_fig
            idx_col = in_fig_idx % num_col_in_one_fig
            now_axs = diff_indiv_axs[idx_row][idx_col]
    if (idx_data % num_plot_in_one_fig) != (num_plot_in_one_fig - 1):
        now_axs.set_title(title)
        now_axs.legend(fontsize=8)
        diff_indiv_fig.set_tight_layout(True)
        diff_indiv_fig.show()
    elif in_fig_idx != 0:
        diff_indiv_fig.set_tight_layout(True)
        diff_indiv_fig.show()
each_delay_diff = []
delay_to_plot = []
for idx_delay, delay_dat_list in enumerate(norm_diff_int_list):
    delay_to_plot.append(idx_delay)
    delay_dat_list = np.array(delay_dat_list)
    now_delay_avg_diff = np.average(delay_dat_list, axis=0)
    each_delay_diff.append(now_delay_avg_diff)

fig, axs = plt.subplots()
axs.set_title(f"{now_run_name} each delay diff (on - neg) avg")
for idx_avg_data, each_avg in enumerate(each_delay_diff):
    idx_delay = delay_to_plot[idx_avg_data]
    axs.plot(common_q, each_avg, label=f"{idx_delay}-th_{delay_list[idx_delay]}ns")
    axs.axhline(y=0, c='k', alpha=0.3)

axs.legend()
fig.set_tight_layout(True)
fig.show()

each_fig, each_axs = plt.subplots(nrows=num_row_in_one_fig, ncols=num_col_in_one_fig, figsize=(16,10))
each_fig.suptitle(f"{now_run_name} each delay diff (on - neg) avg")
for idx_avg_data, each_avg in enumerate(each_delay_diff):
    idx_row = idx_avg_data // num_col_in_one_fig
    idx_col = idx_avg_data % num_col_in_one_fig
    now_axs = each_axs[idx_row][idx_col]
    idx_delay = delay_to_plot[idx_avg_data]
    now_axs.plot(common_q, each_avg, label=f"{idx_delay}-th_{delay_list[idx_delay]}ns")
    now_axs.axhline(y=0, c='k', alpha=0.6)
    now_axs.legend()

each_fig.set_tight_layout(True)
each_fig.show()


