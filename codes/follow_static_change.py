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
run_num_list = range(64, 73)
preprocess_file_dir = "../results/shot_azi_intg/"
rawdata_file_dir = "../data/"

q_start = 1
q_end = 2
ref_q_start = 6
ref_q_end = 8

intg_cut_68 = 1e6
delay_idx_to_svd = 3

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
    # print(convert_dict)
    return divider, convert_dict


cut_0_2 = []
cut_6_8 = []
static_int_list = []

for run_idx, now_run in enumerate(run_num_list):
    now_run_name = f"run{run_num_list[run_idx]:02d}"
    now_run_azi_int_dir = f"{preprocess_file_dir}{now_run_name}/"

    static_int = []

    delay_info_file = f"{rawdata_file_dir}{folder_name}/{now_run_name}/{now_run_name}_delayinfo.log"
    delay_list = np.loadtxt(delay_info_file)

    dir_file_list = os.listdir(now_run_azi_int_dir)
    on_file_name_list = []
    for each_file_name in dir_file_list:
        if each_file_name.endswith("_on.dat"):
            on_file_name_list.append(each_file_name)
    shot_num_list = range(len(on_file_name_list))

    divider_for_delay, remainder2idx = make_remainder_to_delay_idx_dict(delay_list)

    common_q = []

    intg_start_idx = None
    intg_end_idx = None

    for shot_num in shot_num_list:
        on_file_path = f"{now_run_azi_int_dir}{now_run_name}_shot{shot_num:03d}_on.dat"
        # on_file_path = f"{now_run_azi_int_dir}{now_run_name}_shot{shot_num:03d}_on.dat"
        on_rawdata = np.loadtxt(on_file_path)
        if len(common_q) == 0:
            common_q = on_rawdata[:, 0]
            intg_start_idx_02, intg_end_idx_02 = calc_q_cut_idx(common_q, q_start, q_end)
            intg_start_idx_68, intg_end_idx_68 = calc_q_cut_idx(common_q, ref_q_start, ref_q_end)
        now_on_int = on_rawdata[:, 1]

        norm_on_val = np.sum(now_on_int[intg_start_idx_02:intg_end_idx_02])
        norm_on_val_ref = np.sum(now_on_int[intg_start_idx_68:intg_end_idx_68])
        if norm_on_val_ref < intg_cut_68:
            continue
        norm_on_int = now_on_int / norm_on_val
        now_time_delay_idx = remainder2idx[(shot_num % divider_for_delay)]
        if now_time_delay_idx == 0 or now_time_delay_idx == 1:
            cut_0_2.append(norm_on_val)
            static_int.append(norm_on_int)
            cut_6_8.append(norm_on_val_ref)

    static_int_list.append(np.average(static_int, axis=0))

on_hist_fig, on_axs = plt.subplots()
num_bin = 50
on_axs.set_title(f"run {run_num_list[0]}-{run_num_list[-1]} -3ns delay itg btw q {ref_q_start}-{ref_q_end}")
on_axs.hist(cut_6_8, bins=num_bin)
on_axs.axvline(x=intg_cut_68, c='r')
on_hist_fig.show()

x_axis = range(len(cut_0_2))
plt.plot(x_axis, cut_0_2, linestyle="none", marker="o")
plt.title(f"run {run_num_list[0]}-{run_num_list[-1]} -3ns delay itg btw q {q_start}-{q_end}")
plt.show()

plt.plot(x_axis, np.array(cut_0_2)/np.array(cut_6_8), linestyle="none", marker="o")
plt.title(f"run {run_num_list[0]}-{run_num_list[-1]} -3ns delay itg ratio btw q {q_start}-{q_end}/{ref_q_start}-{ref_q_end}")
# plt.ylim(0.876, 0.879)
# plt.ylim(2.91, 2.92)
plt.axhline(np.average(np.array(cut_0_2)/np.array(cut_6_8)), color='k')
plt.show()


for idx, now_static_int in enumerate(static_int_list):
    plt.plot(common_q, now_static_int, label=f"run {run_num_list[idx]}", lw=0.5)
plt.title(f"Change of static during run {run_num_list[0]}-{run_num_list[-1]}, normalized")
plt.xlim(1.4, 1.7)
plt.ylim(0.0105,0.01125)
plt.legend()
plt.show()

for idx, now_static_int in enumerate(static_int_list):
    plt.plot(common_q, now_static_int-static_int_list[0], label=f"run {run_num_list[idx]}")
plt.title(f"Change of static during run {run_num_list[0]}-{run_num_list[-1]}, difference")
plt.legend()
plt.show()