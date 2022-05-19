from marccd import MarCCD
import numpy as np
import matplotlib.pyplot as plt
import os
import pyFAI

dark_dir = "../data/I2_Benzene/run19/"
# dark_save_path = "../results/dark_run15.npy"
dir_file_list = os.listdir(dark_dir)

poni_file = "../calibrate/si22.poni"
ai = pyFAI.load(poni_file)

mask_file_name = "../calibrate/run_14_mask_20220519.npy"
mask = np.load(mask_file_name)

common_q = []
int_dat_list = []
for each_file_name in dir_file_list:
    if each_file_name.endswith(".mccd"):
        print("read : ", each_file_name)
        now_data = np.array(MarCCD(f"{dark_dir}{each_file_name}").image)
        q_val, now_int = ai.integrate1d(now_data, npt=1024, unit="q_A^-1", polarization_factor=0.99)

        if len(common_q) == 0:
            common_q = q_val
        int_dat_list.append(now_int)

        # plt.pcolor(now_data, cmap='gray_r', vmin=20, vmax=60)
        # plt.colorbar()
        # plt.axis("off")
        # plt.show()

for each_int in int_dat_list:
    plt.plot(common_q, each_int)
# plt.ylim(3500, 4500)
plt.show()
# plt.pcolor(dark_avg, cmap="gray_r", vmin=500, vmax=600)
# plt.colorbar()
# plt.axis("off")
# plt.show()
#
# np.save(dark_save_path, dark_avg)
# print("save dark : ", dark_save_path)


