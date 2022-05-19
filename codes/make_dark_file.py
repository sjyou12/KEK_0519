from marccd import MarCCD
import numpy as np
import matplotlib.pyplot as plt
import os

# dark_dir = "../data/background/run15/"
# dark_save_path = "../results/dark_run15.npy"
dark_dir = "../data/I2_Benzene/run19/"
dark_save_path = "../results/background_run19.npy"
dir_file_list = os.listdir(dark_dir)

img_data_list = []
for each_file_name in dir_file_list:
    if each_file_name.endswith(".mccd"):
        print("read : ", each_file_name)
        now_data = np.array(MarCCD(f"{dark_dir}{each_file_name}").image)
        img_data_list.append(now_data)
        # plt.pcolor(now_data, cmap='gray_r', vmin=20, vmax=60)
        # plt.colorbar()
        # plt.axis("off")
        # plt.show()

dark_avg = np.average(np.array(img_data_list), axis=0)

plt.pcolor(dark_avg, cmap="gray_r")
plt.colorbar()
plt.axis("off")
plt.show()

np.save(dark_save_path, dark_avg)
print("save dark : ", dark_save_path)


