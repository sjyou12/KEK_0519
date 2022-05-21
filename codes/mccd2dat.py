import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
import numpy as np
import os
import matplotlib.pyplot as plt
from marccd import MarCCD

folder_name = "I2_Benzene"
run_num = 85
# folder_name = "KI_Methanol"
# run_num = 43
file_common_path = "../data/"
common_save_path = "../results/shot_azi_intg/"
do_overwrite_file = False


def load_n_azi_intg(file_common_path, common_save_path, folder_name, run_num, do_overwrite_file=False):
    poni_file = "../calibrate/si22.poni"
    # mask_file_name = "../calibrate/si22_mask.npy"
    mask_file_name = "../calibrate/run_45_mask_20220520.npy"
    dark_file_name = "../results/dark_run15.npy"
    # dark_file_name = "../results/background_run19.npy"

    now_run_name = f"run{run_num:02d}"
    save_sub_folder = f"{common_save_path}{now_run_name}/"
    try:
        os.mkdir(save_sub_folder)
    except FileExistsError:
        print("dir already exist : ", save_sub_folder)

    now_run_dir = f"{file_common_path}{folder_name}/{now_run_name}/"
    dir_file_list = os.listdir(now_run_dir)
    delay_info_file = f"{now_run_dir}{now_run_name}_delayinfo.log"
    delay_list = np.loadtxt(delay_info_file)
    print("now run delay list : ", delay_list)
    on_file_name_list = []
    neg_file_name_list = []
    for each_file_name in dir_file_list:
        if each_file_name.startswith(f"{now_run_name}_on") and each_file_name.endswith(".mccd"):
            on_file_name_list.append(each_file_name)
        elif each_file_name.startswith(f"{now_run_name}_neg") and each_file_name.endswith(".mccd"):
            neg_file_name_list.append(each_file_name)
    on_file_name_list.sort()
    neg_file_name_list.sort()
    ai = pyFAI.load(poni_file)
    mask = np.load(mask_file_name)
    dark_img = np.load(dark_file_name)

    q_val, dark_int = ai.integrate1d(dark_img, npt=1024, mask=mask, unit="q_A^-1", polarization_factor=0.99)
    plt.plot(q_val, dark_int)
    plt.title("dark 1D curve")
    plt.show()

    for shot_idx, now_file_name in enumerate(on_file_name_list):
        save_neg_file = f"{save_sub_folder}{now_run_name}_shot{shot_idx:03d}_off.dat"
        save_on_file = f"{save_sub_folder}{now_run_name}_shot{shot_idx:03d}_on.dat"

        # do overwirte True : do integrate1d
        # False : file exist -> skip
        # False : file not exist -> do
        # Thus : (do ovewrite) or (file not exist)

        if do_overwrite_file or (not os.path.isfile(save_on_file)) :
            now_on_img_name = f"{now_run_dir}{now_file_name}"
            on_img = np.array(MarCCD(now_on_img_name).image) - dark_img
            # plt.pcolor(MarCCD(now_on_img_name).image)
            # plt.colorbar()
            # plt.title("raw img")
            # plt.show()
            #
            # plt.pcolor(dark_img)
            # plt.colorbar()
            # plt.title("dark img")
            # plt.show()
            #
            # plt.pcolor(on_img)
            # plt.colorbar()
            # plt.clim(0, None)
            # plt.title("on img (raw - dark)")
            # plt.show()

            # now_q, now_test_int = ai.integrate1d(data=test_img, npt=1024, mask=mask, unit="q_A^-1",
            #                                  polarization_factor=0.99, filename=save_on_file)
            now_q, now_on_int = ai.integrate1d(data=on_img, npt=1024, mask=mask, unit="q_A^-1",
                                               polarization_factor=0.99, filename=save_on_file)
            # if shot_idx > 20:
            plt.plot(now_q, now_on_int)
            plt.title(f"raw img 1D curve shot{shot_idx}")
            plt.show()

        # if do_overwrite_file or (not os.path.isfile(save_neg_file)):
        #     now_neg_img_name = f"{now_run_dir}{neg_file_name_list[shot_idx]}"
        #     off_img = np.array(MarCCD(now_neg_img_name).image) - dark_img
        #     now_q, now_off_int = ai.integrate1d(data=off_img, npt=1024, mask=mask, unit="q_A^-1",
        #                                         polarization_factor=0.99, filename=save_neg_file)
        #     # if shot_idx > 20:
        #     plt.plot(now_q, now_off_int)
        #     plt.title(f"raw img 1D curve shot{shot_idx}")
        #     plt.show()
        # plt.plot(now_q, now_on_int, label="on int")        # loaded intensity check
        # plt.plot(now_q, now_off_int, label="off int")
        # plt.legend()
        # plt.title("loaded on, off int check")
        # plt.show()

        print_per_shot = 20
        if (shot_idx % print_per_shot) == (print_per_shot - 1):
            print(f"integrate done until shot{shot_idx + 1}")


if __name__ == "__main__":
    load_n_azi_intg(file_common_path, common_save_path, folder_name, run_num, do_overwrite_file)
