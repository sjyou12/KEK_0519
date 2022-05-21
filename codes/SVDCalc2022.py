import numpy as np
from matplotlib import pyplot as plt

class SVDCalc:
    def __init__(self, data):
        self.originalData = data
        print("input data shape : ", data.shape)
        print(f"LSV {data.shape[0]} data point // RSV {data.shape[1]} data point")
        self.leftVec = None
        self.rightVecTrans = None
        self.rightVec = None
        self.singValVec = None
        self.meanLeftVec = None
        self.meanRightVec = None
        self.meanSingValVec = None
        self.meanNum = None

    def calc_svd(self):
        u, s, vh = np.linalg.svd(self.originalData, full_matrices=False)
        self.leftVec = u
        self.rightVecTrans = vh
        self.singValVec = s
        self.rightVec = np.transpose(self.rightVecTrans)
        return s

    @staticmethod
    def choose_n_left_column(array, num):
        tempArr = np.transpose(array)
        tempArr = tempArr[0:num]
        return np.transpose(tempArr)

    def pick_meaningful_data(self, mean_num):
        self.meanNum = mean_num
        self.meanSingValVec = self.singValVec[0:mean_num]
        self.meanLeftVec = self.choose_n_left_column(self.leftVec, mean_num)
        self.meanRightVec = self.choose_n_left_column(self.rightVec, mean_num)

    def plot_singular_val(self, singular_show_num=10, plot_log_also=True, title_data_name=""):
        sval_fig, sval_axs = plt.subplots()
        self.plot_singular_val_given_axis(sval_axs, singular_show_num, plot_log_also, title_data_name)
        sval_fig.tight_layout()
        sval_fig.show()

    def plot_singular_val_given_axis(self, axs, singular_show_num=10, plot_log_also=True, title_data_name=""):
        plot_sing_val = self.singValVec[:singular_show_num]
        plot_sing_val_log = np.log(plot_sing_val)
        len_plot = plot_sing_val.shape[0]
        plot_x = range(1, len_plot + 1)

        color_r = 'tab:red'
        ax1 = axs
        ax1.set_xlabel("index of singular value")
        ax1.set_ylabel("singular value", color=color_r)
        ax1.scatter(plot_x, plot_sing_val, color=color_r)
        ax1.plot(plot_x, plot_sing_val, color=color_r)
        ax1.tick_params(axis='y', labelcolor=color_r)
        ax1.set_xticks(plot_x)

        if plot_log_also:
            ax2 = axs.twinx()
            color_b = 'tab:blue'
            ax2.set_ylabel("log scale singular value", color=color_b)
            ax2.scatter(plot_x, plot_sing_val_log, color=color_b)
            ax2.plot(plot_x, plot_sing_val_log, color=color_b)
            ax2.tick_params(axis='y', labelcolor=color_b)
            ax2.set_xticks(plot_x)

        axs.set_title(f"{title_data_name} singular value (front {len_plot})")

    def plot_left_vec(self, sep_graph=False, flip=None, title_data_name="", plot_x_val=None):
        lsv_fig, lsv_axs = plt.subplots()
        self.plot_left_vec_given_axis(lsv_axs, sep_graph, flip, title_data_name, plot_x_val)
        lsv_fig.show()

    def plot_left_vec_given_axis(self, axs, sep_graph=False, flip=None, title_data_name="", plot_x_val=None):
        now_plot_title = f"{title_data_name} left singular vectors"
        if flip is not None:
            flip = np.array(flip)
            flip_len = flip.shape[0]
            assert flip_len == self.meanNum, f"flip len ({flip_len}) != meaningful SingVal num {self.meanNum}\nflip : {flip}"
            print("now flip LSV :", flip)
        transLeft = np.transpose(self.meanLeftVec)
        if sep_graph:
            for sp_idx in range(0, self.meanNum):
                sep_fig, sep_axs = plt.subplots()
                if plot_x_val is None:
                    sep_axs.plot(transLeft[sp_idx], label=("leftVec" + str(sp_idx + 1)))
                else:
                    sep_axs.plot(plot_x_val, transLeft[sp_idx], label=("leftVec" + str(sp_idx + 1)))
                sep_axs.set_title(now_plot_title)
                sep_axs.legend()
                sep_fig.show()
                plt.close(sep_fig)
        for sp_idx in range(0, self.meanNum):
            now_plot_y = transLeft[sp_idx]
            if flip is not None:
                now_plot_y = transLeft[sp_idx] * flip[sp_idx]
            if plot_x_val is None:
                axs.plot(now_plot_y, label=("leftVec" + str(sp_idx + 1)))
            else:
                axs.plot(plot_x_val, now_plot_y, label=("leftVec" + str(sp_idx + 1)))
        axs.set_title(now_plot_title)
        axs.legend()

    def plot_right_vec(self, sep_graph=False, flip=None, title_data_name="", plot_x_val=None,
                       x_val_as_text=False, v_line_list=None):
        rsv_fig, rsv_axs = plt.subplots()
        self.plot_right_vec_given_axis(rsv_axs, sep_graph, flip, title_data_name, plot_x_val, x_val_as_text, v_line_list)
        rsv_fig.show()

    def plot_right_vec_given_axis(self, axs, sep_graph=False, flip=None, title_data_name="", plot_x_val=None,
                                  x_val_as_text=False, v_line_list=None):
        if v_line_list is None:
            v_line_list = []
        now_plot_title = f"{title_data_name} right singular vectors"
        if flip is not None:
            flip = np.array(flip)
            flip_len = flip.shape[0]
            assert flip_len == self.meanNum, f"flip len ({flip_len}) != meaningful SingVal num {self.meanNum}\nflip : {flip}"
            print("now flip RSV :", flip)
        transRight = np.transpose(self.meanRightVec)
        if sep_graph:
            for sp_idx in range(0, self.meanNum):
                sep_fig, sep_axs = plt.subplots()
                if plot_x_val is None:
                    sep_axs.plot(transRight[sp_idx], label=("rightVec" + str(sp_idx + 1)))
                else:
                    sep_axs.plot(plot_x_val, transRight[sp_idx], label=("rightVec" + str(sp_idx + 1)))
                for v_line_val in v_line_list:
                    sep_axs.axvline(x=v_line_val, color='r')
                sep_axs.set_title(now_plot_title)
                sep_axs.axhline(y=0, c='k', alpha=0.3)
                sep_axs.legend()
                sep_fig.show()
                plt.close(sep_fig)
        for sp_idx in range(0, self.meanNum):
            now_plot_y = transRight[sp_idx]
            value_len = len(now_plot_y)
            if flip is not None:
                now_plot_y = transRight[sp_idx] * flip[sp_idx]
            if plot_x_val is None:
                axs.plot(now_plot_y, label=("rightVec" + str(sp_idx + 1)))
            else:
                if x_val_as_text:
                    axs.plot(now_plot_y, label=("rightVec" + str(sp_idx + 1)))
                    # TODO :  after matplotlib 3.5 version, set_xticklables is merged to set_xticks
                    axs.set_xticks(ticks=range(value_len))
                    axs.set_xticklabels(labels=plot_x_val[:value_len])
                    # try:
                    #     axs.set_xticks(ticks=range(value_len), labels=plot_x_val[:value_len])
                    # except:
                    #     print("plot xticks error :", range(value_len), plot_x_val[:value_len])
                else:
                    axs.plot(plot_x_val, now_plot_y, label=("rightVec" + str(sp_idx + 1)))
            for v_line_val in v_line_list:
                axs.axvline(x=v_line_val, color='r')
        axs.set_title(now_plot_title)
        axs.legend()

    def file_output_singular_vectors(self, leftFp, rightFp):
        transLeft = np.transpose(self.meanLeftVec)
        transRight = np.transpose(self.meanRightVec)
        # print left
        leftFp.write("value-idx\t")
        for idx in range(0, self.meanNum):
            leftFp.write("leftVec" + str(idx + 1) + "\t")
        leftFp.write("\n")
        for line_num in range(0, self.meanLeftVec.shape[0]):
            leftFp.write(str(line_num + 1))
            leftFp.write("\t")
            for sp_idx in range(0, self.meanNum):
                leftFp.write(str(self.meanLeftVec[line_num][sp_idx]))
                leftFp.write("\t")
            leftFp.write("\n")

        # print right
        rightFp.write("value-idx\t")
        for idx in range(0, self.meanNum):
            rightFp.write("rightVec" + str(idx + 1) + "\t")
        rightFp.write("\n")
        for line_num in range(0, self.meanRightVec.shape[0]):
            rightFp.write(str(line_num + 1))
            rightFp.write("\t")
            for sp_idx in range(0, self.meanNum):
                rightFp.write(str(self.meanRightVec[line_num][sp_idx]))
                rightFp.write("\t")
            rightFp.write("\n")

    def file_output_svd_result(self, svd_file_dir, save_data_name=None, lsv_x_name=None, lsv_x_val=None,
                               rsv_x_name=None, rsv_x_val=None, rsv_x_val_as_text=False):

        if save_data_name is None:
            sVal_file_name = "SingVal.dat"
            rsv_file_name = "RSV.dat"
            lsv_file_name = "LSV.dat"
        else:
            sVal_file_name = save_data_name + "_SingVal.dat"
            rsv_file_name = save_data_name + "_RSV.dat"
            lsv_file_name = save_data_name + "_LSV.dat"

        sval_x_val = np.arange(self.meanSingValVec.shape[0])
        sval_save_data = np.transpose([sval_x_val, self.meanSingValVec])
        sval_save_header = "idx\tsingular_value"
        np.savetxt(svd_file_dir + sVal_file_name, sval_save_data, fmt="%.8e", delimiter="\t", header=sval_save_header)
        print("save singular value : ", svd_file_dir + sVal_file_name)

        if lsv_x_name is None:
            lsv_x_name = "lsv_idx"
        if lsv_x_val is None:
            lsv_x_val = np.arange(self.meanLeftVec.shape[0])

        lsv_save_data = np.concatenate((lsv_x_val[:, np.newaxis], self.meanLeftVec), axis=1)
        lsv_save_header = lsv_x_name
        for idx in range(self.meanNum):
            lsv_save_header += f"\tlsv_{idx}-th"
        np.savetxt(svd_file_dir + lsv_file_name, lsv_save_data, fmt="%.8e", delimiter="\t", header=lsv_save_header)
        print("save LSV : ", svd_file_dir + lsv_file_name)

        if rsv_x_name is None:
            rsv_x_name = "rsv_idx"
        if rsv_x_val is None:
            rsv_x_val = np.arange(self.meanRightVec.shape[0])
            rsv_x_val_as_text = False

        rsv_save_header = rsv_x_name
        for idx in range(self.meanNum):
            rsv_save_header += f"\trsv_{idx}-th"

        if rsv_x_val_as_text:
            rightFp = open((svd_file_dir + rsv_file_name), 'w')
            rightFp.write(f"# {rsv_save_header}\n")
            for line_num in range(self.meanRightVec.shape[0]):
                rightFp.write(f"{rsv_x_val[line_num]}\t")
                for sp_idx in range(self.meanNum):
                    rightFp.write("{0:.8e}\t".format(self.meanRightVec[line_num][sp_idx]))
                rightFp.write("\n")
            rightFp.close()
        else:
            rsv_save_data = np.concatenate((rsv_x_val[:, np.newaxis], self.meanRightVec), axis=1)
            np.savetxt(svd_file_dir + rsv_file_name, rsv_save_data, fmt="%.8e", delimiter="\t", header=rsv_save_header)
        print("save RSV : ", svd_file_dir + rsv_file_name)


    def file_output_singular_vectors_with_label(self, leftFp, rightFp, leftLableName, leftLabel, rightLabelName,
                                                rightLabel):
        # transLeft = np.transpose(self.meanLeftVec)
        # transRight = np.transpose(self.meanRightVec)

        # print left
        leftFp.write(leftLableName + "\t")
        for idx in range(self.meanNum):
            leftFp.write("leftVec" + str(idx + 1) + "\t")
        leftFp.write("\n")
        for line_num in range(self.meanLeftVec.shape[0]):
            leftFp.write(str(leftLabel[line_num]))
            leftFp.write("\t")
            for sp_idx in range(self.meanNum):
                leftFp.write(str(self.meanLeftVec[line_num][sp_idx]))
                leftFp.write("\t")
            leftFp.write("\n")

        # print right
        rightFp.write(rightLabelName + "\t")
        for idx in range(self.meanNum):
            rightFp.write("rightVec" + str(idx + 1) + "\t")
        rightFp.write("\n")
        for line_num in range(self.meanRightVec.shape[0]):
            rightFp.write(str(rightLabel[line_num]))
            rightFp.write("\t")
            for sp_idx in range(self.meanNum):
                rightFp.write(str(self.meanRightVec[line_num][sp_idx]))
                rightFp.write("\t")
            rightFp.write("\n")

    def file_output_singular_value(self, svalFp):
        svalFp.write("Singular-Value\n")
        for line_num in range(self.meanSingValVec.shape[0]):
            svalFp.write(str(self.meanSingValVec[line_num]) + "\n")