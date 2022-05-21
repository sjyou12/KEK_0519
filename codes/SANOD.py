from codes.SVDCalc2022 import SVDCalc
import numpy as np
import matplotlib.pyplot as plt
import copy

# time_delay = [-3, -2, -1.5, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 7, 10, 13, 18, 24, 32, 42, 56, 75, 100, 130, 180, 240, 320, 420, 560, 750, 1e3, 1.3e3, 1.8e3, 2.4e3]


def mgs(X, verbose=False):
    n, p = np.shape(X)
    if p > n:
        print("need to transpose input matrix")
        return
    else:
        q, r = np.linalg.qr(X)
        if verbose:
            print("q is ", q)
            print("r is ", r)
        return q, r

class SANODCalc:
    def __init__(self):
        self.pos_signal = []
        self.neg_signal = []
        self.total_signal_tp = []
        self.common_q_val = []
        self.do_file_save = False
        self.save_file_family_name = None
        self.non_ortho_components =[]
        self.num_non_ortho_comp = 0
        self.num_total_data = 0
        self.num_neg_data = 0
        self.num_pos_data = 0
        self.alpha_weights = []
        self.num_neg_comp = 0
        self.num_whole_comp = 0

    def set_file_save(self, do_file_save=True):
        self.do_file_save = do_file_save

    def set_file_family_name(self, family_name):
        self.save_file_family_name = family_name

    def set_data_for_sanod(self, pos_data, neg_data, q_val):
        """
        recommend data shape for input (pos_data, neg_data)
        ----shot 1 data----
        ----shot 2 data----
        ----shot 3 data----
        if shape is (A, B)
        A is # of data shot
        B is # of data point per each shot
        :param pos_data: numpy 2d array
        :param neg_data: numpy 2d array
        :param q_val: numpy 1d array. shape is (B)
        :return:
        """
        self.pos_signal = pos_data
        self.neg_signal = neg_data
        self.common_q_val = q_val

        self.num_neg_data = np.shape(neg_data)[0]
        self.num_pos_data = np.shape(pos_data)[0]

    @staticmethod
    def plot_singular_value(SVD_obj, plot_title, singular_show_num):
        print(SVD_obj.singValVec)
        data_y = SVD_obj.singValVec
        data_y = data_y[:singular_show_num]
        data_x = range(1, len(data_y) + 1)
        data_y_log = np.log(data_y)

        color_r = 'tab:red'
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("index of singular value")
        ax1.set_ylabel("singular value", color=color_r)
        ax1.scatter(data_x, data_y, color=color_r)
        ax1.plot(data_x, data_y, color=color_r)
        ax1.tick_params(axis='y', labelcolor=color_r)

        ax2 = ax1.twinx()
        color_b = 'tab:blue'
        ax2.set_ylabel("log scale singular value", color=color_b)
        ax2.scatter(data_x, data_y_log, color=color_b)
        ax2.plot(data_x, data_y_log, color=color_b)
        ax2.tick_params(axis='y', labelcolor=color_b)

        fig.tight_layout()
        plt.title(plot_title)
        plt.show()
        plt.close()

    @staticmethod
    def data_SVD(obj_data, q_val, singular_show_num, singular_cut_num, svd_result_root, save_file_family_name):
        dataSVD = SVDCalc(obj_data)
        nowSingVal = dataSVD.calc_svd()
        now_data_num = obj_data.shape[1]

        print(nowSingVal[:singular_show_num])
        singular_data_y = nowSingVal[:singular_show_num]
        singular_data_y_log = np.log(singular_data_y)
        singular_data_x = range(1, len(singular_data_y) + 1)

        def plot_singular_value(data_x, data_y, data_y_log):
            color_r = 'tab:red'
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("index of singular value")
            ax1.set_ylabel("singular value", color=color_r)
            ax1.scatter(data_x, data_y, color=color_r)
            ax1.plot(data_x, data_y, color=color_r)
            ax1.tick_params(axis='y', labelcolor=color_r)

            ax2 = ax1.twinx()
            color_b = 'tab:blue'
            ax2.set_ylabel("log scale singular value", color=color_b)
            ax2.scatter(data_x, data_y_log, color=color_b)
            ax2.plot(data_x, data_y_log, color=color_b)
            ax2.tick_params(axis='y', labelcolor=color_b)

            fig.tight_layout()
            plt.show()

        plot_singular_value(singular_data_x, singular_data_y, singular_data_y_log)

        bigSingVal = nowSingVal[:singular_cut_num]
        print(bigSingVal)

        print("left", dataSVD.leftVec.shape)
        print("right", dataSVD.rightVecTrans.shape)
        dataSVD.pick_meaningful_data(singular_cut_num)
        print("left", dataSVD.meanLeftVec.shape)
        print("right", dataSVD.meanRightVec.shape)

        lsv_title = save_file_family_name + " LSV plot"
        rsv_title = save_file_family_name + " RSV plot"
        dataSVD.plot_left_vec(title_data_name=save_file_family_name, plot_x_val=q_val)
        dataSVD.plot_right_vec(title_data_name=save_file_family_name)

        # skip intermediate svd result save
        # sVal_file_name = save_file_family_name + "_SingVal.dat"
        # rsv_file_name = save_file_family_name + "_RSV.dat"
        # lsv_file_name = save_file_family_name + "_LSV.dat"
        #
        # sValOutFp = open((svd_result_root + sVal_file_name), 'w')
        # rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
        # lsvOutFp = open((svd_result_root + lsv_file_name), 'w')
        #
        # dataSVD.file_output_singular_value(sValOutFp)
        # dataSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="q_A^-1", leftLabel=q_val,
        #                                                 rightLabelName="data_idx", rightLabel=list(range(now_data_num)))
        # sValOutFp.close()
        # rsvOutFp.close()
        # lsvOutFp.close()

        return dataSVD.meanLeftVec, dataSVD.meanRightVec

    @staticmethod
    def repeat_svd_rm_outlier(data_to_anal, q_val, singular_show_num, singular_cut_num, svd_result_root, save_file_family_name):
        now_anal_data = data_to_anal
        loop_idx = 0
        final_lsv = []
        final_rsv = []
        while True:
            result_mean_lsv, result_mean_rsv = SANODCalc.data_SVD(np.transpose(now_anal_data), q_val, singular_show_num, singular_cut_num, svd_result_root, save_file_family_name)
            now_loop_rm_idx = []
            for idx_rsv in range(singular_cut_num):
                now_rsv = result_mean_rsv[:, idx_rsv]
                abs_rsv = np.abs(now_rsv)
                idx_to_rm = np.where(abs_rsv > 1)[0]
                if len(now_loop_rm_idx) == 0:
                    now_loop_rm_idx = idx_to_rm
                else:
                    now_loop_rm_idx = np.union1d(now_loop_rm_idx, idx_to_rm)
            print("loop {}] rm {} idx :".format(loop_idx, len(now_loop_rm_idx)), now_loop_rm_idx)

            if len(now_loop_rm_idx) == 0:
                final_lsv = result_mean_lsv
                final_rsv = result_mean_rsv
                break
            else:
                now_data_len = now_anal_data.shape[0]
                remain_idx = np.setdiff1d(np.arange(now_data_len), now_loop_rm_idx)
                now_anal_data = now_anal_data[np.array(remain_idx), :]
                loop_idx += 1
        return final_lsv, final_rsv

    def svd_neg_and_all_signal(self, sing_val_show_num=10, num_neg_comp=2, num_whole_comp=3):
        total_signal = np.concatenate((self.neg_signal, self.pos_signal), axis=0)
        neg_tp = np.transpose(self.neg_signal)
        signal_tp = np.transpose(total_signal)
        self.total_signal_tp = signal_tp

        print("total signal shape : ", total_signal.shape)
        self.num_total_data = total_signal.shape[0]

        neg_svd = SVDCalc(neg_tp)
        whole_svd = SVDCalc(signal_tp)

        neg_sing_val = neg_svd.calc_svd()
        pos_sing_val = whole_svd.calc_svd()

        self.plot_singular_value(neg_svd, "negative singular val", sing_val_show_num)
        self.plot_singular_value(whole_svd, "whole singular val", sing_val_show_num)

        neg_svd.pick_meaningful_data(num_neg_comp)
        whole_svd.pick_meaningful_data(num_whole_comp)
        self.num_neg_comp = num_neg_comp
        self.num_whole_comp = num_whole_comp

        neg_svd.plot_left_vec(title_data_name="negative", plot_x_val=self.common_q_val)
        neg_svd.plot_right_vec()
        whole_svd.plot_left_vec_with_x_val(title_data_name="whole", plot_x_val=self.common_q_val)
        whole_svd.plot_right_vec()

        self.non_ortho_components = np.concatenate((neg_svd.meanLeftVec, whole_svd.meanLeftVec), axis=1)
        print(self.non_ortho_components.shape)

    def advanced_svd_neg_and_all_signal(self, sing_val_show_num=10, num_neg_comp=2, num_whole_comp=3):
        total_signal = np.concatenate((self.neg_signal, self.pos_signal), axis=0)
        signal_tp = np.transpose(total_signal)
        self.total_signal_tp = signal_tp

        print("total signal shape : ", total_signal.shape)
        self.num_total_data = total_signal.shape[0]

        neg_lsv, neg_rsv = self.repeat_svd_rm_outlier(self.neg_signal, self.common_q_val, sing_val_show_num, num_neg_comp, "../results/sanod/three_file_sanod/", self.save_file_family_name + "_neg_data_svd")
        whole_lsv, whole_rsv = self.repeat_svd_rm_outlier(total_signal, self.common_q_val, sing_val_show_num, num_whole_comp, "../results/sanod/three_file_sanod/", self.save_file_family_name + "_whole_data_svd")

        self.num_neg_comp = num_neg_comp
        self.num_whole_comp = num_whole_comp

        self.non_ortho_components = np.concatenate((neg_lsv, whole_lsv), axis=1)
        print(self.non_ortho_components.shape)

    @staticmethod
    def plot_multiple_comps(comp_data, x_axis_data, graph_title, labels, is_separate=True, same_y_scale=True):
        num_comp = comp_data.shape[1]
        conct_fig_v_size = 2 * num_comp
        if is_separate:
            plt.figure(figsize=(6, conct_fig_v_size))
            y_max = np.max(comp_data)
            y_min = np.min(comp_data)
            margin = (y_max - y_min) / 20
            for plot_idx in range(num_comp):
                plt.subplot(num_comp, 1, plot_idx + 1)
                if plot_idx == 0:
                    plt.title(graph_title)
                plt.plot(x_axis_data, comp_data[:, plot_idx], label=labels[plot_idx])
                plt.axhline(y=0, color='gray', ls='--', alpha=0.6)
                if same_y_scale:
                    plt.ylim((y_min - margin, y_max + margin))
                plt.legend()
            plt.xlabel("q")
            plt.tight_layout()
            plt.show()
            plt.close()

        else:
            for o_idx in range(num_comp):
                plt.plot(x_axis_data, comp_data[:, o_idx], label=labels[o_idx])

            plt.xlabel("q")
            plt.title(graph_title)
            plt.legend()
            plt.show()
            plt.close()

    def plot_non_ortho_comps(self):
        self.num_non_ortho_comp = self.non_ortho_components.shape[1]
        c_label = ["C_" + str(idx + 1) for idx in range(self.num_non_ortho_comp)]
        print(c_label)
        self.plot_multiple_comps(self.non_ortho_components, self.common_q_val, graph_title="spectral components", labels=c_label)

        q_orthonorm, r_uptri = mgs(self.non_ortho_components)
        print("q shape", q_orthonorm.shape)
        print("r shape", r_uptri.shape)

        num_ortho_norm_comps = q_orthonorm.shape[1]
        o_label = ["O_" + str(idx + 1) for idx in range(num_ortho_norm_comps)]
        print(o_label)
        self.plot_multiple_comps(q_orthonorm, self.common_q_val, graph_title="orthonormalized components - same y scale",
                                 labels=o_label, same_y_scale=True)
        self.plot_multiple_comps(q_orthonorm, self.common_q_val, graph_title="orthonormalized components",
                                 labels=o_label, same_y_scale=False)

        time_delay = np.arange(self.num_total_data)

        alpha_weight = np.zeros((self.num_total_data, num_ortho_norm_comps))
        for delay_idx in range(self.num_total_data):
            for comp_idx in range(num_ortho_norm_comps):
                temp = np.dot(self.total_signal_tp[:, delay_idx], q_orthonorm[:, comp_idx]) / r_uptri[comp_idx][comp_idx]
                alpha_weight[delay_idx][comp_idx] = temp

        print(alpha_weight)
        self.alpha_weights = alpha_weight


        def plot_chronogram(alpha_data, time_delay_ps, graph_title, labels, is_separate=True, same_y_scale=True):
            # x_ticks = [-5, 0, 5, 10, 20, 50, 100, 1000]
            # x_ticks = [-5, -1, 0, 1, 10, 100, 1000]
            # str_ticks = []
            # for each_time_val in x_ticks:
            #     str_ticks.append(str(each_time_val))

            num_comp = alpha_data.shape[1]
            conct_fig_v_size = 2 * num_comp
            if is_separate:
                plt.figure(figsize=(6, conct_fig_v_size))
                y_max = np.max(alpha_data)
                y_min = np.min(alpha_data)
                margin = (y_max - y_min) / 20
                for plot_idx in range(num_comp):
                    plt.subplot(num_comp, 1, plot_idx + 1)
                    if plot_idx == 0:
                        plt.title(graph_title)
                    plt.plot(time_delay_ps, alpha_data[:, plot_idx], '.-', label=labels[plot_idx])
                    plt.axhline(y=0, color='gray', ls='-', alpha=0.6)
                    if same_y_scale:
                        plt.ylim((y_min - margin, y_max + margin))
                    # plt.xlim(-5, 5e3)
                    # plt.xscale('symlog', linthresh=1, linscale=1.5)
                    # plt.xticks(ticks=x_ticks, labels=str_ticks)
                    plt.grid(True)
                    plt.legend()
                plt.xlabel("shot idx")
                plt.tight_layout()
                plt.show()

            else:
                for o_idx in range(num_comp):
                    plt.plot(time_delay_ps, alpha_data[:, o_idx], label=labels[o_idx])

                plt.xlabel("shot idx")
                plt.title(graph_title)
                # plt.xlim(-5, 5e3)
                # plt.xscale('symlog', linthresh=1, linscale=1.5)
                # plt.xticks(ticks=x_ticks, labels=str_ticks)
                plt.grid(True)
                plt.legend()
                plt.show()

        alpha_label = ["a_" + str(idx + 1) for idx in range(num_ortho_norm_comps)]
        print(alpha_label)
        plot_chronogram(alpha_weight, time_delay_ps=time_delay, graph_title="chronogram (alpha for C_N) - same y scale",
                        labels=alpha_label, is_separate=True, same_y_scale=True)
        plot_chronogram(alpha_weight, time_delay_ps=time_delay, graph_title="chronogram (alpha for C_N)",
                        labels=alpha_label, is_separate=True, same_y_scale=False)

    def artifact_filtering(self):
        reconstructed = []
        total_signal = np.transpose(self.total_signal_tp)
        for delay_idx in range(self.num_total_data):
            delay_data = copy.deepcopy(total_signal[delay_idx])
            # remove artifact from original data
            for arti_idx in range(self.num_neg_comp):
                delay_data -= self.non_ortho_components[:,arti_idx] * self.alpha_weights[delay_idx][arti_idx]
            reconstructed.append(delay_data)

        self.reconstructed_data = np.array(reconstructed)

        if self.do_file_save:
            np.save('../results/sanod/filterd_{}.npy'.format(self.save_file_family_name), np.transpose(self.reconstructed_data))

        return self.reconstructed_data

    def compare_filter_plot(self, graph_per_fig=10, graph_title="", time_delay_ps=None):
        # TODO : update time_delay_ps
        total_signal_before = np.transpose(self.total_signal_tp)
        y_max = np.max([self.total_signal_tp, self.reconstructed_data])
        y_min = np.min([self.total_signal_tp, self.reconstructed_data])
        margin = (y_max - y_min) / 20

        for delay_idx in range(self.num_total_data):
            if (delay_idx % graph_per_fig) == 0:
                plt.figure(figsize=(6, 18))
            plt.subplot(graph_per_fig, 1, (delay_idx % graph_per_fig) + 1)
            if (delay_idx % graph_per_fig) == 0:
                plt.title(graph_title)
            plt.plot()
            plt.plot(self.common_q_val, self.total_signal_tp[delay_idx], label=str(time_delay_ps[delay_idx]) + "ps", color='k')
            plt.plot(self.common_q_val, self.reconstructed_data[delay_idx], label="filtered " + str(time_delay_ps[delay_idx]) + "ps", color='r')
            plt.axhline(y=0, color='gray', ls='--', alpha=0.6)
            plt.ylim((y_min - margin, y_max + margin))
            plt.legend()
            if ((delay_idx + 1) % graph_per_fig) == 0:
                plt.xlabel("q")
                plt.show()
        if (self.num_total_data % graph_per_fig) != 0:
            plt.xlabel("q")
            plt.show()



