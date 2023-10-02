from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, rankdata
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import matplotlib.pyplot as plt

# Code adapted from https://github.com/flowersteam/rl_stats
#


def run_permutation_test(all_data, n1, n2):
    np.random.shuffle(all_data)
    data_a = all_data[:n1]
    data_b = all_data[-n2:]
    return data_a.mean() - data_b.mean()


def compute_central_tendency_and_error(id_central, id_error, sample):
    try:
        id_error = int(id_error)
    except Exception:
        pass

    if id_central == "mean":
        central = np.nanmean(sample, axis=1)
    elif id_central == "median":
        central = np.nanmedian(sample, axis=1)
    else:
        raise NotImplementedError

    if isinstance(id_error, int):
        low = np.nanpercentile(sample, q=int((100 - id_error) / 2), axis=1)
        high = np.nanpercentile(sample, q=int(100 - (100 - id_error) / 2), axis=1)
    elif id_error == "std":
        low = central - np.nanstd(sample, axis=1)
        high = central + np.nanstd(sample, axis=1)
    elif id_error == "sem":
        low = central - np.nanstd(sample, axis=1) / np.sqrt(sample.shape[0])
        high = central + np.nanstd(sample, axis=1) / np.sqrt(sample.shape[0])
    else:
        raise NotImplementedError

    return central, low, high


class Test(ABC):
    def plot(
        self,
        data1,  # array of performance of dimension (n_steps, n_seeds) for alg 1
        data2,  # array of performance of dimension (n_steps, n_seeds) for alg 2
        point_every=1,  # evaluation frequency, one datapoint every X steps/episodes
        confidence_level=0.01,  # confidence level alpha of the test
        id_central="median",  # id of the central tendency ('mean' or 'median')
        id_error=80,  # id of the error areas ('std', 'sem', or percentiles in ]0, 100]
        legends="alg 1/alg 2",  # labels of the two input vectors
        xlabel="training steps",  # label of the x axis
        save=True,  # save in ./plot.png if True
        downsampling_fact=5,  # factor of downsampling on the x-axis for visualization purpose (increase for smoother plots)
    ):
        assert (
            data1.ndim == 2
        ), "data should be an array of dimension 2 (n_steps, n_seeds)"
        assert (
            data2.ndim == 2
        ), "data should be an array of dimension 2 (n_steps, n_seeds)"

        nb_steps = max(data1.shape[0], data2.shape[0])
        steps = [0]
        while len(steps) < nb_steps:
            steps.append(steps[-1] + point_every)
        steps = np.array(steps)
        if steps is not None:
            assert (
                steps.size == nb_steps
            ), "x should be of the size of the longest data array"

        sample_size1 = data1.shape[1]
        sample_size2 = data2.shape[1]

        # downsample for visualization purpose
        sub_steps = np.arange(0, nb_steps, downsampling_fact)
        steps = steps[sub_steps]
        nb_steps = steps.size

        # handle arrays of different lengths by padding with nans
        sample1 = np.zeros([nb_steps, sample_size1])
        sample1.fill(np.nan)
        sample2 = np.zeros([nb_steps, sample_size2])
        sample2.fill(np.nan)
        sub_steps1 = sub_steps[: data1.shape[0] // downsampling_fact]
        sub_steps2 = sub_steps[: data2.shape[0] // downsampling_fact]
        sample1[: data1[sub_steps1, :].shape[0], :] = data1[sub_steps1, :]
        sample2[: data2[sub_steps2, :].shape[0], :] = data2[sub_steps2, :]

        # test
        sign_diff = np.zeros([len(steps)])
        for i in range(len(steps)):
            sign_diff[i] = self.run_test(
                sample1[i, :].squeeze(), sample2[i, :].squeeze(), alpha=confidence_level
            )

        central1, low1, high1 = compute_central_tendency_and_error(
            id_central, id_error, sample1
        )
        central2, low2, high2 = compute_central_tendency_and_error(
            id_central, id_error, sample2
        )

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        lab1 = plt.xlabel(xlabel)
        lab2 = plt.ylabel("performance")

        plt.plot(steps, central1, linewidth=10)
        plt.plot(steps, central2, linewidth=10)
        plt.fill_between(steps, low1, high1, alpha=0.3)
        plt.fill_between(steps, low2, high2, alpha=0.3)
        splitted = legends.split("/")
        leg = ax.legend((splitted[0], splitted[1]), frameon=False)

        # plot significative difference as dots
        idx = np.argwhere(sign_diff == 1)
        y = max(np.nanmax(high1), np.nanmax(high2))
        plt.scatter(
            steps[idx], y * 1.05 * np.ones([idx.size]), s=100, c="k", marker="o"
        )

        # style
        for line in leg.get_lines():
            line.set_linewidth(10.0)
        ax.spines["top"].set_linewidth(5)
        ax.spines["right"].set_linewidth(5)
        ax.spines["bottom"].set_linewidth(5)
        ax.spines["left"].set_linewidth(5)

        if save:
            plt.savefig(
                "./plot.png",
                bbox_extra_artists=(leg, lab1, lab2),
                bbox_inches="tight",
                dpi=100,
            )

        plt.show()

    @abstractmethod
    def run_test(self, data1, data2, alpha=0.05):
        """
        Compute tests comparing data1 and data2 with confidence level alpha
        :param data1: (np.ndarray) sample 1
        :param data2: (np.ndarray) sample 2
        :param alpha: (float) confidence level of the test
        :return: (bool) if True, the null hypothesis is rejected
        """


class BootstrapTest(Test):
    def run_test(self, data1, data2, alpha=0.05):
        assert alpha < 1 and alpha > 0, "alpha should be between 0 and 1"
        res = bs.bootstrap_ab(
            data1,
            data2,
            bs_stats.mean,
            bs_compare.difference,
            alpha=alpha,
            num_iterations=1000,
        )
        rejection = np.sign(res.upper_bound) == np.sign(res.lower_bound)
        return rejection


class TTest(Test):
    def run_test(self, data1, data2, alpha=0.05):
        _, p = ttest_ind(data1, data2, equal_var=True)
        return p < alpha


class WelchTTest(Test):
    """Welch t-test (recommended)"""

    def run_test(self, data1, data2, alpha=0.05):
        _, p = ttest_ind(data1, data2, equal_var=False)
        return p < alpha


class MannWhitneyTest(Test):
    def run_test(self, data1, data2, alpha=0.05):
        _, p = mannwhitneyu(data1, data2, alternative="two-sided")
        return p < alpha


class RankedTTest(Test):
    def run_test(self, data1, data2, alpha=0.05):
        n1, n2 = data1.size, data2.size
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        ranks = rankdata(all_data)
        ranks1 = ranks[:n1]
        ranks2 = ranks[n1 : n1 + n2]
        assert ranks2.size == n2
        _, p = ttest_ind(ranks1, ranks2, equal_var=True)
        return p < alpha


class PermutationTest(Test):
    def run_test(self, data1, data2, alpha=0.05):
        n1, n2 = data1.size, data2.size
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        delta = np.abs(data1.mean() - data2.mean())
        num_samples = 1000
        estimates = []
        for _ in range(num_samples):
            estimates.append(run_permutation_test(all_data.copy(), n1, n2))
        estimates = np.abs(np.array(estimates))
        diff_count = len(np.where(estimates <= delta)[0])
        return (1.0 - (float(diff_count) / float(num_samples))) < alpha


# if __name__ == '__main__':
#     import argparse
#     import sys
#     data1 = np.loadtxt('./data/sac_hc_all_perfs.txt')
#     data2 = np.loadtxt('./data/td3_hc_all_perfs.txt')
#     sample_size = 20
#     data1 = data1[:, np.random.randint(0, data1.shape[1], sample_size)]
#     data2 = data2[:, np.random.randint(0, data1.shape[1], sample_size)]
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     add = parser.add_argument
#     add('--data1', type=str, default=data1, help='path to text file containing array of performance of dimension (n_steps, n_seeds) for alg 1. Can also receive the array '
#                                                  'directly.')
#     add('--data2', type=str, default=data2, help='path to text file containing array of performance of dimension (n_steps, n_seeds) for alg 2. Can also receive the array '
#                                              'directly.')
#     add('--point_every', type=int, default=1, help='evaluation frequency, one datapoint every X steps/episodes')
#     add('--test_id', type=str, default="Welch t-test", help="choose in [t-test, Welch t-test, Mann-Whitney, Ranked t-test, bootstrap, permutation], welch recommended (see paper)")
#     add('--confidence_level', type=float, default=0.01, help='confidence level alpha of the test')
#     add('--id_central', type=str, default='median', help="id of the central tendency ('mean' or 'median')")
#     add('--id_error', default=80, help="id of the error areas ('std', 'sem', or percentiles in ]0, 100]")
#     add('--legends', type=str, default='SAC/TD3', help='labels of the two input vectors "legend1/legend2"')
#     add('--xlabel', type=str, default='training episodes', help='label of the x axis, usually episodes or steps')
#     add('--save', type=bool, default=True, help='save in ./plot.png if True')
#     add('--downsampling_fact', type=int, default=5, help='factor of downsampling on the x-axis for visualization purpose (increase for smoother plots)')
#     kwargs = vars(parser.parse_args())
#     run_test_and_plot(**kwargs)
