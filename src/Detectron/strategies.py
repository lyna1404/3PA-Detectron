from .record import DetectronRecordsManager
import numpy as np
import scipy.stats as stats

def ecdf(x):
    """
    Compute the empirical cumulative distribution function
    :param x: array of 1-D numerical data
    :return: a function that takes a value and returns the probability that
        a random sample from x is less than or equal to that value
    """
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result

class DetectronStrategy:
    @staticmethod
    def execute(self, calibration_records : DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        pass
    def evaluate(self, calibration_record: DetectronRecordsManager,
                        test_record: DetectronRecordsManager,
                        alpha=0.05,
                        max_ensemble_size=None):
        pass

class DisagreementStrategy(DetectronStrategy):
    def execute(self, calibration_records : DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        cal_counts = calibration_records.counts()
        test_count = test_records.counts()[0]
        cdf = ecdf(cal_counts)
        p_value = cdf(test_count)

        test_statistic={test_count}
        baseline_mean = cal_counts.mean()
        baseline_std = cal_counts.std()

        results = {'p_value':p_value, 'test_statistic': test_statistic, 'baseline_mean': baseline_mean, 'baseline_std': baseline_std, 'shift_indicator': (p_value < significance_level)}
        return results
    
    def evaluate(self, calibration_record: DetectronRecordsManager,
                        test_record: DetectronRecordsManager,
                        alpha=0.05,
                        max_ensemble_size=None):
        """
        Compute the discovery power of the detectron algorithm.
        :param calibration_record: (XGBDetectronRecord) the results of the calibration run
        :param test_record: (XGBDetectronRecord) the results of the test run
        :param alpha: (0.05) the significance level
        :param max_ensemble_size: (None) the maximum number of models in the ensemble to consider.
            If None, all models are considered.
        :return: the discovery power
        """
        cal_counts = calibration_record.counts(max_ensemble_size=max_ensemble_size)
        test_counts = test_record.counts(max_ensemble_size=max_ensemble_size)
        N = calibration_record.sample_size
        assert N == test_record.sample_size, 'The sample sizes of the calibration and test runs must be the same'

        fpr = (cal_counts <= np.arange(0, N + 2)[:, None]).mean(1)
        tpr = (test_counts <= np.arange(0, N + 2)[:, None]).mean(1)

        quantile = np.quantile(cal_counts, alpha)
        tpr_low = (test_counts < quantile).mean()
        tpr_high = (test_counts <= quantile).mean()

        fpr_low = (cal_counts < quantile).mean()
        fpr_high = (cal_counts <= quantile).mean()

        if fpr_high == fpr_low:
            tpr_at_alpha = tpr_high
        else:  # use linear interpolation if there is no threshold at alpha
            tpr_at_alpha = (tpr_high - tpr_low) / (fpr_high - fpr_low) * (alpha - fpr_low) + tpr_low

        return dict(power=tpr_at_alpha, auc=np.trapz(tpr, fpr), N=N)

class DisagreementStrategy_MW(DetectronStrategy):
    def execute(self, calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        # Retrieve count data from both calibration and test records
        cal_counts = calibration_records.counts()
        test_counts = test_records.counts()
        
        # Combine both groups for ranking
        combined_counts = np.concatenate((cal_counts, test_counts))
        ranks = stats.rankdata(combined_counts)
        
        # Separate the ranks back into two groups
        cal_ranks = ranks[:len(cal_counts)]
        test_ranks = ranks[len(cal_counts):]
        
        # Calculate mean and standard deviation of ranks for both groups
        cal_rank_mean = np.mean(cal_ranks)
        cal_rank_std = np.std(cal_ranks)
        test_rank_mean = np.mean(test_ranks)
        test_rank_std = np.std(test_ranks)

        # Perform the Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(cal_counts, test_counts, alternative='two-sided')

        # Print rank statistics and test results
        print(f"Calibration Ranks - Mean: {cal_rank_mean:.2f}, Std: {cal_rank_std:.2f}")
        print(f"Test Ranks - Mean: {test_rank_mean:.2f}, Std: {test_rank_std:.2f}")
        print(f"Mann-Whitney U Statistic: {u_statistic}")
        print(f"P-value: {p_value:.3f}")

        # Results dictionary including rank statistics
        results = {
            'p_value': p_value,
            'test_statistic': u_statistic,
            'cal_rank_mean': cal_rank_mean,
            'cal_rank_std': cal_rank_std,
            'test_rank_mean': test_rank_mean,
            'test_rank_std': test_rank_std,
            'shift_indicator': (p_value < significance_level)
        }

        return results

class DisagreementStrategy_KS(DetectronStrategy):
    def execute(self, calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        # Retrieve count data from both calibration and test records
        cal_counts = calibration_records.counts()
        test_counts = test_records.counts()
        
        # Perform the Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(cal_counts, test_counts)

        # Calculate statistics for interpretation
        cal_mean = cal_counts.mean()
        cal_std = cal_counts.std()
        test_mean = test_counts.mean()
        test_std = test_counts.std()

        # Print test results and distribution statistics
        print(f"Calibration Data - Mean: {cal_mean:.2f}, Std: {cal_std:.2f}")
        print(f"Test Data - Mean: {test_mean:.2f}, Std: {test_std:.2f}")
        print(f"KS Statistic: {ks_statistic:.3f}")
        print(f"P-value: {p_value:.3f}")

        # Results dictionary including KS test results and distribution statistics
        results = {
            'p_value': p_value,
            'ks_statistic': ks_statistic,
            'cal_mean': cal_mean,
            'cal_std': cal_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'shift_indicator': (p_value < significance_level)
        }

        return results

class DisagreementStrategy_quantile(DetectronStrategy):
    def execute(self, calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        # Retrieve count data from both calibration and test records
        cal_rejected_counts = calibration_records.rejected_counts()
        test_rejected_count = test_records.rejected_counts().mean()
        
        quantile = calibration_records.rejected_count_quantile(1-significance_level, None)
        shift_indicator = test_rejected_count > quantile
        # Results dictionary including KS test results and distribution statistics
        results = {
            'quantile': quantile,
            'test_statistic' : test_rejected_count,
            'shift_indicator': shift_indicator
        }

        return results