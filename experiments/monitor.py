import numpy as np
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops


class CorrelationMonitor1d(object):
    """
    Compute correlation between 1d predictions

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        # this will be timeseries of predictions
        # for each trial
        # braindecode functions expect classes x time predictions
        # so add fake class dimension and remove it again
        preds_2d = [p[:, None] for p in all_preds]
        preds_per_trial = compute_preds_per_trial_from_crops(preds_2d,
                                                          self.input_time_length,
                                                          dataset.X)
        preds_per_trial = [p[0] for p in preds_per_trial]
        pred_timeseries = np.concatenate(preds_per_trial, axis=0)
        ys_2d = [y[:, None] for y in all_targets]
        targets_per_trial = compute_preds_per_trial_from_crops(ys_2d,
                                                            self.input_time_length,
                                                            dataset.X)
        targets_per_trial = [t[0] for t in targets_per_trial]
        target_timeseries = np.concatenate(targets_per_trial, axis=0)

        corr = np.corrcoef(target_timeseries, pred_timeseries)[0, 1]
        key = setname + '_corr'

        return {key: float(corr)}