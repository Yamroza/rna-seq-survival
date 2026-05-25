from metrics.metrics import concordance_index_censored, concordance_index_ipcw
from metrics.util import Surv

def compute_survival_metrics(all_censorships, all_event_times, all_risk_scores, survival_info_train):
    """ Compute the concordance index and the IPCW concordance index. """
    # Compute the concordance index
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    # Compute the IPCW concordance index if training survival info is provided
    c_index_ipcw = 0.
    if survival_info_train:
        structured_survival_data_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
        structured_survival_data_train = Surv.from_arrays(event=(1-survival_info_train['censorship']).astype(bool), time=survival_info_train['time'])

        c_index_ipcw = concordance_index_ipcw(structured_survival_data_train, structured_survival_data_test, estimate=all_risk_scores)[0]

    return c_index, c_index_ipcw


class LoggingMeter(object):
    """Computes and stores the average and current value, adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/utils.py"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count