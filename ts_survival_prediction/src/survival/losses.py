import torch
import torch.nn as nn

class CoxLoss(nn.Module):
    """
    The Cox proportional hazards loss.
    Code adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/losses.py
    """
    def __init__(self):
        super().__init__()

    def __call__(self, logits, times, censorships):
        # return partial_ll_loss(lrisks = logits, survival_times=times, event_indicators=(1-censorships).float())
        """
        logits: log risks, B x 1
        times: time bin, B x 1
        event_indicators: event indicator, B x 1
        """    
    
        event_indicators=(1-censorships).float()
        num_uncensored = torch.sum(event_indicators, 0)

        if num_uncensored.item() == 0:
            loss = torch.sum(logits) * 0
            return loss, {'loss': loss.item()}
        
        times = times.squeeze(1)
        event_indicators = event_indicators.squeeze(1)
        logits = logits.squeeze(1)

        sindex = torch.argsort(-times)
        times = times[sindex]
        event_indicators = event_indicators[sindex]
        logits = logits[sindex]

        log_risk_stable = torch.logcumsumexp(logits, 0)

        likelihood = logits - log_risk_stable
        uncensored_likelihood = likelihood * event_indicators
        logL = -torch.sum(uncensored_likelihood)
        # negative average log-likelihood
        loss = logL / num_uncensored
        return loss, {'loss': loss.item()}