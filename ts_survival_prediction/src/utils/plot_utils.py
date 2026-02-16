import numpy as np

def update_risk_dict(current, new):
    """ Add new values to dictionary containing lists. """
    for item, value in new.items():
        if item not in current.keys(): 
            # If this is the first time we see this item, create a new list
            current[item] = value
        else:
            # Otherwise, append the new values to the existing list
            current[item] = np.concatenate((current[item], value), axis=0)
    
    return current

def rebuild_dict(cases_list):
    """ Helper function creating a dictionary of lists. """
    case_ids = [case[0] for case in cases_list]
    risk_scores = [case[1] for case in cases_list]
    time = [case[2] for case in cases_list]
    events = [case[3] for case in cases_list]
    
    return {
        "case_ids": case_ids,
        "Risk scores": risk_scores,
        "Time": time,
        "Event": events
    }