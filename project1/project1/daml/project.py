# project.py: Methods relevant specifically to DAML project 1.
#
# Author: Andreas Sogaard <andreas.sogaard@ed.ac.uk>
# ------------------------------------------------------------------------------

# Import(s)
import numpy as np
from sklearn.metrics import accuracy_score


def save_submission (predictions, UUN):
    """
    Save test dataset predictions to a CSV file for submission.
    
    Arguments:
        ...
        
    Raises:
        AssertionError, if the arguments are not supported.
    """
        
    # Initial variable definitions
    NUM_TEST = 15000  # Number of test examples
    
    # Check(s) -- UUN
    assert isinstance(UUN, int), \
        "UUN should be an int; is {}.".format(type(UUN))
    assert UUN < 2100000, \
        f"UUN should be a valid, seven-digit number; {UUN} seems too big."
    assert UUN > 1000000, \
        f"UUN should be a valid, seven-digit number; {UUN} seems too small."

    # Check(s) -- predictions
    assert isinstance(predictions, np.ndarray), \
        "Predictions should be in the form of an numpy array; is {}.".format(type(predictions))
    p = predictions.squeeze()
    assert not np.any(np.isnan(predictions)), \
        "Found NaNs in predictions."
    assert p.shape[0] == NUM_TEST, \
        "Predictions should be given for all test examples ({}); is {}".format(NUM_TEST, p.shape[0])
    assert len(p.shape) <= 2, \
        "Predictions should at most have two dimensions; has {}.".format(len(predictions.shape))
    assert np.all(p >= 0), \
        "Predictions should be non-negative; found min {}".format(p.min())

    fmt    = '%.18e'
    labels = False
    if p.dtype.kind == 'i':
        print("Got integer predictions; assuming class labels.")
        fmt    = '%d'
        labels = True
    elif p.dtype.kind == 'f':
        print("Got floating-point predictions; assuming class probabilities.")
        assert np.all(p <=1), \
            "Predicted probabilities should not be greater than 1; found max {}".format(p.max())
        pass
    
    # Single-column
    if len(p.shape) == 1: 
        if not labels:
            print("Assuming binary classification of background (type 0; target == 0) vs. signals (type 1 or 2; target == 1), resp.")
            pass
    
    # Multi-column
    elif len(p.shape) == 2:
        assert not labels, \
            "Got multi-column array with predicted class labels; please specify as single column."
        assert p.shape[1] in [2,3], \
            "Predictions should at most be among 3 classes; is {}.".format(predictions.shape[1])
        assert np.all(p >= 0), \
            "Found probabilities < 0."
        assert np.all(p <= 1), \
            "Found probabilities > 1."
        assert np.all(np.abs(p.sum(axis=1) - 1) < 1.0E-04), \
            "Not all rows (probabilities) add to 1."
        if p.shape[1] == 2:
            print("Got an array with two columns. Assuming background (type 0) and signal (type 1 or 2) probabilities, and taking only second column.")
            p = p[:,1]
        else:
            print("Got an array with three columns. Assuming background (type 0), signal 1, and signal 2 probabilities.")
            pass
        pass
    
    # Save the predictions to file
    suffix = 'binary' if ((not labels and len(p.shape) == 1) or (labels and p.max() == 1)) else 'multiclass'
    filename = f"preds-s{UUN}-{suffix}.csv"
    
    print(f"Saving predictions to file: {filename}")
    np.savetxt(filename, p, fmt=fmt)
    return


def evaluate (preds, signal, weight=None):
    """
    Method used to evaluate predictions based on class prediction accuracy.
    
    Arguments:
        preds: Numpy array of predictions, either class predictions or 
            probabilties.
        signal: Numpy array of class labels (0, 1, or 2)
        weight: Numpy array of sample weights, used when calculating the 
            accuracy. If None, use weights of all ones
    """
    if weight is None:
        weight = np.ones_like(signal, dtype=np.float32)
        pass
    
    isInt = np.all(preds == preds.astype(int))
    if not isInt:
        if len(preds.shape) > 1:
            preds = np.argmax(preds, axis=1)
        else:
            preds = (preds > 0.5).astype(int)
            pass
        pass
    
    # Choose appropriate signal labels
    sig   = (signal > 0)  if (preds.max() == 1) else signal
    
    return accuracy_score(sig, preds, sample_weight=weight)