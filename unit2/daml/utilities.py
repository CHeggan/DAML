# utilities.py: Collection of utility methods common to several DAML CPs.
#
# Author: Andreas Sogaard <andreas.sogaard@ed.ac.uk>
#
# In this file you will find several convenient utility functions, that can 
# hopefully reduce the time spent on reading code documentation, and increase 
# the time spent on doing cool machine learning stuff. These methods are 
# provided with no guarantees. If you experience any problems using these, let 
# me know and I will try to help out; but you should be able to write code with 
# similar functionality yourself if necessary.
# ------------------------------------------------------------------------------

# Import(s)
import torch as _torch
import numpy as _np
import sklearn as _sklearn
import matplotlib as _matplotlib
import tensorflow as _tf
import tensorflow.python.keras as _keras

# Suppress unnecessary ConvergenceWarnings and DeprecationWarnings
import warnings as _warnings
_warnings.filterwarnings(action='ignore', category=_sklearn.exceptions.ConvergenceWarning)
_warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# Suppress TensorFlow-contrib/deprecations warnings
if type(_tf.contrib) != type(_tf): 
    _tf.contrib._warning = None
    pass
import tensorflow.python.util.deprecation as _deprecation
_deprecation._PRINT_DEPRECATION_WARNINGS = False

# Local import(s)
from . import _internal


def make_reproducible (seed):
    """
    Make sure that all results are reproducible. Well, mostly reproducible. If 
    you're using multi-threading, e.g. which performing hyper-parameter 
    optimisation, this will *not* hold. But it should mean that, in most cases, 
    when we re-run you notebook, we will see the same results as you did.

    Arguments:
        seed: The seed value to be used for all random number generators. Must
            be positive integer in the range [1, 2**32 - 1], e.g. your UUN
            (s[XXXXXXX] <- these seven digits).
            
    Raises:
        AssertionError, if the provided seed is invalid.
    """

    # Check(s)
    assert isinstance(seed, int) and seed > 0 and seed < 2**32, \
        "Please specify a positive integer seed in the range [1, 2**32 - 1], e.g. your UUN (s[XXXXXXX] <- these seven digits)."

    print(f"Making reproducible with seed {seed}. Please not that running in parallel (e.g. by setting `n_jobs > 1` in certain scikit-learn functions) breaks reproducibility.")
          
    # Import(s)
    import os
    import random as rn
    
    # Set seed variables
    _np.random.seed(seed)
    rn.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    _tf.compat.v1.set_random_seed(seed)
    _torch.manual_seed(seed)

    # Switch off multi-threading for TensorFlow
    from tensorflow.python.keras import backend as K
    config = _tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                       inter_op_parallelism_threads=1)
    sess = _tf.compat.v1.Session(graph=_tf.compat.v1.get_default_graph(), config=config)
    K.set_session(sess)
    return


def get_cmap (nb_classes):
    """
    Get a pyplot colormap using the first `nb_classes` in the current colour cycle.
    
    Arguments:
        nb_classes: Integer, number of classes for which to get a colour.
        
    Returns:
        pyplot colormap, which can be accessed as `cmap(ix)` for ix in 
            [0, nb_classes - 1] to yield a unique, consisten colour for each 
            class.
    """
    colours = [d['color'] for d in _matplotlib.rcParams['axes.prop_cycle'][:nb_classes]]
    cmap = _matplotlib.colors.ListedColormap(colours)
    return cmap


def get_confidence (probs, axis=-1):
    """
    Return the "confidence" in a classification. This is measured as the 
    relative difference between complete certainty (most likely class has 
    probability = 1, all other classes have probability = 0; in which case the 
    confidence is 1) and complete uncertainty (most likely class, and therefore 
    all classes, have probability = 1/len(probs); in which case the confidence 
    is 0).
    
    Arguments:
        probs: List or numpy.array, containing classification probabilities.
        axis: array axis along which the maximal probability should be found.
        
    Returns:
        numpy.array with one dimension less than `probs`, containing the 
            confidence in the input classification probabilities.
    """
    probs = _np.asarray(probs)
    nb_classes = probs.shape[-1]
    pmax       = probs.max(axis=axis)
    confidence = (pmax * nb_classes - 1) / (nb_classes - 1.)
    return confidence



# ==============================================================================
# Decision trees
# ------------------------------------------------------------------------------

def get_node_colour (node):
    """
    Parse the label of a pydot graph node, such as those used when displaying
    decision tree logic, to get the class fractions on this node. Based on these,
    computed the colour for the node, based on the current colormap, as well as
    the opacity, based on the confidence in the maximal class label. That is, if
    all classes have equal probability, return white; if class i has probability
    1, return the i'th colour of the current colormap.

    Arguments:
        node: pydot graph node, for which an updated colour should be found.

    Raises:
        AssertionError: If the node has a label, but the text is not as expected,
            i.e. it does not contain a 'value = ' string

    Returns:
        str specifying the 8-digit colour hex enclosed in quotation marks, as
            required by matplotlib. Returns None, indicating no change of colour,
            if the node does not have a label.
    """

    # Check(s)
    if node.get_label() is None:
        return

    # Get label line specifying node value
    lines = list(filter(lambda s: 'value =' in s, node.get_label().split('<br/>')))
    assert len(lines) == 1

    # Get class probability
    counts = _np.asarray(list(map(int, lines[0].replace('value = [', '').replace(']', '').split(','))))
    probs  = counts / float(counts.sum())

    # Get number of classes
    nb_classes = len(probs)

    # Get colormap
    cmap = get_cmap(nb_classes)

    # Get node colour
    colour = cmap(probs.argmax()) # RGBA tuple

    # Get node confidence
    confidence = get_confidence(probs)
    colour = colour[:3] + (confidence,)

    return _matplotlib.colors.to_hex(colour, keep_alpha=True)


def update_graph_colours (graph):
    """
    Update the colour of each node on a pydot graph, using the `get_node_colour`
    method.

    Arguments:
        graph: pydot graph to be re-coloured.

    Returns:
        Same graph, but with re-coloured nodes.
    """
    for node in graph.get_node_list():
        colour = get_node_colour(node)
        if colour is None:
            continue
        node.set_fillcolor('"{}"'.format(colour))
        pass
    return graph



# ==============================================================================
# Neural network training
# ------------------------------------------------------------------------------


def fit (model, X, y, validation=None, quiet=False, **fit_kwargs):
    """
    Convenience method for fitting both scikit-learn and Keras classifiers. In 
    the internal.utilities module, the internal method infers the type of `model`
    and runs the appropriate wrapper around the fitting function for this class.
    If `model` is a scikit-learn classifier, the method decorates it with the 
    attributes `history_loss` and `history_acc`; if `model` is a Keras model, 
    the loss history can be accessed from the `model.history` attribute, set by 
    the Keras fit method.
    
    If you want, you can see the internal method(s) for implementation details.
    
    Arguments:
        model: Instance of scikit-learn classifier, e.g. 
            sklearn.neural_network.MLPClassifier, or keras.models.Model, 
            classifier model to be fitted. The model is fitted in-place, and so 
            it is not necessary to capture the output of the method.
        X: numpy.array, of shape (nb_samples, nb_features), features to be used 
            in classification.
        y: numpy.array, of shape either (nb_samples,) or (nb_samples, nb_classes),
            class labels to be used for classification.
        validation: either float, fraction of data `X` and `y` to be used for 
            validation; or tuple `(X_val, y_val)` of numpy.arrays with same 
            dimensions as `X` and `y`, respectively, to be used for validation.
        quiet: Boolean, whether to suppress logging output during fitting.
        fit_kwargs: Dictionary, specified as keyword arguments in the function 
            call, which is passed as keyword arguments to the class-specific 
            fitting method. In the case of a Keras model, this can be used as 
            e.g.
            ```
            fit(model, X, y, epochs=50, batch_size=32, shuffle=True)
            ```
            such that, in this function call,
            ```
            fit_kwargs == dict(epochs=50, batch_size=32, shuffle=True)
            ```
    Raises:
        AssertionError, if any of the checks fail.
    
    Returns:
        Fitted model, possibly with modified attributes to log the training 
        history (see above).
    """
    return _internal.utilities._fit (model, X, y, validation=validation, quiet=quiet, **fit_kwargs)


def fit_cv (model, X, y, nb_folds=5, quiet=False, **fit_kwargs):
    """
    Convenience method for performing cross-validation training of both scikit-
    learn and Keras classifiers. In the internal.utilities module, the internal 
    method infers the type of `model` and runs the appropriate wrapper around 
    the fitting function for this class. If `model` is a scikit-learn classifier,
    the method decorates it with the attributes `cv_history_loss` and 
    `cv_history_acc`; if `model` is a Keras model, the loss histories for each 
    CV fold can be accessed from the `model.history` attribute.
    
    Note, in this method only *copies* of `model` are fitted, not `model` itself. 
    This means that the loss and accuracy histories are set for each of the CV 
    folds, but the network weights of `model` are not updated (because this 
    would be ambiguous). If you want to train `model` itself, use the `fit` 
    method above. (Note also that running either of the `fit{,_cv}` methods for 
    a Keras model will overwrite any previously set `history` attribute.)
    
    If you want, you can see the internal method(s) for implementation details.
    
    Arguments: (same as for `fit` above, except for)
        nb_folds: Integer, number of cross-validation folds to use. In the 
            internal method, this is used to automatically specify the 
            `validation` argument of the `fit` method.
    
    Raises:
        AssertionError, if any of the checks fail.
    
    Returns:
        *Un-*fitted model, with modified attributes to log the CV training 
        histories (see above).
    """
    return _internal.utilities._fit_cv (model, X, y, nb_folds=nb_folds, quiet=quiet, **fit_kwargs)


def get_output (model, X):
    """
    Class-agnostic method to get the output classification probabilities of the 
    input `model`. If the model is not a scikit-learn model, then the bare 
    output of the model is returned; it is assumed that this is a list of class 
    probabilities.
    
    Arguments:
        model: Instance of keras.models.Model, sklearn.neural_network.MLPClassifier, 
            or torch.nn.Module. 
        X: numpy.array, of shape (nb_samples, nb_features), containing the 
            feature array for which classification probabilities are sought.
        
    Raises:
        Exception, if the model type is not supported
        
    Returns:
        numpy.array, of shape (nb_samples, nb_classes), containing (it is 
            assumed, cf. above) the class probabilities assigned by the model.
    """
    
    if   isinstance(model, _keras.models.Model):
        return model.predict(X)
    elif isinstance(model, _sklearn.neural_network.MLPClassifier):
        return model.predict_proba(X)
    elif isinstance(model, _torch.nn.Module):
        pred = model(X)
        return pred.cpu().data.numpy()
    else:
        raise Exception(f"Model of type {type(model)} not supported.")
    return


# ==============================================================================
# Neural network architecture inference
# ------------------------------------------------------------------------------

def get_architecture (model):
    """
    Infer the architecture of `model`.
    
    Arguments:
        model: Instance of keras.models.Model, sklearn.neural_network.MLPClassifier, 
            or torch.nn.Module. 
    
    Returns:
        Tuple of number of layers in input, hidden, and output layers, or 
        formatted names of layers as strings.
    """
    if   isinstance(model, _sklearn.neural_network.MLPClassifier):
        return _internal.utilities._get_architecture_from_sklearn(model)
    elif isinstance(model, _keras.models.Model):
        return _internal.utilities._get_architecture_from_keras(model)
    elif isinstance(model, _torch.nn.Module):
        return _internal.utilities._get_architecture_from_pytorch(model)
    print("Error: Model of type {} not recognised.".format(type(model)))
    return


def get_architecture_string (model):
    """
    Format the architecture of `model` as a string.
    """
    return u' → '.join(map(str, get_architecture(model)))


def get_network_name (model):
    """
    Infer the name of `model`. If no name is specifically set, the library in 
    which the model of the created along with the model architecture, are 
    returned as a string.
    """
    if hasattr(model, 'name'):  # Keras
        source = "Keras: "
        if not (model.name == 'model' or model.name.startswith('model_')):  # Not default name
            return model.name
    elif isinstance(model, _torch.nn.Module):
        source = "Pytorch: "
    else:
        source = "MLP:    "
        pass
    return source + get_architecture_string(model)
