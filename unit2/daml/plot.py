# plots.py: Collection of plotting methods common to several DAML CPs.
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
import copy as _copy
import numpy as _np
import pandas as _pd
import sklearn as _sklearn
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt
import tensorflow.python.keras as _keras
from sklearn.model_selection import GridSearchCV as _GridSearchCV

# Local import(s)
from . import utilities    as _utilities
from . import optimisation as _optimisation
from . import _internal


# ==============================================================================
# Data and performance evaluation
# ------------------------------------------------------------------------------

def decision_contour (clf, nb_classes, ax):
    """
    Draw the decision contour for the scikit-learn classifier `clf` on the 
    pyplot axes `ax`. The classifier should be trained to classify among 
    `nb_classes` classes.
    
    Arguments:
        clf: Scikit-learn classifier, e.g. sklearn.tree.DecisionTreeClassifier, 
            sklearn.neural_network.MLPClassifier, or similar.
        nb_classes: Number of classes which the classifier is trained to 
            distinguish between.
        ax: pyplot.Axis instance, on which the decision contour should be drawn.
    """

    # Draw decision countour
    cmap = _utilities.get_cmap(nb_classes)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Construct mesh grid
    xx, yy = _np.meshgrid(_np.linspace(xmin, xmax, 100 + 1, endpoint=True),
                          _np.linspace(ymin, ymax, 100 + 1, endpoint=True))
    XX = _np.stack([xx.flatten(), yy.flatten()]).T

    # Get classifier predicton and probabilities on grid
    preds = clf.predict(XX).reshape(xx.shape)
    probs = clf.predict_proba(XX).reshape(xx.shape + (nb_classes,))

    zz = preds

    # Common `imshow` options
    opts = dict(extent=(xmin, xmax, ymin, ymax), origin='lower', aspect='auto')

    # Plot decision contour
    ax.imshow(zz, cmap=cmap, alpha=0.5, zorder=-1001, **opts)

    # Compute and plot opacity based on confidence
    confidence = _utilities.get_confidence(probs)
    
    opacity = _np.zeros_like(zz)
    opacity = _matplotlib.cm.get_cmap('binary')(opacity)
    opacity[..., -1] = 1 - confidence

    ax.imshow(opacity, zorder=-1000, **opts)
    return


def scatter (X, y=None, clf=None, s=None, feature_names=None, target_names=None, 
             ax=None, legend=True, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Draw scatter-plot, optionally with decision contours overlaid. Supports 
    several stylistic arguments.

    Arguments:
        X: numpy.array or pandas.DataFrame, of variables to be shown in scatter 
            plot, assumed to have shape (N,2).
        y: numpy.array or pandas.DataFrame, of the class for each example, 
            assumed to have shape (N,). If None, taken to be all of same type.
        clf: Scikit-learn classifier, the decision contour of which will be 
            drawn if specified.
        s: Marker size to be used in scatter plot, passed to pyplot.scatter.
        feature_names: List or tuple of strings, used as axis labels. Should 
            match the number of columns in `X`. If not specified, either the 
            column names in `X` (if this is a pandas.DataFrame) or general 
            labels are used.
        target_name: List or tuple of strings, used as legend entries for the 
            classes in `y`. Should match the number of classes in `y`. If not 
            specified, generic labels are used.
        ax: pytplot.Axis object on which the scatter plot should be drawn. If 
            not specified, a new figure will be created.
        legend: Boolean, whether to draw a legend on the plot.
        xmin: Float, lower end of x-axis to draw.
        xmax: Float, upper end of x-axis to draw.
        ymin: Float, lower end of y-axis to draw.
        ymax: Float, upper end of y-axis to draw.
        
    Raises:
        AssertionError, if any of the checks fail.

    Returns:
        pyplot.Figure object associated with the pytplot.Axis object on which 
        the scatter plot is drawn.
    """

    # Check(s)
    # -- Features
    if isinstance(X, _pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
            pass
        X = X.values
    elif isinstance(X, (list, tuple)):
        X = _np.asarray(X)
        pass

    # -- Targets
    if y is None:
        y = _np.zeros(X.shape[0], dtype=int)
    elif isinstance(y, (_pd.DataFrame, _pd.Series)):
        y = y.values
    elif isinstance(y, (list, tuple)):
        y = _np.asarray(y)
        pass

    assert isinstance(y, _np.ndarray), \
        "Argument y of type '{}' is not understood".format(type(y))

    y = y.flatten()

    # -- Dimensions
    assert X.shape[0] == y.shape[0], \
        "Number of features ({}) and number of targets ({}) differ.".format(X.shape[0], y.shape[0])
    assert X.shape[1] == 2, \
        "Please specify two features to be plotted."

    # -- Get list of unique target classes
    yvals = sorted(set(y))

    # -- Names
    if feature_names is None:
        feature_names = ['$x_{%d}$' % (ix + 1) for ix in range(X.shape[1])]
        pass

    if target_names is None:
        target_names = [yval if isinstance(yval, str) else 'y = {}'.format(yval) for yval in yvals]
        pass

    # -- Marker sizes
    if s is None:
        s = _np.ones_like(y) * _matplotlib.rcParams['lines.markersize'] ** 2
    elif isinstance(s, (list, tuple)):
        s = _np.array(s)
        pass

    # Create figure
    if ax is None:
        _, ax = _plt.subplots(figsize=(4.5,4))
        pass

    # Get axis limits
    (x1min, x2min), (x1max, x2max) = X.min(axis=0), X.max(axis=0)

    # Padding
    padding = 0.10
    dx1 = (x1max - x1min)
    dx2 = (x2max - x2min)
    x1min -= padding * dx1
    x1max += padding * dx1
    x2min -= padding * dx2
    x2max += padding * dx2

    x1min = xmin or x1min
    x2min = ymin or x2min
    x1max = xmax or x1max
    x2max = ymax or x2max

    # Scatter points
    for ix, yval in enumerate(yvals):
        mask = (y == yval)
        ax.scatter(*X[mask].T.tolist(), alpha=0.7,
                   label=target_names[ix],
                   edgecolor='k',
                   s=list(s[mask]))
        pass

    # Decorations
    if legend and len(yvals) > 1:
        ax.legend()
        pass

    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_xlim(x1min, x1max)
    ax.set_ylim(x2min, x2max)

    # Draw decision countour?
    if clf is not None:
        decision_contour(clf, len(yvals), ax)
        pass

    fig = ax.figure
    fig.tight_layout()

    return fig


def pair_grid (data, target, features=None, clf=None, refit=False, nbins=30):
    """
    Draw a lower-triangle figure of histograms (on-diagonal) and scatter plots 
    (off-diagonal) of all features and pairs of features, respectively, in `data`.

    Arguments:
        data: pandas.DataFrame or numpy.array, containing features to be plotted.
        target: Source of class labels; either a string, in case `data` is a 
            pandas.DataFrame in which `target` is interpreted as a column; or a 
            numpy.array, in case `data` is a numpy array.
        features: List of strings, specifying the features to plot.
        clf: Scikit-learn classifier, the decision contour of which will be 
            drawn if specified.
        refit: Boolean, whether to refit `clf` for all combinations of pairs of 
            features, or whether to alternatively interpolate the decision 
            contour for a pre-fitted classifier. See the `lecture2.ipynb` 
            notebook for details.
        nbins: Integer, number of histogram bins to use.

    Raises:
        AssertionError, if any of the checks fail.

    Returns:
        pyplot.figure containing grid of pair-wise plots and distributions.
    """

    # Check(s)
    if isinstance(data, _np.ndarray):
        assert isinstance(target, _np.ndarray), \
            "Unsupported combination of types {} and {}.".format(type(data), type(target))
        if features and (data.shape[1] == len(features)):
            columns = features
        else:
            columns = ['$x_{{{}}}$'.format(ix + 1) for ix in range(data.shape[1])]
            pass
        df = _pd.DataFrame(data, columns=columns)
        df['target'] = target
        target = 'target'
    else:
        df = data
        pass
    
    assert isinstance(df, _pd.DataFrame), \
        f"Could not convert input data of type {type(df)} to pandas.DataFrame."
    assert target in df.columns, \
        "Requested target '{}' does not exist in DataFrame with columns: {}.".format(target, df.columns)

    if features is None:
        features = [f for f in df.columns if f != target]
    else:
        for feat in features:
            assert feat in df.columns, \
                "Requested feature '{}' does not exist in DataFrame with columns: {}.".format(feat, df.columns)
            pass
        pass

    # Get arrays
    labels = df[target].values
    labels_set = sorted(set(labels.flatten()))
    y = _np.zeros_like(labels, dtype=int)
    for yval, cls in enumerate(labels_set):
        msk = (labels == cls)
        y[msk] = yval
        pass
    X  = df[features].values
    nb = X.shape[1]

    # Create figure canvas
    fig = _plt.figure(figsize=(3*nb, 3*nb))

    # Method for easily accessing axes on grid. Cache is used to avoid warning
    # from potential overwriting in future versions of matplotlib.
    _axis_cache = _np.asarray([[None] * nb] * nb)
    def axis (ix, jx):
        if _axis_cache[ix, jx] is None:
            opts = dict()
            if ix != jx:
                if jx < nb - 1:
                    opts['sharex'] = axis(ix,nb - 1)
                    pass
                if ix > 0:
                    opts['sharey'] = axis(0,jx)
                    pass
                pass
            _axis_cache[ix, jx] = _plt.subplot(nb, nb, jx * nb + ix + 1, **opts)
            pass
        return _axis_cache[ix, jx]

    for ix in range(nb):
        # Scatter
        for jx in range(ix + 1, nb):
            x1, x2 = X[:,ix], X[:,jx]

            ax = axis(ix,jx)
            for yval, label in enumerate(labels_set):
                msk = (y == yval)
                ax.scatter(x1[msk], x2[msk], label=str(label), alpha=0.5, edgecolor='black')
                pass
            if jx == nb - 1:
                ax.set_xlabel(features[ix])
                pass
            if ix == 0:
                ax.set_ylabel(features[jx])
                pass
            pass

        # Histograms
        x = X[:,ix]
        bins = _np.linspace(x.min(), x.max(), nbins + 1, endpoint=True)
        ax = axis(ix,ix)
        for yval, label in enumerate(labels_set):
            msk = (y == yval)
            ax.hist(x[msk], bins=bins, label=str(label), alpha=0.7)
            pass
        if ix == nb - 1:
            ax.set_xlabel(features[ix])
            pass
        pass

    # Draw legend (once)
    fig.axes[-1].legend()

    if clf is not None:
        for ix in range(nb):
            for jx in range(ix + 1, nb):

                # Get fit locations
                x1, x2 = X[:,ix], X[:,jx]
                X_ = _np.stack([x1, x2]).T

                # Get mesh locations
                xx, yy = _np.meshgrid(_np.linspace(x1.min(), x1.max(), 100 + 1, endpoint=True),
                                     _np.linspace(x2.min(), x2.max(), 100 + 1, endpoint=True))
                XX_ = _np.stack([xx.flatten(), yy.flatten()]).T

                # Refit on two just two dimensions?
                if refit:
                    clf = _copy.deepcopy(clf)
                    fit = clf.fit(X_, y)

                # Or interpolate pre-fitted classifier using k-NN?
                else:
                    ipt = _internal.plot.Interpolator(clf)
                    fit = ipt.fit(X, X_)
                    pass

                # Get predictions on mesh
                zz = fit.predict(XX_).reshape(xx.shape)

                # Draw contours
                decision_contour(fit, len(labels_set), axis(ix,jx))
                pass
            pass

        pass

    fig.tight_layout()
    return fig


def confusion_matrix (cm, labels=None, normalise=False):
    """
    Plot the confusion matrix `cm`.
    
    Arguments:
        cm: numpy.array, assumed to be square with dimensions along each axis 
            equal to the number of classes being classified.
        labels: List of strings, used as labels for the classes being classified. 
            If specified, the number of labels should be equal to the number 
            inferred from the shape of `cm`.
        normalise: Boolean, whether to normalise the confusion matrix, to show 
            fractions, or to show the raw sample count.

    Raises:
        AssertionError, if any of the checks fail.

    Returns:
        pyplot.figure containing the confusion matrix plot.
    """
    
    # Check(s)
    nb_classes = cm.shape[0]
    if labels is None:
        labels = _np.arange(nb_classes)
    else:
        assert len(labels) == nb_classes
        pass

    # (Opt.) Normalise the confusion matrics by 'True label' (i.e. all predicted
    # labels for a given true label sum to one).
    if normalise:
        cm = cm / cm.sum(axis=1)[:,_np.newaxis]
        pass
    
    # Determine colour-axis maximum
    vmax = 1. if normalise else float(cm.max())
    
    # Create figure
    fig, ax = _plt.subplots()
    
    # Display confusion matrix
    ax.imshow(cm, origin='lower', cmap='Blues', vmax=vmax)
     
    # Draw matrix entry values
    for (j,i), label in _np.ndenumerate(cm):
        ax.text(i, j, '{:.1f}%'.format(label * 100.) if normalise else label, 
                 ha='center', va='center', fontdict={'color': 'white' if label/vmax > 0.6 else 'black'})
        pass
   
    # Decorations
    ax.set_xticks(range(nb_classes))
    ax.set_yticks(range(nb_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label', fontdict={'weight': 600}, labelpad=20)
    ax.set_ylabel('True label',      fontdict={'weight': 600}, labelpad=15)
    if normalise:
        fig.suptitle("(Normalised by 'True label')")
        pass
    
    return fig



# ==============================================================================
# Decision trees
# ------------------------------------------------------------------------------

def tree (dt, feature_names, target_names, fname=None):
    """
    Plot decision tree logic using sklearn export_graphviz.
    See: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

    Arguments:
        dt: sklearn.tree.DecisionTreeClassifier, the decision logic of which to 
            plot.
        feature_names: List of strings, naming the features provided as inputs 
            to `dt`.
        target_names: List of strings, naming the classes being classified by 
            `dt`.
        fname: String, name of file to which the tree plot is saved, if provided.
        
    Returns:
        pydotplus.graphviz.Dot graphic containing the decision tree logic. Can 
            be visualised in a Jupyter notebook by:
            ```
            from IPython.display import Image
            graph = plot.tree(...)
            Image(graph.create_png())
            ```
    """

    # Import(s)
    from sklearn import tree
    from sklearn.externals.six import StringIO
    import pydotplus

    if (fname is not None):
        tree.export_graphviz(dt, out_file=fname, filled=True, rounded=True,
                             special_characters=True,
                             feature_names=feature_names,
                             class_names=target_names)
        graph = 0

    else:
        dot_data = StringIO()
        tree.export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                             special_characters=True,
                             feature_names=feature_names,
                             class_names=target_names)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph = _utilities.update_graph_colours(graph)
        pass

    return graph



# ==============================================================================
# Training loss
# ------------------------------------------------------------------------------ 

def loss (*clfs, scale='linear', cv=None, minima=False):
    """
    Plot the neural network loss curve(s) for the classifier(s)`clfs`.
    
    Arguments:
        clfs: Variable-length list of scikit-learn classifiers, such as 
            sklearn.neural_network.MLPClassifier, or Keras networks, of type 
            keras.models.Model. The classifiers are assumed to have been fitted, 
            either using their respective `fit`-methods, or using the 
            `utilities.fit` or `utilities.fit_cv` methods. These are specified 
            in the function call as e.g.:
            ```
            plot.loss(clf1, clf2, clf3);
            ```
        scale: String, either 'linear' or 'log', specifying the type of y-axis 
            scale to draw.
        cv: Whether to plot cross-validation (CV) bands. By default, this is 
            inferred automatically for each classifier argument. CV-bands require 
            that the classifier was fitted using the `utilities.fit_cv` method.
        minima: Boolean, whether to draw markers for the (validation) loss minima.
        
    Raises:
        AssertionError, if any of the checks fail.

    Returns:
        pyplot.figure containing the loss curve plot.    
    """
       
    # Check(s)
    for ix, clf in enumerate(clfs, start=1):
        assert isinstance(clf, (_sklearn.base.BaseEstimator, _keras.models.Model, _keras.callbacks.History)), \
            f"Argument {ix}: Type {type(clf)} not understood."
        assert not isinstance(clf, (_GridSearchCV)), \
            f"Argument {ix}: Cannot plot loss for classifier of type {type(clf)}."
        pass
    assert scale in ['linear', 'log']
    
    histories = list()
    for clf in clfs:
        if isinstance(clf, _sklearn.base.BaseEstimator):
            histories.append(_internal.plot._history_from_sklearn(clf))
        else:
            histories.append(_internal.plot._history_from_keras(clf))
            pass
        pass

    # Call base function
    fig = _internal.plot._loss(*histories, scale=scale, cv=cv, minima=minima)
    
    return fig



# ==============================================================================
# Classification performance
# ------------------------------------------------------------------------------

def roc (*clfs, X=None, y=None, target_eff=0.5, scale='log'):
    """
    Plot the so-called receiver operating characteristic (ROC) curve(s) for the 
    classifier(s) `clfs`.
    
    Arguments:
        clfs: Variable-length list of scikit-learn classifiers, such as 
            sklearn.neural_network.MLPClassifier, or Keras networks, of type 
            keras.models.Model. The classifiers are assumed to have been fitted 
            on the same set of features to classify only two target classes 
            either using their respective  `fit`-methods, or using the 
            `utilities.fit` method. These are specified in the function call as 
            e.g.:
            ```
            plot.roc(clf1, clf2, clf3, X=X_test, y=y_test);
            ```
            *Please note* that the remaining arguments have to be specified as 
            keywords, i.e. using the `..., kw=<something>, ...` syntax suggested 
            above.
        X: numpy.array, of shape (nb_samples, nb_features), containing the array 
            of features on which the classifier(s) should be evaluated.
        y: numpy.array, of shape (nb_samples, 1), containing the list targets 
            classes, assumed to be either 0 or 1.
        target_eff: Float, the target signal (y = 1) efficiency at which the 
            corresponding background rejection rate should be evaluated.
        scale: String, either 'log' or 'linear', specifying the type of y-axis 
            scale to draw.
            
    Raises:
        AssertionError, if any of the checks fail.
        
    Returns:
        pyplot.figure containing the ROC curve plot.
    """
    
    # Check(s)
    assert X is not None, \
        "Please specify a testing dataset, `X`."
    assert y is not None, \
        "Please specify true labels, `y`."
    assert X.shape[0] == y.shape[0]
    assert len(y.squeeze().shape) == 1
    
    # Import(s)
    from sklearn.metrics import roc_curve    
    
    # Create figure
    fig, ax = _plt.subplots(figsize=(5,4))
    
    # Random guessing line
    tpr = _np.linspace(0, 1, 200 + 1, endpoint=True)[1:]
    ax.plot(tpr, 1./tpr, color='gray', ls='--', label='Random guessing')
    
    # ROC curves
    rocs = list()
    for clf in clfs:
        pred = get_output(clf, X)
        sign = 1. if pred[y == 1].mean() > pred[y == 0].mean() else -1.
        fpr, tpr, _ = roc_curve(y, sign * pred)
        
        # Filter out entries with no background passing
        msk = (fpr > 0)
        tpr = tpr[msk]
        fpr = fpr[msk]
        
        # Plot ROC curve
        ax.plot(tpr, 1./fpr, label=_utilities.get_network_name(clf))
        
        # Store efficiency arrays
        rocs.append((tpr, fpr))
        pass
    
    # Indicate best result
    best_rejection = -_np.inf
    best_clfs      = []
    ax.axvline(target_eff, c='darkgray', ls=':', lw=1)
    for ix, (clf, (tpr, fpr)) in enumerate(zip(clfs, rocs)):
        if target_eff < tpr.min():
            print("No valid background rejection rate at target efficiency ({:.0f}%) for model: {}".format(target_eff * 100., _utilities.get_network_name(clf)))
            best_rejection = _np.inf
            best_clfs.append(ix)
            continue
            
        rejection = _np.interp(target_eff, tpr, 1./fpr)
      
        if rejection > best_rejection:
            best_rejection = rejection
            best_clfs.append(ix)
            pass
        pass
         
    # Decorations
    ax.set_yscale(scale)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection factor")
   
    if not _np.isinf(best_rejection):
        frac = 0.02
        
        x = target_eff
        y = best_rejection
        y1, y2 = ax.get_ylim()
        
        logyDraw = (_np.log10(y) - _np.log10(y1) + frac * (_np.log10(y2) - _np.log10(y1))) + _np.log10(y1)

        xDraw = x + frac * (1. - 0.)
        yDraw = _np.power(10, logyDraw)

        ax.plot(target_eff, best_rejection, 'r*')
        digits = 1 if best_rejection < 100 else 0
        ax.text(xDraw, yDraw, f'x{{:.{digits}f}}'.format(best_rejection), fontdict={'weight': 600})
        pass
    
    # Draw legend and boldface the best instance(s)
    l = ax.legend()
    for ix, text in enumerate(l.get_texts()[1:]):
        if ix in best_clfs:
            text._fontproperties = l.get_texts()[0]._fontproperties.copy()
            text.set_weight(600)
            pass
        pass
   
    return fig



# ==============================================================================
# Hyperparameter optimisation
# ------------------------------------------------------------------------------

def optimisation_grid (clf, fmt="{:.1f}%", scale=100.):
    """
    Draw hyper-parameter optimisation plot for scikit-learn GridSearchCV. Will 
    show the average cross-validation accuracy for each search parameter 
    configuration, and indicate the "optimal" (rather: best found) configuration.
    
    Arguments:
        clf: Instance of sklearn.model_selection.GridSearchCV, the results of 
            which to plot. The classifier is assumed to have been fitted, and 
            currently only supports exactly two feature dimensions.
        fmt: String, format pattern used when printing evaluation results.
        scale: Float, scaling applied to evaluation results before drawing.
    
    Raises:
        AssertionError, if any of the checks fail.
        
    Returns:
        pyplot.Figure containing the optimisation plot.
    """
    
    # Check(s)
    assert isinstance(clf, _GridSearchCV)
    
    param_names  = list(sorted(clf.param_grid.keys()))
    param_values = [clf.param_grid[name] for name in param_names]
    dims = list(map(len, param_values))
    
    assert len(dims) <= 2, \
        "Can only plot optimisation for up to two dimensions."
    
   
    # 1D
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if len(dims) == 1:
        xticks = param_values[0]
        
        x = _np.arange(len(xticks))
        yval = clf.cv_results_['mean_test_score']
        yerr = clf.cv_results_['std_test_score']
        
        fig, ax = _plt.subplots(figsize=(4.5,4))
        
        line = ax.plot(x, yval)
        fill = ax.fill_between(x, yval - yerr, yval + yerr, alpha=0.2)
        
        ixmax = _np.argmax(yval)
        ax.plot([x[ixmax]], [yval[ixmax]], 'y*', label='Expected best')
        
        # Decorations
        ax.set_xticks(x)
        ax.set_xticklabels(xticks)
        ax.set_xlabel(param_names[0], fontdict={'weight': 600}, labelpad=15)
        ax.set_ylabel('Validation accuracy', fontdict={'weight': 600}, labelpad=15)
        
        handles, labels = ax.get_legend_handles_labels()
        handles += [(line[0], fill)]
        labels  += [u'Mean value ± std. dev.']
        ax.legend(handles, labels)
        
        
    # 2D
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    else:
        M = _np.zeros(dims).T
        for iperm, score in enumerate(clf.cv_results_['mean_test_score']):

            # Get parameter axis indices for current score
            indices = list()
            for name in param_names:
                value = clf.cv_results_['param_' + name][iperm]
                indices.append(clf.param_grid[name].index(value))
                pass

            M[indices[1], indices[0]] = score
            pass

        fig, ax = _plt.subplots(figsize=(dims[0] * 2, dims[1]))
        im = ax.imshow(M, origin='lower', cmap='Blues')
        ax.set_xlabel(param_names[0], fontdict={'weight': 600}, labelpad=15)
        ax.set_ylabel(param_names[1], fontdict={'weight': 600}, labelpad=15)

        # Draw matrix entry values
        for (j,i), label in _np.ndenumerate(M):
            ax.text(i, j, fmt.format(label * scale),
                     ha='center', va='center', fontdict={'color': 'white' if (label - M.min())/(M.max() - M.min()) > 0.6 else 'black'})
            pass

        # Indicate optimal configuration
        xgrid, ygrid = _np.meshgrid(*param_values)
        xmax,  ymax  = xgrid[M == M.max()][0], ygrid[M == M.max()][0]
        ixmax, iymax = param_values[0].index(xmax), param_values[1].index(ymax)
        ax.plot([ixmax], [iymax - 0.25], 'y*', markersize=10)

        # Decoration
        ax.set_xticks(range(dims[0]))
        ax.set_yticks(range(dims[1]))
        ax.set_xticklabels(param_values[0])
        ax.set_yticklabels(param_values[1])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mean validation accuracy', fontdict={'weight': 600}, labelpad=15)
        pass
    
    return fig


def optimisation_gp (bo, fmt="{:.1f}%", scale=100.):
    """
    Draw hyper-parameter optimisation plots for BayesianOptimiser. Will show the
    average cross-validation accuracy for each sampled configuration on three 
    plots, containing (1) the best-fit Gaussian process (GP) prediction of the 
    classifier accuracy accross the search space; (2) the +/- 1 sigma GP 
    uncertainty band on the classifier accuracy; and (3) the expected 
    improvement (EI) metric, used when sampling the next parameter configuration 
    in the search. The plots will indicate the best parameter configuration 
    sampled, along with the position of the expected classifier accuracy 
    according to the GP regression.
    
    Arguments:
        bo: Instance of optimisation.BayesianOptimiser, the results of which to 
            plot. The classifier is assumed to have been fitted, and currently 
            only supports exactly two feature dimensions.
        fmt: String, format pattern used when printing trial results.
        scale: Float, scaling applied to trial results before drawing.
            
    Raises:
        AssertionError, if any of the checks fail.
        
    Returns:
        pyplot.Figure containing the optimisation plots.
    """
    
    # Check(s)
    assert isinstance(bo, _optimisation.BayesianOptimiser)
    assert len(bo.dimensions) <= 2, \
        "Can only plot optimisation for up to two dimensions."
    
   
    # 1D
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if len(bo.dimensions) == 1:
        fig, axes = _plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=False)
        
        xmin, xmax = bo.dimensions[0].low, bo.dimensions[0].high
 
        dx = xmax - xmin
        
        for ix, ax in enumerate(axes):
            if ix == 0:
                func = lambda X: bo.gp.predict(X, return_std=True)
            else:
                func = lambda X: (bo.expected_improvement(X), None)
                pass
            
            yval, yerr = func(bo.X)
        
            line = ax.plot(bo.X, yval)
            if yerr is not None:
                fill = ax.fill_between(bo.X.flatten(), yval - yerr, yval + yerr, alpha=0.2)
                pass
            
            # Measurements
            if ix == 0:
                xt = [t.site[0] for t in bo.trials]
                yt = [t.value   for t in bo.trials]
                et = [t.error   for t in bo.trials]
                ax.errorbar(xt, yt, et, fmt='k.', label='Evaluations')
                pass
            
            # Best expected
            best = bo.get_best()
            ax.axvline(best[0], color='y', ls='--', lw=1)
            if ix == 0:
                ax.plot([best[0]], [bo.gp.predict([best])], 'y*', label='Expected best') 
                pass
            
            # Legend
            handles, labels = ax.get_legend_handles_labels()
            if ix == 0:
                handles += [(line[0], fill)]
                labels  += [u'GP pred. µ(x) ± σ(x)']
                pass
            ax.legend(handles, labels)
            
            # Decorations
            ax.set_xlabel(bo.dimensions[0].name, fontdict={'weight': 600}, labelpad=15)
            ax.set_xlim(xmin, xmax)
            
            if   ix == 0:
                ax.set_ylabel('Objective function value', fontdict={'weight': 600}, labelpad=15)
            else:
                ax.set_ylabel('Expected improvement (EI)', fontdict={'weight': 600}, labelpad=15)
                pass
            pass
            

    # 2D
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    else:

        fig, axes = _plt.subplots(1, 3, figsize=(15,4), sharex=True, sharey=True)
        
        xmin, xmax = bo.dimensions[0].low, bo.dimensions[0].high
        ymin, ymax = bo.dimensions[1].low, bo.dimensions[1].high

        dx = xmax - xmin
        dy = ymax - ymin

        for ix, ax in enumerate(axes):
            if ix == 0:
                func = bo.gp.predict
            elif ix == 1:
                func = lambda X: bo.gp.predict(X, return_std=True)[1]
            else:
                func = bo.expected_improvement
                pass
            yval = func(bo.X).reshape(bo.mesh[0].shape)
            vmin, vmax = yval.min(), yval.max()

            if ix == 1:
                vmin = 0
                pass

            im = ax.contourf(bo.mesh[0], bo.mesh[1], yval, levels=11, cmap='Blues', vmin=vmin, vmax=vmax)

            for trial in bo.trials:
                x, y = trial.site

                if ix == 0:
                    p = func([trial.site])
                else:
                    p = func([trial.site])
                    pass
                colour = 'white' if (p - vmin)/(vmax - vmin) > 0.6 else 'black'
                if trial.value == _np.max([t.value for t in bo.trials]):
                    colour = 'red'
                    pass

                ax.scatter([x], [y], color=colour, s=10)
                ax.text(x + 0.01 * dx, 
                        y + 0.01 * dy, 
                        fmt.format(trial.value * scale),
                        ha='left'   if (x - xmin) / dx < 0.90 else 'right', 
                        va='bottom' if (y - ymin) / dy < 0.90 else 'top', 
                        fontdict={'color': colour})
                pass

            # Predicted best
            best = bo.get_best()
            ax.plot([best[0]], [best[1]], 'y*') 

            # Decorations
            ax.set_xlabel(bo.dimensions[0].name, fontdict={'weight': 600}, labelpad=15)
            if ix == 0:
                ax.set_ylabel(bo.dimensions[1].name, fontdict={'weight': 600}, labelpad=15)
                pass
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            cbar = fig.colorbar(im, ax=ax)
            if   ix == 0:
                clabel = u'GP pred. mean, µ(x)'
            elif ix == 1:
                clabel = u'GP pred. std. dev., σ(x)'
            else:
                clabel = 'Expected improvement (EI)'
                pass
            cbar.set_label(clabel, fontdict={'weight': 600}, labelpad=15)
            pass
        pass

    fig.tight_layout()
    return fig


def optimisation (clf, fmt="{:.1f}%", scale=100.):
    """
    Wrapper around the class-specific hyperparameter search classes supported.
    
    Arguments:
        clf: Instance of GridSearchCV or optimisation.BayesianOptimiser.
        kwargs: Keyword-arguments, passed to the relevant class-specific method.
        fmt: String, format pattern used when printing evaluation results.
        scale: Float, scaling applied to evaluation results before drawing.
    
    Returns:
        pyplot.Figure containing the optimisation plot(s), or `None` if the 
            argument type is not supported.
    """
    
    if   isinstance(clf, _GridSearchCV):
        return optimisation_grid(clf, fmt=fmt, scale=scale)
    elif isinstance(clf, _optimisation.BayesianOptimiser):
        return optimisation_gp(clf, fmt=fmt, scale=scale)
        pass
    print("optimisation: Argument type {} not supported.".format(type(clf)))
    return



# ==============================================================================
# Regularisation
# ------------------------------------------------------------------------------

def network_weights (model):
    """
    Plot distributions of network weights.
    
    This method will iterate through all (assumed) dense connections between the 
    input, hidden, and output layers, and plot separate distributions of the 
    entries of the weight matrices (*not* the bias vectors) for each layer.
    
    Arguments:
        model: Instance of sklearn.neural_network.MLPClassifier or 
            keras.models.Model, the weights of which should be plotted.
            
    Returns:
        pyplot.Figure containing the weight distribution plot.
    """
    
    xmin, xmax = 1.0E-06, 10.0
    bins = _np.logspace(_np.log10(xmin), _np.log10(xmax), 100 + 1, endpoint=True)
    eps  = 1.0E-10 

    # Create figure
    fig, ax = _plt.subplots()
    
    # Keras model
    if isinstance(model, _keras.models.Model):
        nb_layers = len(model.layers) - 1  # Excl. input layer
        def get_weights (index):
            return model.layers[index + 1].get_weights()

    # Assuming MLPClassifier
    else:
        nb_layers = len(model.coefs_)
        def get_weights (index):
            return model.coefs_[index], model.intercepts_[index]
        pass
    
    title  = _utilities.get_network_name(model)
    print(f"{title}:")
    
    # Loop layers
    for ilayer in range(nb_layers):
            
        W, b = get_weights(ilayer)
        print(u"  Layer {} → Weight matrix: {:3d} x {:3d} | Bias vector: {:3d}".format(ilayer + 1, *(W.shape + b.shape)))

        x = _np.abs(W.flatten())
        
        h,_,_ = ax.hist(x, bins=bins, weights=_np.ones_like(x) / float(x.size), label=f'Layer {ilayer}', alpha=0.3)
        pass
    
    # Decorations
    ax.set_ylabel('Fraction of weights')
    ax.set_xlabel("Magnitude of weight matrix elements, $|w_{ij}|$")
    ax.set_xscale('log')
    ax.legend()
    fig.suptitle(title)
    
    return fig


def network_weights_information (model, perc=0.98):
    """
    Plot the cumulative distribution for the absolute weights for each layer in 
    `model`. In much the same way as `network_weights` (above) this method finds 
    the weights matrices for all layers in `model` and, for each layer, takes 
    the absolute values of the entries in the weight matrix, and computes the
    cumulative distribution, normalised to 1. This is an attempt to illustrate 
    the *sparsity* of a given weight matrix: if the cumulative distribution is 
    diagonal, the weight matrix is completely uniform; and the further the 
    cumulative distribution reaches towards the to-left corner, the more sparse 
    it is.
    
    Arguments:
        model: Instance of sklearn.neural_network.MLPClassifier or 
            keras.models.Model, the weights informtaion of which should be 
            plotted.
        perc: Float, the "information" percentile at which a guiding line should 
            be drawn. This illustrates the fraction of weight matrix entries 
            which should be retained to keep at least `perc` of the "information"
            contained in the weight matrix.
    
    Raises:
        AssertionError, if any of the checks fail.
    
    Returns:
        pyplot.Figure containing the weight information plot.
    """
    
    # Check(s)
    assert isinstance(perc, float)
    assert perc > 0 and perc < 1
    
    fig, ax = _plt.subplots(figsize=(6,6))
    title   = _utilities.get_network_name(model)
    print(f"{title}:")
    
     # Keras model
    if isinstance(model, _keras.models.Model):
        nb_layers = len(model.layers) - 1  # Excl. input layer
        def get_weights (index):
            return model.layers[index + 1].get_weights()

    # Assuming MLPClassifier
    else:
        nb_layers = len(model.coefs_)
        def get_weights (index):
            return model.coefs_[index], model.intercepts_[index]
        pass
    
    cmap = _utilities.get_cmap(nb_layers)
    
    for ilayer in range(nb_layers):
        W,b = get_weights(ilayer)
        w = _np.abs(W.flatten())
        w = sorted(w, reverse=True)

        cdf = _np.cumsum(w)/_np.sum(w)
        nb_weights = len(cdf)
        ix_cross = _np.interp(perc, cdf, _np.arange(nb_weights))
        ix_cross = int(_np.ceil(ix_cross))
        print("  Keep {:5d} of {:5d} weights, and {:.1f}% 'information'".format(ix_cross, nb_weights, cdf[ix_cross] * 100.))
        
        ax.plot(_np.arange(nb_weights) / float(nb_weights), cdf, c=cmap(ilayer), label=f'Layer {ilayer + 1}')
        ax.axhline(perc, c='gray', lw=1, ls='--')
        ax.plot([ix_cross / float(nb_weights)], [cdf[ix_cross]], 'o', c=cmap(ilayer), markerfacecolor='none')
        pass
    
    # Uniform distribution reference line
    ax.plot([0] + list(cdf) + [1], [0] + list(cdf) + [1], c='gray', lw=1, label='Uniform distrib.')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Fraction of weights $w_{ij}$")
    ax.set_ylabel("Cumulative distribution function (c.d.f.) of $|w_{ij}|$")
    ax.legend()
    fig.suptitle(title)
    
    return fig



# ==============================================================================
# Images
# ------------------------------------------------------------------------------

def image (img, size=4, title=None, cmap='gray', xlabel=None, ylabel=None, 
           xticks=None, yticks=None, meshgrid=None, colorbar=False, origin='upper',
           symmetric=False, vmin=None, vmax=None, ax=None):
    """
    Display image(s).
    
    Display a single image, or a list of images in a square grid.
    
    Arguments:
        img: numpy.array or list-like. If `img` is deemed to be a single image, 
            it is drawn; if it is deemed to be a list or array of images, the 
            method will be called recursively to draw each image in a grid-
            square in a collective figure.
        size: Float, size of image(s).
        title: String, title to be drawn over the figure.
        cmap: String or pyplot colormap instance, determining the colour of the 
            image(s).
        xlabel: String, label to be draw on x-axis/-es.
        ylabel: String, label to be draw on y-axis/-es.
        xticks: Array-like, list of ticks to draw on x-axis/-es.
        yticks: Array-like, list of ticks to draw on y-axis/-es.
        meshgrid: numpy.array, like output of np.meshgrid, indicting x- and y-
            axis coordinate for each pixel im `img`. If not specified, integer 
            ranges are assumed.
        colorbar: String or boolean; whether to draw colorbar. Additionally, if 
            argument is a string, this is used as the title on the colorbar.
        origin: String, argument provided to pyplot.imshow.
        symmetric: Boolean, whether to make colour-axis symmetric around 0.
        vmin: Float, minimal value along colour-axis.
        vmax: Float, maximal value along colour-axis.
        ax: pyplot.Axis instance. If specified, the image will be drawn on this; 
            otherwise, a new pyplot.Axis object will be created.
    
    Returns:
        pyplot.Figure containing the image(s).
    """
    
    # Make a dictionary of all non-essential keywork arguments, for recurrent calls.
    kwargs = dict(size=size, cmap=cmap, xlabel=xlabel, ylabel=ylabel, 
                  xticks=xticks, yticks=yticks, meshgrid=meshgrid, colorbar=colorbar, origin=origin,
                  symmetric=symmetric, vmin=vmin, vmax=vmax)
    
    # If input is a list
    if isinstance(img, (list, tuple)):
        dim = int(_np.ceil(_np.sqrt(len(img))))
        if dim == 1:  # List of 1 image
            return image(img[0], ax=ax, title=title, **kwargs)
        if len(img) > 100:
            print(f"Requesting plot of {len(img)} images, which seems excessive.")
            return
        fig, _ = _plt.subplots(figsize=(dim*size//2, dim*size//2), sharex=True, sharey=True)
        for ix, p in enumerate(img, start=1):
            image(p, ax=_plt.subplot(dim, dim, ix), **kwargs)
            pass
  
    # If input is an array of images
    elif len(_np.squeeze(img).shape) > 2:
        return image([layer for layer in img], ax=ax, title=title, **kwargs)
     
    # Otherwise, assume input is an image
    else:
        xsize = size * (1.09 if colorbar else 1.0)
        ysize = size
        
        ax  = ax or _plt.subplots(figsize=(xsize,ysize))[1]
        fig = ax.figure 

        if symmetric:
            if None not in [vmin, vmax]:
                print("image: Requesting `symmetric`, but both `vmin` and `vmax` set.")
            elif vmin is None and vmax is None:
                vmax =  _np.max(_np.abs(img))
                vmin = -vmax
            elif vmin is None:
                vmin = -vmax
            else:
                vmax = -vmin
                pass
            pass
        
        if meshgrid is None:
            i = ax.imshow(img.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
        else:
            i = ax.pcolor(*(list(meshgrid) + [img.squeeze()]), cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Turn on axis tick by default if meshgrid provided
            if xticks is None:
                xticks = True
                pass
            if yticks is None:
                yticks = True
                pass
            pass
        
        # Axes
        if not xticks:
            ax.set_xticks([])
            pass
        
        if not yticks:
            ax.set_yticks([])
            pass
        
        if xlabel:
            ax.set_xlabel(xlabel)
            pass
        
        if ylabel:
            ax.set_ylabel(ylabel)
            pass
        
        # Colorbar
        if colorbar:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
            cbar = fig.colorbar(i, cax=cax)
            if isinstance(colorbar, str):
                cbar.set_label(colorbar)
                pass
            pass
        pass
    
    # (Opt.) Decorations
    if title:
        fig.suptitle(title, fontsize=16)
        pass
    
    fig.tight_layout(rect=[0.025, 0.0, 0.975, 0.95])
    return fig
