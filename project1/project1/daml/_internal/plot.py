# _internal/plot.py: Collection of internal plotting methods common to several 
# DAML CPs, but which the students should not need to interact with.
#
# Author: Andreas Sogaard <andreas.sogaard@ed.ac.uk>
# ------------------------------------------------------------------------------

# Import(s)
import numpy as _np
import matplotlib.pyplot as _plt
import tensorflow.python.keras as _keras

# Local import(s)
from .. import utilities as _utilities


# ==============================================================================
# Training loss
# ------------------------------------------------------------------------------

def _history_from_keras (hist):
    
    if isinstance(hist, _keras.models.Model):
        model = hist
        assert hasattr(model, 'history'), \
            "Provided Keras model argument with no 'history' attribute. Please call `fit` first."
        return _history_from_keras(model.history)
    
    # Format dictionary of losses from original object
    h = dict(**hist.history)
    for old, new in [('loss',     'train'), 
                     ('val_loss', 'val')]:
        if old in h:
            h[new] = h.pop(old)
            pass
        pass
    
    # Set title
    h['title'] = _utilities.get_network_name(hist.model)
                                       
    return h


def _history_from_sklearn (clf):
    
    # Format dictionary of losses from original object
    if hasattr(clf, 'cv_history_loss'):
        h = dict(**clf.cv_history_loss)
    elif hasattr(clf, 'history_loss'):
        h = dict(**clf.history_loss)
    else:
        print("loss: Please train with `utilities.fit_mlp{,_cv}` to also plot validation losses.")
        h = {'train': list(clf.loss_curve_)}
        pass
    
    # Set title
    h['title'] = _utilities.get_network_name(clf)

    return h
    
    
def _loss_single (h, ax, title='', colour=None, minimum=False, cv_band=None, max_yval=2):
    
    # Check(s)
    assert 'train' in h, "Key 'train' not in {}".format(h.keys())

    val = 'val' in h
    cv  = len(_np.shape(h['train'])) == 2
    x   = _np.arange(_np.asarray(h['train']).shape[-1])

    # Default
    if cv_band is None:
        cv_band = cv
        pass        

    # Compute mean/std
    if cv:
        m_train = _np.mean(h['train'], axis=0)
        s_train = _np.std (h['train'], axis=0)
        m_val   = _np.mean(h['val'], axis=0) 
        s_val   = _np.std (h['val'], axis=0)
    else:
        m_train = h['train']
        m_val   = h['val'] if val else None
        s_train = None
        s_val   = None
        pass
    
    # -- Training loss curve
    fillopts = dict(color=colour, alpha=0.1)
    ax.plot(m_train, ls='--', color=colour)
    if cv and cv_band:
        ax.fill_between(x, m_train - s_train, m_train + s_train, **fillopts)
        pass
    
    if val:
        # -- Validation loss curve
        ax.plot(m_val, ls='-', color=colour)
        if cv and cv_band:
            ax.fill_between(x, m_val - s_val, m_val + s_val, **fillopts)
        else:
            if cv_band:
                print("loss: Requested CV band, but CV training hasn't been run.")
                pass
            ax.fill_between(x, m_train, m_val, **fillopts)
            pass
        pass
    
    # Optimisation minimum for training curve
    if minimum:
        ixMin = _np.argmin(m_val if val else m_train)
        xmin  = x[ixMin]
        minTrain = m_train[ixMin]

        ax.plot([xmin], [minTrain], 'o', color=colour, markerfacecolor='none')

        # Optimisation minimum for validation curve
        if val:
            minVal = m_val[ixMin]
            ax.plot([xmin, xmin], [minVal, minTrain], color=colour, ls=':', lw=1)
            ax.plot([xmin], [minVal], 'o', color=colour)
            pass
        pass
    
    if max_yval:
        ymax = _np.max(m_train)
        if val:
            ymax = max(ymax, _np.max(m_val))
            pass
        if ymax > max_yval:
            ax.set_ylim(None, max_yval)
            pass
        pass
    
    return val, cv


def _loss (*histories, **kwargs):
   
    cmap = _utilities.get_cmap(len(histories))

    single   = len(histories) == 1
    multiple = len(histories) >  1
    
    # Create figure
    fig, ax = _plt.subplots()
    vals, cvs = list(), list()
    for ix, h in enumerate(histories):
        colour = cmap(ix) if multiple else 'black'
        val, cv = _loss_single(h, ax, colour=colour, minimum=kwargs['minima'], cv_band=kwargs['cv'])

        vals.append(val)
        cvs .append(cv)
        
        if multiple:
            ax.plot([], [], color=colour, label=h.get('title', f"History {ix+1}"))
            pass
        pass
    
    # Manually add entries
    # -- Training/validation curves
    _plt.plot([], [], ls='--', color='black', label='Train')
    if any(vals):
        _plt.plot([], [], ls='-',  color='black', label='Validation')
        pass
    
    # -- Loss minima
    if kwargs['minima']:
        ax.plot([], [], 'o', color='black', markerfacecolor='none', 
                label='Minima' if any(vals) else 'Minimum')
        pass
    
    # -- CV band
    handles, labels = ax.get_legend_handles_labels()
    if any(cvs) and kwargs['cv'] != False:
        line = ax.plot([], [], color='black')
        fill = ax.fill(_np.NaN, _np.NaN, color='black', alpha=0.1, lw=0)
        handles += [(line[0], fill[0])]
        labels  += [u'µ(x) ± σ(x) (CV)']
        pass
    
    # Decorations
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss function")
    ax.set_yscale(kwargs['scale'])

    return fig



class Interpolator:
    """
    Interpolate unspecified features using k-nearest neighbour regression.
    """
    def __init__ (self, base, n_neighbors=20):
        
        # Import(s)
        from sklearn.neighbors import KNeighborsRegressor
        
        self.base    = base
        self.knn     = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        self.indices = list()
        return

    def fit (self, X_full, X_pair):
        # Determine which columns the subset corresponds to
        for column in X_pair.T:
            differences = _np.abs(X_full - column.reshape((-1,1))).mean(axis=0)
            indices = _np.where(differences == 0)[0]
            assert len(indices) == 1
            self.indices.append(indices[0])
            pass

        # Use k-NN regression to estimate the mean values of the other
        # parameters for each subset parameter configuration, i.e. learn the
        # mean of the conditional p.d.f. p(X_full|X_pair).
        self.knn.fit(X_pair, X_full)
        return self

    def _knn_predict (self, X_pair):
        # Estimate the mean of the full parameter configuration, but insert the
        # actual values of the feature pair currently being scanned over.
        X_full_est = self.knn.predict(X_pair)
        for ix, column in zip(self.indices, X_pair.T):
            X_full_est[:,ix] = column
            pass
        return X_full_est

    def predict_proba (self, X_pair):
        X_full_est = self._knn_predict(X_pair)
        return self.base.predict_proba(X_full_est)

    def predict (self, X_pair):
        X_full_est = self._knn_predict(X_pair)
        return self.base.predict(X_full_est)

    pass
