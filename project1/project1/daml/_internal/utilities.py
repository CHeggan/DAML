# _internal/utilities.py: Collection of internal utility methods common to 
# several DAML CPs, but which the students should not need to interact with.
#
# Author: Andreas Sogaard <andreas.sogaard@ed.ac.uk>
# ------------------------------------------------------------------------------

# Import(s)
import numpy as _np
import sklearn as _sklearn
import tensorflow.python.keras as _keras


# ==============================================================================
# Neural network training
# ------------------------------------------------------------------------------

def _fit_sklearn (clf, X, y, validation_data=None, quiet=False):
    """
    Convenience wrapper around the `sklearn.neural_network.MLPClassifier.fit` 
    method. Decorates the input classifier with the attributes `history_loss` 
    and `history_acc`.
    """
    
    # Import(s)
    from sklearn.metrics import log_loss
    
    # Dictionary to store loss histories
    history_loss = {'train': [], 'val': []}
    history_acc  = {'train': [], 'val': []}

    # Check(s)
    X_train, y_train = X, y
    if validation_data is None:
        history_loss.pop('val')
        history_acc .pop('val')
    else:
        X_val, y_val = validation_data
        pass
    
    # Settings suitable for single-epoch-at-a-time training
    nb_epochs = clf.max_iter
    clf.max_iter   = 1
    clf.warm_start = True
    clf.verbose    = False

    # Loop epochs
    mult   = 1
    digits = 1 + max(int(_np.floor(_np.log10(nb_epochs))), 0)
    for epoch in range(1, nb_epochs + 1):
        
        # Logging
        if (not quiet) and (epoch % mult == 0):
            print("Epoch {{:{digits}d}}/{{:{digits}d}}".format(digits=digits).format(epoch, nb_epochs))
            if epoch == 10 * mult:
                mult *= 10
                pass
            pass
        
        # Fit next epoch
        clf.fit(X_train, y_train.ravel())
         
        # Store training loss computed *during* epoch
        history_loss['train'].append(clf.loss_)
        history_acc ['train'].append(clf.score(X_train, y_train))

        # (Opt.) Manually compute log-loss for validation data *after* epoch
        if validation_data is not None:
            history_loss['val'].append(log_loss(y_val, clf.predict_proba(X_val)))
            history_acc ['val'].append(clf.score(X_val, y_val))
            pass
        pass
    
    # Assign loss and accuracy histories as attribute
    clf.history_loss = history_loss
    clf.history_acc  = history_acc
    
    # Setting back max. number of iterations
    clf.max_iter = nb_epochs

    # Return loss history
    return clf


def _fit_keras (model, X_train, y_train, validation_data=None, quiet=False, **fit_kwargs):
    """
    Convenience wrapper around the `keras.models.Model.fit` method. The loss 
    history can be accessed from the `model.history` attribute, set by the Keras 
    fit method. Assumes the model has already been compiled.
    """
    
    # Just fit the thing
    model.fit(X_train, y_train, validation_data=validation_data, verbose=not quiet, **fit_kwargs)
    
    return model


def _fit (model, X, y, validation=None, quiet=False, **fit_kwargs):
    """
    Class-agnostic interface around the convenience fitting functions.
    """

    # Check(s): Make sure that training and validation data are properly formatted
    assert X.shape[0] == y.shape[0]

    X_train, y_train = X, y
    validation_data  = validation
    if isinstance(validation, float):
        assert validation > 0
        assert validation < 1
        X_train, X_val, y_train, y_val = _sklearn.model_selection.train_test_split(X, y, test_size=validation)
        if len(y_train.shape) == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_val   = y_val.ravel()
            pass
        validation_data = (X_val, y_val)
    elif isinstance(validation, (list, tuple)):
        assert len(validation) <= 3
        assert validation[0].shape[0] == validation[1].shape[0]
        if len(validation) == 3:
            assert validation[0].shape[0] == validation[2].shape[0]
            pass
    else:
        assert validation is None, "Validation data of type {} not understood.".format(type(validation))
    
    if   isinstance(model, _keras.models.Model):
        return _fit_keras  (model, X_train, y_train, validation_data, quiet, **fit_kwargs)
    elif isinstance(model, _sklearn.base.BaseEstimator):
        return _fit_sklearn(model, X_train, y_train, validation_data, quiet, **fit_kwargs)
    else:
        raise Exception(f"Model of type {type(model)} not understood.")
        
    # Shouldn't reach here
    return


def _fit_sklearn_cv (clf, X, y, nb_folds=5, quiet=False, **fit_kwargs):
    """
    Convenience wrapper around the `utilities.fit` method. Decorates the input 
    classifier with the attributes `cv_history_loss` and `cv_history_acc`.
    """
   
    kfold = _sklearn.model_selection.KFold(n_splits=nb_folds, shuffle=True)
    cv_history_loss = {'train': [], 'val': []}
    cv_history_acc  = {'train': [], 'val': []}
    for ifold, (ix_train, ix_val) in enumerate(kfold.split(X), start=1):

        # Logging
        if not quiet:
            print(f"CV fold {ifold}/{nb_folds}", end='')
            pass

        # Clone classifier (will be un-fitted even if the original classifier is)
        cv_clf = _sklearn.base.clone(clf)

        # Prepare CV datasets
        X_train = X[ix_train]
        y_train = y[ix_train]
        X_val   = X[ix_val]
        y_val   = y[ix_val]

        # Fit current CV fold
        _fit(cv_clf, X_train, y_train, validation=(X_val, y_val), quiet=True, **fit_kwargs)

        # Store CV loss histories
        cv_history_loss['train'].append(cv_clf.history_loss['train'])
        cv_history_loss['val'] .append(cv_clf.history_loss ['val'])

        # Score CV accuracies
        cv_history_acc['train'].append(cv_clf.history_acc['train'])
        cv_history_acc['val']  .append(cv_clf.history_acc['val'])

        # Logging (cont'd)
        if not quiet:
            print(" | Acc. (train/val.): {:4.1f}% / {:4.1f}%".format(cv_history_acc['train'][-1][-1] * 100.,
                                                                     cv_history_acc['val']  [-1][-1] * 100.))
            pass
        pass

    # Assign CV loss histories and score as attributes
    clf.cv_history_loss = cv_history_loss
    clf.cv_history_acc  = cv_history_acc
    
    return clf


def _fit_keras_cv (model, X, y, nb_folds=5, quiet=False, **fit_kwargs):
    """
    Convenience wrapper around the `utilities.fit` method. Overwrites the 
    `history` attribute with histories for each CV fold.
    """
    # Import(s)
    from sklearn.metrics import accuracy_score

    kfold = _sklearn.model_selection.KFold(n_splits=nb_folds, shuffle=True)
    histories = dict()
    for ifold, (ix_train, ix_val) in enumerate(kfold.split(X), start=1):

        # Logging
        if not quiet:
            print(f"CV fold {ifold}/{nb_folds}", end='')
            pass

        # Clone model (will be un-fitted even if the original classifier is)
        cv_model = _keras.models.clone_model(model)
        
        # Compile model
        cv_model.compile(model.optimizer, loss=model.loss, metrics=model.metrics_names[1:])

        # Prepare CV datasets
        X_train = X[ix_train]
        y_train = y[ix_train]
        X_val   = X[ix_val]
        y_val   = y[ix_val]

        # Fit current CV fold
        _fit(cv_model, X_train, y_train, validation=(X_val, y_val), quiet=True, **fit_kwargs)
        
        # Add CV history
        for key, val in cv_model.history.history.items():
            if key not in histories:
                histories[key] = list()
                pass
            histories[key] += [list(val)]
            pass

        # Logging (cont'd)
        if not quiet:
            p_train = _np.argmax(cv_model.predict(X_train), axis=1)
            p_val   = _np.argmax(cv_model.predict(X_val),   axis=1)
            print(" | Acc. (train/val.): {:4.1f}% / {:4.1f}%".format(accuracy_score(y_train, p_train) * 100.,
                                                                     accuracy_score(y_val,   p_val)   * 100.))
            pass
        pass

    # Assign CV histories as attribute
    if not hasattr(model, 'history'): # `model` hasn't been fitted before
        model.history = cv_model.history
        pass
    model.history.history = histories
    
    return model


def _fit_cv (model, X, y, nb_folds=5, quiet=False, **fit_kwargs):
    """
    Class-agnostic interface around the convenience cross-validation fitting functions.
    """
    
    print(f"Running {nb_folds}-fold cross-validation. Note: `model` weights will not be updated in this call. To train the model itself, run `utilities.fit` or similar method on a specific training dataset.")
    if   isinstance(model, _keras.models.Model):
        return _fit_keras_cv  (model, X, y, nb_folds, quiet, **fit_kwargs)
    elif isinstance(model, _sklearn.base.BaseEstimator):
        return _fit_sklearn_cv(model, X, y, nb_folds, quiet, **fit_kwargs)
    else:
        raise Exception(f"Model of type {type(model)} not understood.")
     
    return clf



# ==============================================================================
# Neural network architecture inference
# ------------------------------------------------------------------------------

def _get_architecture_from_pytorch (model):
    assert isinstance(model, torch.nn.Module)
    return map(lambda s: s.strip(), str(model).split('\n')[1:-1])


def _get_architecture_from_keras (model):
    assert isinstance(model, _keras.models.Model)
    arch = list()

    def get_dims(shape):
        # Format the layer shape dimensions neatly
        dims = shape[1:]
        if len(dims) == 1:
            dims = dims[0]
            pass
        return dims
    
    def get_activation(layer):
        # Access the layer activation conveniently
        return layer.activation._keras_api_names[0].split('.')[-1]
    
    for l in model.layers:
        if   isinstance(l, _keras.layers.InputLayer):
            assert len(l.input_shape) == 1  # Assuming single input (i.e. sequential)
            arch += [get_dims(l.input_shape[0])]
        elif isinstance(l, _keras.layers.Dense):
            arch += [get_dims(l.output_shape)]
            arch += [get_activation(l)]
        elif isinstance(l, _keras.layers.Dropout):
            arch += [f'Drop({l.rate:.2f})']
        elif isinstance(l, _keras.layers.BatchNormalization):
            arch += ['BatchNorm']
        elif isinstance(l, _keras.layers.Activation):
            arch += [l.activation._keras_api_names[0].split('.')[-1]]
        elif isinstance(l, _keras.layers.MaxPooling2D):
            arch += [u'MaxPool[{} → {}]'.format(l.pool_size, get_dims(l.output_shape))]
        elif isinstance(l, _keras.layers.Conv2D):
            arch += [u'Conv[{} → {}]'.format(l.kernel_size, get_dims(l.output_shape))]
            arch += [get_activation(l)]
        elif isinstance(l, _keras.layers.Flatten):
            arch += ['Flatten[{}]'.format(get_dims(l.output_shape))]
        elif isinstance(l, _keras.layers.Reshape):
            arch += ['Reshape[{}]'.format(get_dims(l.output_shape))]
        else:
            print("Error: Layer of type {} not recognised.".format(type(l)))
            pass
        pass
    return arch


def _get_architecture_from_sklearn (clf):
    assert isinstance(clf, _sklearn.neural_network.MLPClassifier)
    arch = list()
    for ix, coefs in enumerate(clf.coefs_):
        arch += [coefs.shape[0]]
        arch += [clf.activation] if ix > 0 else []
        pass
    arch += [clf.coefs_[-1].shape[1]]
    arch += [clf.out_activation_]
    return arch
