# optimisation.py: Classes relevant to Bayesian hyperparameter optimisation.
#
# Author: Andreas Sogaard <andreas.sogaard@ed.ac.uk>
#
# In this file you will find a few class definitions related to Bayesian 
# optimisation. The `BayesianOptimiser` class is rather crude, meant only to 
# illustrate some central concepts related to non--grid-search-based 
# optimisation. Don't use it for anything real; then you'd be better of with 
# packages such as spearmint, scikit-optimize, or hyperopt.
# ------------------------------------------------------------------------------

# Import(s)
import numpy as _np


class Trial:
    """
    Class for logging result of Bayesian optimisation evaluation trials.

    Arguments:
        site: Scalar or array-like, the parameter configuration.
        value: Float, the evaluation result.
        error: Float, the uncertainty on the evaluation result. If not set, an
            uncertainty of ~zero is used.
    """
    def __init__ (self, site, value, error=None):
        
        # Check(s)
        if _np.isscalar(site):
            site = [site]
            pass
        
        # Member variable(s)
        self.site  = tuple(site)
        self.value = value
        self.error = error or _np.sqrt(1.0E-10)
        return
    pass


class Dimension:
    """
    Class for specifying the parameter dimensions to be searched during the 
    Bayesian optimisation.

    Arguments:
        low: Float, low end of the search range.
        high: Float, high end of the search range.
        name: String, name of the parameter spanning this dimension.
        t: type-instance, specifying the type of the parameter spanning this
            dimensions.
        transform: String, either 'linear' or 'log', specifying whether this 
            dimension should be sampled on a linear or logarithmic basis. (Good 
            for scanning parameter potentially spanning many orders of magnitude.)
    """
    def __init__ (self, low, high, name=None, t=None, transform='linear'):
       
        # Check(s)
        assert transform in ['linear', 'log']
        assert transform == 'linear', "Log-transform not supported yet."
        assert low <= high
        
        # Member variable(s)
        self.low  = low
        self.high = high
        self.name = name
        self.type = t or type(low)
        assert self.type == type(high), "Type mis-match."
        self.transform = transform
        return
    pass


class _ConvergenceException(Exception):
    """
    Custom exception, for when the Bayesian optimisation has converged.
    """
    pass
        
    
class BayesianOptimiser:
    """
    Class for performing Bayesian optimisation.

    Arguments:
        objective: Function, taking as input a tuple of parameter values, 
            and returning either a float `value`, interpreted as the 
            function value of objective function for the given parameter 
            configuration; or a tuple of `(value, error)`, where the second 
            element is interpreted sa the uncertainty on the value of the 
            objective function for the given parameter configuration.
        dimensions: Array-like, instances of `optimisation.Dimension`, 
            specifying the parameter dimensions to be searched. The number 
            of `dimensions` should be the same as the length of the input 
            tuple expected by `objective`.
    """
    def __init__ (self, objective, dimensions):
        
        # Checks(s)
        if isinstance(dimensions, Dimension):
            dimensions = [dimensions]
            pass
        assert all([isinstance(d, Dimension) for d in dimensions]), \
            "Elements of argument `dimensions` should be of type `optimisation.Dimension`."
        
       # Member variable(s)
        self.objective  = objective
        self.dimensions = dimensions
        self.nb_dims    = len(dimensions)

        # Measurement trials
        self.trials = list()
        
        # Create optimisation mesh
        # @TODO: Log?
        grid_axes = list()
        for ix, d in enumerate(self.dimensions):
            if d.type == int:
                axis = _np.arange(d.low, d.high + 1)
            else:
                axis = _np.linspace(d.low, d.high, 200 + 1, endpoint=True)
                pass
            grid_axes.append(axis)
            pass
        
        self.mesh    = _np.meshgrid(*grid_axes)
        self.X       = _np.stack([a.flatten() for a in self.mesh]).T
        self.options = list(map(tuple, self.X))
        
         # Create Gaussian process regressor
        self.reset_gp()
        return
    
    def reset_gp (self, min_lengthscale_frac=0.1, max_lengthscale_frac=1.0):
        
        # Import(s)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        
        # Measurement variances
        alpha = _np.asarray([t.error**2 for t in self.trials])
        
        # Squared-exponential kernel with sensible bounds
        length_scales       = [1. for _ in self.dimensions]
        length_scale_bounds = [((d.high - d.low) * min_lengthscale_frac, 
                                (d.high - d.low) * max_lengthscale_frac) for d in self.dimensions]
        kernel = C(1.0, (0.01, 10.)) * RBF(length_scales, length_scale_bounds)
 
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=alpha)
        return 

    def expected_improvement (self, X, xi=0.01):
        # Import(s)
        from scipy.stats import norm
        
        mu, sigma = self.gp.predict(X, return_std=True)
        fmax      = max([t.value for t in self.trials]) if len(self.trials) else 0
        sigma = _np.clip(sigma, 1.0E-03, None)
        
        Z  = (mu - fmax - xi) / sigma
        EI = (mu - fmax - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return EI
    
    def acquisition_function (self, X, **params):
        return self.expected_improvement(X, **params)
        
    def get_random (self):
        if len(self.options) == 0:
            raise _ConvergenceException("No new (random) configurations to try.")
        ix = _np.random.choice(len(self.options))
        return self.options[ix]
        
    def get_suggestion (self, **acq_params):
        EI = self.acquisition_function(self.X, **acq_params) 
        ix = _np.argmax(EI)
        site = tuple([d.type(v) for v,d in zip(self.X[ix], self.dimensions)])
        if site in self.sites():
            raise _ConvergenceException("Best suggestion already tried.")
        return site
    
    def get_best (self):
        pred = self.gp.predict(self.X)
        ix = _np.argmax(pred)
        x  = tuple([d.type(v) for v,d in zip(self.X[ix], self.dimensions)])
        return x
    
    def add_measurement (self, site, value, error=None):
        trial = Trial(site, value, error)
        self.options.remove(tuple(site))
        self.trials.append(trial)
        self.reset_gp()
        self.update()
        return
    
    def sites (self):
        return [t.site for t in self.trials]
    
    def values (self):
        return [t.value for t in self.trials]
    
    def errors (self):
        return [t.error for t in self.trials]
    
    def update (self):
        x = _np.asarray(self.sites())
        y = _np.asarray(self.values())
      
        self.gp.fit(x,y)
        return
  
    def fit (self, nb_total=10, nb_random=5):
        """
        Run bayesian optimisation.
        
        First, the objective function is evaluated `nb_random` times at random 
        parameter sites. Then, Gaussian process regression is used to estimate 
        the objective function and to compute the expected improvement (EI) an 
        un-tested parameter sites. The parameter site with the maximal EI is 
        then evaluated sequentially for `nb_total - nb_random` times.
        
        Arguments:
            self: Calling instance.
            nb_total: Integer, total number of evaluations to run.
            nb_random: Integer, number of random initialisations to run. This 
                number should not be greater than `nb_total`.
        
        Raises:
            AssertionError, if the requested number of trials are inconsistent.
        """
        
        # Check(s)
        assert nb_random <= nb_total

        for it in range(nb_total):
            # Get set site to test
            random = it < nb_random
            try:
                if random:
                    site = self.get_random()
                else:
                    site = self.get_suggestion()
                    pass
            except _ConvergenceException as e:
                print("Reached convergence criterion: {} Exiting.".format(str(e)))
                return
            
            print(f"Trial {it+1:2d}/{nb_total:2d}: Sampling {site}%s" % (" (random)" if random else ""))

            # Get values of objective function
            ret   = self.objective(site)
            if isinstance(ret, tuple):
                assert len(ret) == 2
                value, error = ret
                print(f"  Got objective value {value:.3f} ± {error:.3f}")
            else:
                value, error = ret, None
                print(f"  Got objective value {value:.3f}")
                pass
            
                  
            # Add to list of measurements
            self.add_measurement(site, value, error)
            pass
        
        return
    pass
