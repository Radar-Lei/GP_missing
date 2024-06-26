import numpy as _np
from GPy import kern
from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch
import GPy
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
from matplotlib import pyplot as plt

def _generate_high_dimensional_output(D1, D2, D3, s1, s2, s3, sS):
    S1 = _np.hstack([s1, sS])
    S2 = _np.hstack([sS])
    S3 = _np.hstack([s1, s3, sS])
    Y1 = S1.dot(_np.random.randn(S1.shape[1], D1))
    Y2 = S2.dot(_np.random.randn(S2.shape[1], D2))
    Y3 = S3.dot(_np.random.randn(S3.shape[1], D3))
    Y1 += 0.3 * _np.random.randn(*Y1.shape)
    Y2 += 0.2 * _np.random.randn(*Y2.shape)
    Y3 += 0.25 * _np.random.randn(*Y3.shape)
    Y1 -= Y1.mean(0)
    Y2 -= Y2.mean(0)
    Y3 -= Y3.mean(0)
    Y1 /= Y1.std(0)
    Y2 /= Y2.std(0)
    Y3 /= Y3.std(0)
    return Y1, Y2, Y3, S1, S2, S3


def _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim=False):
    """Simulate some data drawn from a matern covariance and a periodic exponential for use in MRD demos."""
    Q_signal = 4
    import GPy
    import numpy as np

    np.random.seed(3000)

    k = GPy.kern.Matern32(
        Q_signal, 1.0, lengthscale=(np.random.uniform(1, 6, Q_signal)), ARD=1
    )
    for i in range(Q_signal):
        k += GPy.kern.PeriodicExponential(
            1, variance=1.0, active_dims=[i], period=3.0, lower=-2, upper=6
        )
    t = np.c_[[np.linspace(-1, 5, N) for _ in range(Q_signal)]].T
    K = k.K(t)
    s2, s1, s3, sS = np.random.multivariate_normal(np.zeros(K.shape[0]), K, size=(4))[
        :, :, None
    ]

    Y1, Y2, Y3, S1, S2, S3 = _generate_high_dimensional_output(
        D1, D2, D3, s1, s2, s3, sS
    )

    slist = [sS, s1, s2, s3]
    slist_names = ["sS", "s1", "s2", "s3"]
    Ylist = [Y1, Y2, Y3]

    if plot_sim:
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        import itertools

        fig = plt.figure("MRD Simulation Data", figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        labls = slist_names
        for S, lab in zip(slist, labls):
            ax.plot(S, label=lab)
        ax.legend()
        for i, Y in enumerate(Ylist):
            ax = fig.add_subplot(2, len(Ylist), len(Ylist) + 1 + i)
            ax.imshow(Y, aspect="auto", cmap=cm.gray)  # @UndefinedVariable
            ax.set_title("Y{}".format(i + 1))
        plt.draw()
        plt.tight_layout()

    return slist, [S1, S2, S3], Ylist

def bgplvm_simulation_missing_data(
    optimize=True,
    verbose=1,
    plot=True,
    plot_sim=False,
    max_iters=2e4,
    percent_missing=0.1,
    d=13,
):

    D1, D2, D3, N, num_inducing, Q = d, 5, 8, 400, 3, 4
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)
    Y = Ylist[0]
    k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)

    inan = _np.random.binomial(1, percent_missing, size=Y.shape).astype(
        bool
    )  # 80% missing data
    Ymissing = Y.copy()
    Ymissing[inan] = _np.nan

    m = BayesianGPLVMMiniBatch(
        Ymissing,
        Q,
        init="random",
        num_inducing=num_inducing,
        kernel=k,
        missing_data=True,
    )

    m.Yreal = Y

    if optimize:
        print("Optimizing model:")
        m.optimize("bfgs", messages=verbose, max_iters=max_iters, gtol=0.05)
    if plot:
        m.X.plot("BGPLVM Latent Space 1D")
        m.kern.plot_ARD()
    return m

def bgplvm_simulation_missing_data_stochastics(
    optimize=True,
    verbose=1,
    plot=True,
    plot_sim=False,
    max_iters=2e4,
    percent_missing=0.1,
    d=1,
    batchsize=2,
):

    D1, D2, D3, N, num_inducing, Q = d, 5, 8, 400, 3, 1
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)
    Y = Ylist[0] # Y is of shape (L, K) 
    k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)

    inan = _np.random.binomial(1, percent_missing, size=Y.shape).astype(
        bool
    )  # 80% missing data
    Ymissing = Y.copy()
    Ymissing[inan] = _np.nan

    m = BayesianGPLVMMiniBatch(
        Ymissing,
        Q,
        init="random",
        num_inducing=num_inducing,
        kernel=k,
        missing_data=True,
        stochastic=True,
        batchsize=batchsize,
    )

    m.Yreal = Y

    if optimize:
        print("Optimizing model:")
        m.optimize("bfgs", messages=verbose, max_iters=max_iters, gtol=0.05)
    if plot:
        m.X.plot("BGPLVM Latent Space 1D")
        m.kern.plot_ARD()
    return m


if __name__ == "__main__":
    bgplvm_simulation_missing_data_stochastics()