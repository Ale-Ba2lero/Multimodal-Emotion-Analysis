import torch

torch.set_default_dtype(torch.float64)  # double precision for numerical stability
import matplotlib.pyplot as plt
from util.search_inference import HashingMarginal, memoize, Search
import numpy as np


def marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))


def plot_dist(d) -> None:
    support = d.enumerate_support()
    data = [d.log_prob(s).exp().item() for s in d.enumerate_support()]
    names = support

    ax = plt.subplot(111)
    width = 0.3
    bins = list(map(lambda x: x - width / 2, range(1, len(data) + 1)))
    ax.bar(bins, data, width=width)
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.tick_params(axis="both", which="minor", labelsize=6)
    ax.set_xticks(list(map(lambda x: x, range(1, len(data) + 1))))
    ax.set_xticklabels(names, rotation=30, rotation_mode="anchor", ha="right")

    plt.show()


def get_marginals(listenerPosterior):
    '''
    Given the joint posterior distribution of trates and phi
    from the Pragmatic Listner, this function computes the marginal
    distributions (probability of states and phi, separately) and
    computes their expected value
    '''
    supp_matrix = np.array(listenerPosterior.enumerate_support())
    states = np.unique(supp_matrix[:, 0])
    phis = np.unique(supp_matrix[:, 1])
    n_states = len(states)
    n_phi = len(phis)
    posterior_matrix = np.zeros([n_states, n_phi])
    for i in range(n_states):
        for j in range(n_phi):
            s = tuple(supp_matrix[i * n_phi + j, :])
            curr_p = listenerPosterior.log_prob(s).exp().item()
            posterior_matrix[i, j] = curr_p
    marginal_state = posterior_matrix.sum(axis=1)
    marginal_phi = posterior_matrix.sum(axis=0)

    m_state = {'support': states,
               'marginal': marginal_state,
               'ev': np.round(np.sum(np.multiply(marginal_state, states)), decimals=2)}
    m_phi = {'support': phis,
             'marginal': marginal_phi,
             'ev': np.round(np.sum(np.multiply(marginal_phi, phis)), decimals=2)}

    return m_state, m_phi
