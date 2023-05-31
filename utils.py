import numpy as np
import itertools
from scipy.optimize import linprog
from tqdm import tqdm
from joblib import Parallel, delayed


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, desc='', *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, leave=False, desc=self._desc) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def get_hamming_7_4():
    """
    Hardcoded [7,4] Hamming code gen and pcm matrices. Gen matrix is in the systematic form
    """
    G = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0],
                  [0, 1, 1, 1]])
    H = np.array([[0, 1, 0, 0, 1, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1],
                  [1, 0, 0, 1, 0, 1, 1]])
    iw_pos = np.arange(4)
    return G, H, iw_pos


def bpsk_mod(x):
    return 1 - 2*x


def es_to_sigma(es):
    return np.sqrt(1/(2*10**(es/10)))


def feldmanInequalities(hmat, fundamentalCone=False):
    """
    Generate LP constraints. Source link: https://github.com/supermihi/lpdec/blob/master/lpdec/polytopes.py

    Compute the forbidden set inequalities for the binary control matrix *matrix*. They are
    returned by means of two *A* (dimension 2) and *b* (dimension 1) such
    that Ax <= b describes the constraints.

    If *fundamentalCone* is True, only the inequalities adjacent to the origin are generated.
    """
    if fundamentalCone:
        numConstraints = np.sum(hmat)
    else:
        numConstraints = 0
        for s in hmat.sum(1):
            numConstraints += 2 ** (s - 1)
    A = np.zeros((numConstraints, hmat.shape[1]), dtype=int)
    b = np.empty((numConstraints,), dtype=int)
    k = 0
    for row in hmat:
        N_j = np.flatnonzero(row)
        if fundamentalCone:
            maxS = 2
        else:
            maxS = len(N_j) + 1
        for s in range(1, maxS, 2):
            for subset in itertools.combinations(N_j, s):
                newRowA = np.zeros(hmat.shape[1])
                for i in N_j:
                    A[k, i] = -1
                for i in subset:
                    A[k, i] = 1
                b[k] = len(subset) - 1
                k += 1
    return A, b


def sim_lp_dec_step(G,H,iw_pos,snr,iter_num):
    N, k = G.shape
    # gen constraints
    A, b = feldmanInequalities(H)
    
    sigma = es_to_sigma(snr)
    nb_err_bits = 0
    nb_err_frames = 0

    for iter_idx in range(iter_num):
        iw = np.random.randint(low=0,high=2,size=k)
        cw = iw @ G.T % 2
        cw_mod = bpsk_mod(cw) + np.random.randn(N)*sigma
        res = linprog(2*cw_mod/sigma**2, A_ub=A, b_ub=b, bounds=(0,1))
        iw_out = (res.x[iw_pos] > 0.5).astype(int)
        nb_err_bits += np.abs(iw - iw_out).sum()
        nb_err_frames += np.abs((iw - iw_out)).any().astype(int)

    ber = nb_err_bits / iter_num / k
    fer = nb_err_frames / iter_num

    return ber, fer


def sim_ml_dec_step(G,snr,iter_num):
    N, k = G.shape
    iws = np.array(list(itertools.product([0, 1],repeat=k)))
    cws = iws @ G.T % 2
    cws_mod = bpsk_mod(cws)
    
    sigma = es_to_sigma(snr)
    nb_err_bits = 0
    nb_err_frames = 0

    for iter_idx in range(iter_num):
        iw_idx = np.random.randint(low=0,high=2**k)
        noisy_cw = cws_mod[iw_idx] + np.random.randn(N)*sigma
        est_iw_idx = np.argmin(np.linalg.norm((cws_mod - noisy_cw), ord=2, axis=1, keepdims=True))
        nb_err_bits += np.abs(iws[iw_idx] - iws[est_iw_idx]).sum()
        nb_err_frames += np.abs((iws[iw_idx] - iws[est_iw_idx])).any().astype(int)

    ber = nb_err_bits / iter_num / k
    fer = nb_err_frames / iter_num

    return ber, fer
