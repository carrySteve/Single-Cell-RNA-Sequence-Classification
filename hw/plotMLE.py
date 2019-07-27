import sys
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

__EXPECTED_MU = 0
__EXPECTED_SIGMA = 1
__SAMPLE_NUM = 1000


def plotMLE(x_candidate, theta_candidate):
    result = []

    for mu, sigma in theta_candidate:
        log_likelihood = 0
        for x in x_candidate:
            prob = stats.norm.pdf(x, mu, sigma)
            log_prob = math.log(prob)
            log_likelihood += log_prob
        result.append(log_likelihood)

    max_idx = result.index(max(result))
    print(max_idx)

    plt.plot([str(theta) for theta in theta_candidate],
             result,
             color='blue',
             linewidth=1.0,
             linestyle='--')
    plt.annotate(s=theta_candidate[max_idx], xy=(max_idx, result[max_idx]))
    plt.title('MLE Graph')
    plt.xlabel(r'($\mu, \sigma$)')
    plt.ylabel(r'L($\theta$)')
    plt.show()


if __name__ == "__main__":
    theta_candidate = [(0, 1), (0, 2), (1, 1), (1, 2)]

    np.random.seed(0)
    x_candidate = np.random.normal(__EXPECTED_MU, __EXPECTED_SIGMA,
                                   __SAMPLE_NUM).tolist()

    plotMLE(x_candidate, theta_candidate)
