"""
Benchmark: robust synthetic control.

Implementation taken from:
https://github.com/SucreRouge/synth_control
"""


from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from synth_functions import learn, swap


class Synth:
    def __init__(self, treat_unit, year, prior_param=0.5, method="linear", p=1, num_sv=1, drop=False, drop_list=None):
        if drop_list is None:
            drop_list = []
        self.treat_unit = treat_unit
        self.year = year
        self.p = p
        self.method = method
        self.num_sv = num_sv
        self.prior_param = prior_param
        self.drop = drop
        self.drop_list = drop_list
        self.beta = []
        self.mean = []
        self.orig = []

    def fit(self, df):
        data = df.copy()
        X = data.values
        X = X.astype(float)
        donor_list = list(data.index)

        # treated unit
        unit = donor_list.index(self.treat_unit)

        # let row zero represent the treatment unit
        X = swap(X, unit)
        self.orig = np.copy(X[0, :])
        self.Y = np.copy(X)

        # missing at random
        # X = MAR(X, self.p)

        # estimation
        self.beta, self.mean, self.sigma_hat = learn(
            X, self.year, num_sv=self.num_sv, method=self.method, prior_param=self.prior_param
        )

    def vis_data(
        self,
        xlabel="year",
        ylabel="metric",
        title="Case Study",
        orig_label="observed data",
        mean_label="counterfactual mean",
        year_shift=0,
        year_mod=5,
        loc="best",
        lw=1.75,
        frame_color="0.925",
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.orig_label = orig_label
        self.mean_label = mean_label
        self.year_shift = year_shift
        self.year_mod = year_mod
        self.loc = loc
        self.lw = lw
        self.frame_color = frame_color

    def vis(self, abadie):
        fig, ax = plt.subplots()
        ax.plot(self.orig, label=self.orig_label, linewidth=self.lw, color="g")
        ax.plot(self.mean, "--", label=self.mean_label, linewidth=self.lw, color="b")
        # ax.plot(abadie, '--', label="abadie", linewidth=self.lw, color='r')
        x_ = np.linspace(0, len(self.mean) - 1, len(self.mean))
        clr1 = "lightcyan"
        clr2 = "paleturquoise"
        upper = self.mean + self.sigma_hat
        lower = self.mean - self.sigma_hat
        ax.fill_between(x_, self.mean, upper, facecolor=clr1, edgecolor=clr2, interpolate=True)
        ax.fill_between(x_, self.mean, lower, facecolor=clr1, edgecolor=clr2, interpolate=True)
        legend = ax.legend(loc=self.loc, shadow=True, prop={"size": 9.5})
        frame = legend.get_frame()
        frame.set_facecolor(self.frame_color)
        ax.plot([self.year, self.year], [ax.get_ylim()[0], ax.get_ylim()[1]], "--", linewidth=self.lw, color="r")
        years = int(np.floor(self.orig.shape[0] / self.year_mod))
        x = np.array([self.year_mod * i for i in range(years + 1)])
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
        plt.xticks(x, x + self.year_shift)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.show()
        plt.close()
