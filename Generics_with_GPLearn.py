import numpy as np
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def main():
    x = np.genfromtxt('x_train.csv', delimiter=',').reshape((1000, 1))
    y = np.genfromtxt('y_train.csv', delimiter=',')
    est_gp = SymbolicRegressor(population_size=50,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(x, y)
    print(est_gp._program)

    est_tree = DecisionTreeRegressor()
    est_tree.fit(x, y)
    est_rf = RandomForestRegressor()
    est_rf.fit(x, y)

    x0 = np.arange(-1, 1, 1 / 10.)
    x1 = np.arange(-1, 1, 1 / 10.)
    x0, x1 = np.meshgrid(x0, x1)
    y_truth = 3 * x0 ** 2 + 5 * x0 + 1  # exact function we are estimating

    y_gp = est_gp.predict(np.c_[x0.ravel()]).reshape(x0.shape)
    score_gp = est_gp.score(x, y)
    y_tree = est_tree.predict(np.c_[x0.ravel()]).reshape(x0.shape)
    score_tree = est_tree.score(x, y)
    y_rf = est_rf.predict(np.c_[x0.ravel()]).reshape(x0.shape)
    score_rf = est_rf.score(x, y)

    for i, (ys, score, title) in enumerate([(y_truth, None, "Ground Truth"),
                                           (y_gp, score_gp, "SymbolicRegressor"),
                                           (y_tree, score_tree, "DecisionTreeRegressor"),
                                           (y_rf, score_rf, "RandomForestRegressor")]):
        plt.subplot(2, 2, i+1)
        plt.plot(x0, ys, 'C0o')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
    plt.show()


main()
