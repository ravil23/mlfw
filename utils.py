# -*- coding: utf-8 -*-

FIGURE_WIDTH = 27
PLOT_HEIGHT = 8


def plot_dataset(X_train, X_test, y_train, y_test, classifiers=None):

    import numpy as np
    from matplotlib import pyplot as plt, colors

    figure = plt.figure(figsize=(FIGURE_WIDTH, PLOT_HEIGHT))
    cm_bright = colors.ListedColormap(['#FF0000', '#0000FF'])

    X_min_lim = 1.1 * np.minimum(X_train.min(axis=0), X_test.min(axis=0))
    X_max_lim = 1.1 * np.maximum(X_train.max(axis=0), X_test.max(axis=0))

    def get_axis(i):
        ax = plt.subplot(1, 2, i)
        ax.set_xlim(X_min_lim[0], X_max_lim[0])
        ax.set_ylim(X_min_lim[1], X_max_lim[1])
        return ax

    ax = get_axis(1)
    ax.set_title('Train set')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

    ax = get_axis(2)
    ax.set_title('Test set')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')

    plt.show()


def plot_classification(X_train, X_test, y_train, y_test, classifiers):

    import numpy as np
    from matplotlib import pyplot as plt, colors

    figure = plt.figure(figsize=(FIGURE_WIDTH, PLOT_HEIGHT * len(classifiers)))
    cm = plt.cm.RdBu
    cm_bright = colors.ListedColormap(['#FF0000', '#0000FF'])

    X_min_lim = 1.1 * np.minimum(X_train.min(axis=0), X_test.min(axis=0))
    X_max_lim = 1.1 * np.maximum(X_train.max(axis=0), X_test.max(axis=0))

    def get_axis(i):
        ax = plt.subplot(len(classifiers), 2, i)
        ax.set_xlim(X_min_lim[0], X_max_lim[0])
        ax.set_ylim(X_min_lim[1], X_max_lim[1])
        return ax

    X_mesh, y_mesh = np.meshgrid(
        np.arange(X_min_lim[0], X_max_lim[0], (X_max_lim[0] - X_min_lim[0]) / 100.),
        np.arange(X_min_lim[1], X_max_lim[1], (X_max_lim[1] - X_min_lim[1]) / 100.)
    )

    i = 0
    for name, clf in classifiers.items():
        score_train = clf.score(X_train, y_train)
        score_test = clf.score(X_test, y_test)
        
        if hasattr(clf, 'decision_function'):
            Z = clf.decision_function(np.c_[X_mesh.ravel(), y_mesh.ravel()])
        else:
            Z = clf.predict_proba(np.c_[X_mesh.ravel(), y_mesh.ravel()])[:, 1]

        Z = Z.reshape(X_mesh.shape)
        
        i += 1
        ax = get_axis(i)
        ax.set_ylabel(name)
        ax.set_title('Train accuracy: %.3f' % score_train)
        ax.contourf(X_mesh, y_mesh, Z, cmap=cm, alpha=.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

        i += 1
        ax = get_axis(i)
        ax.set_title('Test accuracy: %.3f' % score_test)
        ax.contourf(X_mesh, y_mesh, Z, cmap=cm, alpha=.8)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')

    plt.show()


def make_binary_classification_dataset(n_samples=1000, random_state=3, test_size=0.3):

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        weights=(0.5, 0.5),
        random_state=random_state,
    )

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
