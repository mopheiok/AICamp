import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from trees.dt import DecisionTree


def test_DecisionTree(N=1):
    i = 1
    np.random.seed(42)
    while i <= N:
        n_ex = np.random.randint(2, 100)
        n_feats = np.random.randint(2, 100)
        max_depth = np.random.randint(1, 5)

        classifier = np.random.choice([True, False])
        if classifier:
            # create classification problem
            n_classes = np.random.randint(2, 10)
            X, Y = make_blobs(
                n_samples=n_ex, centers=n_classes, n_features=n_feats, random_state=i
            )
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            def loss(yp, y):
                return 1 - accuracy_score(yp, y)

            criterion = np.random.choice(["entropy", "gini"])
            mine = DecisionTree(
                classifier=classifier, max_depth=max_depth, criterion=criterion
            )
            gold = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                splitter="best",
                random_state=i,
            )
        else:
            # create regression problem
            X, Y = make_regression(n_samples=n_ex, n_features=n_feats, random_state=i)
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            criterion = "mse"
            loss = mean_squared_error
            mine = DecisionTree(
                criterion=criterion, max_depth=max_depth, classifier=classifier
            )
            gold = DecisionTreeRegressor(
                criterion=criterion, max_depth=max_depth, splitter="best"
            )

        print("Trial {}".format(i))
        print("\tClassifier={}, criterion={}".format(classifier, criterion))
        print("\tmax_depth={}, n_feats={}, n_ex={}".format(max_depth, n_feats, n_ex))
        if classifier:
            print("\tn_classes: {}".format(n_classes))

        # fit 'em
        mine.fit(X, Y)
        gold.fit(X, Y)

        # get preds on training set
        y_pred_mine = mine.predict(X)
        y_pred_gold = gold.predict(X)

        loss_mine = loss(y_pred_mine, Y)
        loss_gold = loss(y_pred_gold, Y)

        # get preds on test set
        y_pred_mine_test = mine.predict(X_test)
        y_pred_gold_test = gold.predict(X_test)

        loss_mine_test = loss(y_pred_mine_test, Y_test)
        loss_gold_test = loss(y_pred_gold_test, Y_test)

        try:
            np.testing.assert_almost_equal(loss_mine, loss_gold)
            print("\tLoss on training: {}".format(loss_mine))
        except AssertionError as e:
            print("\tTraining losses not equal:\n{}".format(e))

        try:
            np.testing.assert_almost_equal(loss_mine_test, loss_gold_test)
            print("\tLoss on test: {}".format(loss_mine_test))
        except AssertionError as e:
            print("\tTest losses not equal:\n{}".format(e))
        i += 1
