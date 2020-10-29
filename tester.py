from numpy import ndarray

import DataProvider
from Tree import DecisionTree, Leaf


def predict(case: ndarray, node) -> bool:
    if type(node) is Leaf:
        return node.is_good
    if case[node.split_var] < node.threshold:
        return predict(case, node.left)
    else:
        return predict(case, node.right)


def test_predictions(testing_data: ndarray, d_tree: DecisionTree) -> float:
    count_right: float = 0
    count_wrong: float = 0
    for case in testing_data:
        correct = case[-1] > d_tree.avQuality
        if predict(case, d_tree.root) == correct:
            count_right += 1
        else:
            count_wrong += 1
    return count_right / (count_right + count_wrong)


tree: DecisionTree = DecisionTree(DataProvider.buildingSet)

percent_correct = test_predictions(DataProvider.testingSet, tree)
print("% of right prediction: " + str(percent_correct * 100))
