import numpy
from numpy import ndarray
from Tree import DecisionTree, Node, Leaf

import DataProvider

# columns: set = set(range(12))
# print (columns)

# arr: ndarray = DataProvider.buildingSet
# print(arr.shape[1])
# print(type(arr[:, 1]))

# a: ndarray = numpy.array([[4, 6],
#                           [2, 5]])
# res = a[a[:, 1].argsort(kind='quicksort')]
# print(res)

# def func(test_d: dict):
#     test_d.clear()
#
#
# d: dict = {1: "abc", 2: "cde"}
#
# d_cpy: dict = d.copy()
# func(d.copy())
# func(d_cpy)
# print(d)
# print(d_cpy)

# def do_test(case: ndarray, node: Node) -> str:
#     if type(node) is Leaf:
#         return node.decision
#     if case[int(node.decision)] < node.threshold:
#         return do_test(case, node.left)
#     else:
#         return do_test(case, node.right)
#
#
# def test_predictions(testing_data: ndarray, d_tree: DecisionTree):
#     count_right: float = 0
#     count_wrong: float = 0
#     for case in testing_data:
#         correct_decision: str = 'good' if case[-1] > d_tree.avQuality else 'bad'
#         if do_test(case, d_tree.root) == correct_decision:
#             count_right += 1
#         else:
#             count_wrong += 1
#     print("% of right prediction: " + str(count_right / (count_right + count_wrong)))


arr: ndarray = DataProvider.buildingSet
# tree: DecisionTree = DecisionTree(arr)
# print(5)


# test_predictions(DataProvider.testingSet, tree)
