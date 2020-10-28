from numpy import ndarray
from DataProvider import buildingSet


# https://www.youtube.com/watch?v=7VeUPuFGJHk


class Node:
    def __init__(self, threshold: float, split_var: int):
        self.threshold: float = threshold
        self.split_var: int = split_var
        self.left = None  # Node or Leaf
        self.right = None  # Node or Leaf


class Leaf:
    def __init__(self, is_good: bool):
        self.is_good = is_good


class DecisionTree:
    def __init__(self, data: ndarray):
        self.avQuality = data[:, -1].mean()
        free_vars: set = set(range(data.shape[1] - 1))
        node_data: tuple[float, float, int, int] = self.get_min_gini_column(data, free_vars)
        self.root = self.build_tree(data, free_vars, node_data)

    def get_min_gini_column(self, data: ndarray, free_vars: set):  # -> tuple[float, float, int, int]
        column_ginis: dict[int, tuple[float, float, int]] = {}  # gini, threshold, row_index
        for column in free_vars:
            column_ginis[column] = self.calc_var_gini(data, column)
        lowest_gini_column: int = min(column_ginis, key=column_ginis.get)
        free_vars.remove(lowest_gini_column)
        return column_ginis[lowest_gini_column] + (lowest_gini_column,)

    def build_tree(self, data: ndarray, free_vars: set,
                   sep_data: tuple) -> Node:  # sep_data: tuple[float, float, int, int]
        left_subset: ndarray = data[:sep_data[2], :]
        right_subset: ndarray = data[sep_data[2]:, :]
        res_node: Node = Node(sep_data[1], sep_data[3])

        if len(free_vars) == 1:
            res_node.left = Leaf(self.is_mostly_good(left_subset))
            res_node.right = Leaf(self.is_mostly_good(right_subset))
            return res_node

        left_free_vars: set = free_vars.copy()
        right_free_vars: set = free_vars.copy()

        left_sep_data: tuple[float, float, int, int] = self.get_min_gini_column(left_subset, left_free_vars)
        right_sep_data: tuple[float, float, int, int] = self.get_min_gini_column(right_subset, right_free_vars)

        if left_sep_data[0] > sep_data[0]:  # no use separating data to get clearer division
            res_node.left = Leaf(self.is_mostly_good(left_subset))
        else:
            res_node.left = self.build_tree(left_subset, left_free_vars, left_sep_data)

        if right_sep_data[0] > sep_data[0]:  # no use separating data to get clearer division
            res_node.right = Leaf(self.is_mostly_good(right_subset))
        else:
            res_node.right = self.build_tree(right_subset, right_free_vars, right_sep_data)
        return res_node

    def is_mostly_good(self, data: ndarray) -> bool:
        num_lt: int = 0
        num_gte: int = 0
        for case in data:
            if case[-1] < self.avQuality:
                num_lt += 1
            else:
                num_gte += 1
        return num_gte > num_lt

    # def is_mostly_good(self, data: ndarray) -> bool:
    #     return data[:, -1].mean() > self.avQuality

    # def check_purity(self, data: ndarray) -> tuple:
    #     num_lt: int = 0
    #     num_gte: int = 0
    #     for case in data:
    #         if case[-1] < self.avQuality:
    #             num_lt += 1
    #         else:
    #             num_gte += 1
    #     if 1 - (num_lt / (num_lt + num_gte)) ** 2 - (num_gte / (num_lt + num_gte)) ** 2 < 0.05:
    #         if num_lt > num_gte:
    #             return (True, -1)
    #         else:
    #             return (True, 1)
    #     else:
    #         return (False, 0)
    #
    # def build_tree(self, data: ndarray, free_vars: set) -> Node:
    #     if len(free_vars) == 0 or data.shape[0] == 1:  # buggy Leaf condition
    #         return Leaf('good' if data[:, -1].mean() > self.avQuality else 'bad')
    #     column_ginis: dict[int, tuple[float, float, int]] = {}
    #     for column in free_vars:
    #         column_ginis[column] = self.calc_var_gini(data, column)
    #     lowest_gini_column: int = min(column_ginis, key=column_ginis.get)
    #     free_vars.remove(lowest_gini_column)
    #     curr_node: Node = Node(column_ginis[lowest_gini_column][1], str(lowest_gini_column))
    #     left_data: ndarray = data[:column_ginis[lowest_gini_column][2], :]
    #     left_purity: tuple = self.check_purity(left_data)
    #     if left_purity[0]:
    #         curr_node.left = Leaf('good' if left_purity[1] == 1 else 'bad')
    #     else:
    #         curr_node.left = self.build_tree(left_data, free_vars.copy())
    #     right_data: ndarray = data[column_ginis[lowest_gini_column][2]:, :]
    #     right_purity: tuple = self.check_purity(right_data)
    #     if right_purity[0]:
    #         curr_node.right = Leaf('good' if right_purity[1] == 1 else 'bad')
    #     else:
    #         curr_node.right = self.build_tree(right_data, free_vars.copy())
    #     return curr_node

    def calc_var_gini(self, data: ndarray, column_ind: int) -> (float, float, int):
        data = data[data[:, column_ind].argsort(kind='quicksort')]
        column: ndarray = data[:, column_ind]
        curr_gini: float = 0.51  # worst case for initialisation
        row_index: int = -1
        threshold: float = -1

        for i in range(column.size - 1):
            adj_av = (column[i] + column[i + 1]) / 2
            next_gini = self.calc_threshold_gini(adj_av, data, column_ind)
            if next_gini < curr_gini:
                curr_gini = next_gini
                row_index = i
                threshold = adj_av
        return (curr_gini, threshold, row_index + 1)

    def calc_threshold_gini(self, adj_av: float, data: ndarray, column_ind: int) -> float:
        left_good_count: int = 0
        left_bad_count: int = 0
        row: int = 0  # left row

        while row < data.shape[0] and data[row, column_ind] <= adj_av:
            if data[row, -1] > self.avQuality:
                left_good_count += 1
            else:
                left_bad_count += 1
            row += 1
        left_total: int = left_good_count + left_bad_count
        left_gini: float = 1 - (left_good_count / left_total) ** 2 - (left_bad_count / left_total) ** 2

        right_good_count: int = 0
        right_bad_count: int = 0
        for r in range(row, data.shape[0]):
            if data[r, -1] > self.avQuality:
                right_good_count += 1
            else:
                right_bad_count += 1

        right_total: int = right_good_count + right_bad_count
        right_gini: float
        if right_total == 0:
            right_gini = 0  # if no elements happened to be on either side, then it is a perfect separation, gini = 0
        else:
            right_gini = 1 - (right_good_count / right_total) ** 2 - (right_bad_count / right_total) ** 2
        return left_gini * (row / data.shape[0]) + right_gini * ((data.shape[0] - row) / data.shape[0])
