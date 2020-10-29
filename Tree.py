from numpy import ndarray


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

    # sep_data: tuple[float, float, int, int]
    def build_tree(self, data: ndarray, free_vars: set, sep_data: tuple) -> Node:
        data = data[data[:, sep_data[3]].argsort(kind='quicksort')]
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
        return data[:, -1].mean() > self.avQuality

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
        left_gini: float
        if left_total == 0:
            left_gini = 0  # if no elements happened to be on either side, then it is a perfect separation, gini = 0
        else:
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
