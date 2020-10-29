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

    def get_min_gini_column(self, data: ndarray, free_vars: set) -> (float, float, int, int):
        column_ginis: dict[int, tuple[float, float, int]] = {}  # gini, threshold, row_index, column_index
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

        # for testing data_set copying free_vars makes predictions less accurate,
        # but according to algorithm variables can be used for separation in both subtrees
        # starting from a given node
        res_node.left = self.build_side(left_subset, free_vars.copy(), sep_data)
        res_node.right = self.build_side(right_subset, free_vars.copy(), sep_data)
        return res_node

    def build_side(self, data_subset: ndarray, free_vars: set, sep_data: tuple):
        def is_mostly_good(data: ndarray) -> bool:
            return data[:, -1].mean() > self.avQuality

        if len(free_vars) == 1:
            return Leaf(is_mostly_good(data_subset))
        next_sep_data: tuple[float, float, int, int] = self.get_min_gini_column(data_subset, free_vars)
        return Leaf(is_mostly_good(data_subset)) if next_sep_data[0] > sep_data[0] \
            else self.build_tree(data_subset, free_vars, next_sep_data)

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

    def calc_threshold_gini(self, threshold: float, data: ndarray, column_ind: int) -> float:
        left_gini, sep_row = self.calc_side_threshold_gini(data, column_ind, threshold, 0, False)
        right_gini = self.calc_side_threshold_gini(data, column_ind, threshold, sep_row, True)[0]
        return left_gini * (sep_row / data.shape[0]) + right_gini * ((data.shape[0] - sep_row) / data.shape[0])

    def calc_side_threshold_gini(self, data: ndarray, column_ind: int, threshold: float, start_row: int,
                                 is_right: bool) -> (float, int):
        good_count: int = 0
        bad_count: int = 0

        while start_row < data.shape[0] and (is_right or data[start_row, column_ind] <= threshold):
            if data[start_row, -1] > self.avQuality:
                good_count += 1
            else:
                bad_count += 1
            start_row += 1
        total: int = good_count + bad_count
        # if no elements happened to be on either side, then it is a perfect separation, gini = 0
        gini: float = 0 if total == 0 else 1 - (good_count / total) ** 2 - (bad_count / total) ** 2
        return (gini, start_row)
