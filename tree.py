class Node:

    def __init__(self, feature_name, feature_split_value=None):
        self.feature_name = feature_name
        self.feature_split_value = feature_split_value
        self.child = []
        self.leaf_node_value = None

    def add_child(self, node):
        self.child.append(node)


class Tree:

    def __init__(self, start_node: Node):
        self.start_node = start_node

