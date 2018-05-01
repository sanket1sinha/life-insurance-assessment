class Node:

    def __init__(self, feature_name=None, split_value=None, leaf_node_value=None):
        self.feature_name = feature_name
        self.split_value = split_value
        self.child = []
        self.leaf_node_value = leaf_node_value

    def add_child(self, node):
        self.child.append(node)
