class Node:

    def __init__(self, feature_name=None, split_value=None, leaf_node_value=None):
        self.feature_name = feature_name
        self.split_value = split_value
        self.child = []
        self.leaf_node_value = leaf_node_value

    def add_child(self, node):
        self.child.append(node)


class Tree:

    def __init__(self, root: Node):
        self.root = root


if __name__ == "__main__":
    node = Node('age')
    tree = Tree(node)
    child1 = Node('name0','l10')
    child2 = Node('name1','10')
    child3 = Node('name2','g10')

    tree.root.add_child(child1)
    tree.root.add_child(child2)
    tree.root.add_child(child3)



    for c in tree.root.child:
        row

        print(c.split_value)