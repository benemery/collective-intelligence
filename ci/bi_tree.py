"""Bi directional tree"""

class BiNode(object):
    def __init__(self, vector, left=None, right=None, id=0, distance=0.0):
        self.vector = vector
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id

    @staticmethod
    def print_tree(node, labels=None, n=0):
        """Output the tree structure to the StdOut"""
        for _ in range(n):
            print ' ',

        if not node.is_leaf:
            print '-'
        else:
            # Positive means this is an endpoint
            if not labels:
                print node.id
            else:
                print labels[node.id]

        if node.left:
            BiNode.print_tree(node.left, labels=labels, n=n+1)
        if node.right:
            BiNode.print_tree(node.right, labels=labels, n=n+1)

    @property
    def height(self):
        """How tall is this tree? I.e. how many leaf nodes does it have?"""
        if self.is_leaf:
            return 1
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0

        return left_height + right_height

    @property
    def depth(self):
        if self.is_leaf:
            return 0
        left_depth = self.left.depth if self.left else 0
        right_depth = self.right.depth if self.right else 0
        return max(left_depth, right_depth) + self.distance

    @property
    def is_leaf(self):
        """Is this a leaf node?"""
        return bool(not self.left and not self.right)
