"""
This is a utility to convert FirTree's treelog.txt output into a .dot file
which can be rendered using Graphviz, example usage:

    python3 tree_to_dot.py path/to/treelog.txt > treelog.dot
    cat treelog.dot | dot -Tpng > treelog.png


Requires Python3.6+

Graphviz is needed to render the dot files, on MacOS can install via Homebrew:
    brew install graphviz

"""
import sys
from typing import List


class Node:
    def __init__(self, name: str, split_feature: str, split_value: float, 
            core_features: List[str], 
            l: 'Node', r: 'Node'):
        self.name = name
        self.split_feature = split_feature
        self.split_value = split_value
        self.core_features = core_features
        self.l = l
        self.r = r

    def __repr__(self):
        return f"Node(name={self.name}, split_feature={self.split_feature}, split_value={self.split_value}, core_features={self.core_features}, l={self.l}, r={self.r}"


def parse_tree(file_name: str) -> List[Node]:
    nodes = []
    with open(file_name) as f:
        core_features: List[str] = []
        split_value = None
        split_feature = None
        name = None
        for l in f.readlines():
            if l.startswith("\t"):
                core_features.append(l.strip())
            elif "Best feature:" in l:
                split_feature = l.split(":")[1].strip()
            elif "Best split:" in l:
                split_value = float(l.split(":")[1].strip())
            elif "Root" in l:
                name = l.strip()
            elif l.strip() == "":
                curr_node = Node(name=name, core_features=core_features, 
                                split_value=split_value, 
                                split_feature=split_feature, 
                                l=None, r=None)
                print(f"Parsed a Node: {curr_node}", file=sys.stderr)
                nodes.append(curr_node)
                core_features = []
                split_value = None
                split_feature = None
                name = None
    return nodes


def build_tree(nodes: List[Node]) -> Node:
    node_dict = {node.name: node for node in nodes}
    root = node_dict.pop('Root')
    traversal_stack = [root]
    while len(traversal_stack) > 0:
        curr_node = traversal_stack.pop()
        name = curr_node.name
        l_name = name + "_L"
        r_name = name + "_R"
        l = node_dict.pop(l_name, None)
        r = node_dict.pop(r_name, None)
        if l:
            curr_node.l = l
            traversal_stack.append(l)
        if r:
            curr_node.r = r
            traversal_stack.append(r)
    return root


def render_dot_recur(root_node, render_core_features=False):
    if root_node:
        is_leaf = root_node.l is None and root_node.r is None
        if render_core_features or is_leaf:
            core_features_str = str(root_node.core_features).replace(',', '\\n')
            nl = "\\n"
            print(f"{root_node.name} [label=\"{root_node.name}{nl}{core_features_str}\"]")
        else:
            print(f"{root_node.name}")
        if root_node.l:
            print(f"{root_node.name} -> {root_node.l.name} [label=\"{root_node.split_feature} < {root_node.split_value}\"]")
        if root_node.r:
            print(f"{root_node.name} -> {root_node.r.name} [label=\"{root_node.split_feature} >= {root_node.split_value}\"]")
        render_dot_recur(root_node.l)
        render_dot_recur(root_node.r)


def render_dot(root_node, render_core_features=False):
    print("""digraph {
    node [shape=box]
    compound=true
    concentrate=true
    ranksep=1
    
    """)

    render_dot_recur(root_node, render_core_features)

    print("}")


if __name__ == "__main__":
    tree_filename = sys.argv[1]
    nodes = parse_tree(tree_filename)
    root = build_tree(nodes)
    render_dot(root)
