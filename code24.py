# hierarchical_traversal.py
from typing import Dict, List, Tuple

# -------------------------
# Example hierarchical data
# -------------------------
# Nested dict: key = node name, value = dict(children)
org = {
    "CEO": {
        "CTO": {
            "Dev Manager": {
                "Dev1": {},
                "Dev2": {}
            },
            "QA Manager": {
                "QA1": {}
            }
        },
        "CFO": {
            "Accounting Manager": {
                "Accountant1": {}
            }
        },
        "COO": {}  # leaf
    }
}

# -------------------------
# Recursive helpers
# -------------------------
def get_leaves(tree: Dict[str, Dict], path: str = "") -> List[str]:
    """
    Return a list of full-path strings for all leaves (nodes with no children).
    Example path: "CEO / CTO / Dev Manager / Dev1"
    """
    leaves: List[str] = []
    for name, children in tree.items():
        full = (path + " / " + name).lstrip(" /")
        if not children:  # no children => leaf
            leaves.append(full)
        else:
            leaves.extend(get_leaves(children, full))
    return leaves


def get_nodes_at_depth(tree: Dict[str, Dict], depth: int, current: int = 0,
                       path: str = "") -> List[Tuple[str, str]]:
    """
    Return a list of tuples (node_name, full_path) for nodes at the requested depth.
    Depth 0 = top-level keys (e.g., "CEO" in this example).
    """
    results: List[Tuple[str, str]] = []
    for name, children in tree.items():
        full = (path + " / " + name).lstrip(" /")
        if current == depth:
            results.append((name, full))
        # recurse into children with current+1
        if children:
            results.extend(get_nodes_at_depth(children, depth, current + 1, full))
    return results


def traverse(tree: Dict[str, Dict], visitor, path: str = "") -> None:
    """
    Generic traversal: call visitor(name, full_path, children_dict) for every node.
    `visitor` is any callable that accepts (name, full_path, children).
    """
    for name, children in tree.items():
        full = (path + " / " + name).lstrip(" /")
        visitor(name, full, children)
        if children:
            traverse(children, visitor, full)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    print("=== All leaves (full paths) ===")
    for leaf in get_leaves(org):
        print(" -", leaf)

    print("\n=== Nodes at depth 0 ===")
    print(get_nodes_at_depth(org, depth=0))  # [('CEO','CEO')]

    print("\n=== Nodes at depth 1 ===")
    for name, full in get_nodes_at_depth(org, depth=1):
        print(" -", full)

    print("\n=== Nodes at depth 2 ===")
    for name, full in get_nodes_at_depth(org, depth=2):
        print(" -", full)

    print("\n=== Traverse and print (name : path : #children) ===")
    def visitor(name, full_path, children):
        print(f"{name} : {full_path} : children={len(children)}")
    traverse(org, visitor)
