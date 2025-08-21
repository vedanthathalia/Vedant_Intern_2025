import baltic as bt

tree = bt.loadNewick('/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/tree_named.nwk', absoluteTime=False)
tree.treeStats()
