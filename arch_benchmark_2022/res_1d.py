from copyreg import pickle
import numpy as np
import treelib
import pickle

itera = 0
with open(f"Testing/Testing_123/Testing_123_result_generating_files/Testing_123_{itera}.pkl", "rb") as f:
    ftree = pickle.load(f)
# ftree.show()
for node in ftree.leaves():
    data = node.data
    print(data.region_support, data.region_class, data.self_id)