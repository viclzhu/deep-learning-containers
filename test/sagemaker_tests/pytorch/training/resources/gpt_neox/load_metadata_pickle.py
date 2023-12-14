import pickle
from torch.distributed.checkpoint.metadata import Metadata

# ~1.7MB
pkl_path = "/fsx/users/viczhu/unpicklerr/20b_gptneox_metadata0"

with open(pkl_path, "rb") as f:
    obj: Metadata
    obj = pickle.load(f)

print(type(obj))

# Metadata
# Keys are the same from the `state_dict` used.
# - state_dict_metadata: Dict[str, STORAGE_TYPES]
# - planner_data: Any = None
# - storage_data: Any = None
print("state_dict_metadata", len(obj.state_dict_metadata.keys())) # dict, 1589
# print("planner_data", obj.planner_data.keys(), len(obj.planner_data.keys())) # dict, 1589 items
# print("storage_data", obj.storage_data) # None