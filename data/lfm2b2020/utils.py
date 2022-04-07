import os
import json
import numpy as np
from scipy import sparse as sp


def store_results(storage_dir, interaction_matrix, user_info, attribute_descriptions):
    os.makedirs(storage_dir, exist_ok=True)
    sp.save_npz(os.path.join(storage_dir, "interactions.npz"), interaction_matrix)
    
    user_info.to_csv(os.path.join(storage_dir, "user_info.csv"), index=False)
    user_info.head()

    with open(os.path.join(storage_dir, "attribute_descriptions.json"), "w") as fh:
        json.dump(attribute_descriptions, fh, indent="\t")


def ensure_min_interactions(interaction_matrix, user_info, min_interactions_user, min_interactions_item):
    # Remove until there are enough interactions from each side
    while True:
        n_interactions_per_user = np.array(interaction_matrix.sum(axis=1)).flatten()
        n_interactions_per_item = np.array(interaction_matrix.sum(axis=0)).flatten()

        # filter items with too less interactions
        enough_interactions_item = n_interactions_per_item >= min_interactions_item
        interaction_matrix = interaction_matrix[:, enough_interactions_item]

        # only keep those users with enough interactions
        enough_interactions_user = n_interactions_per_user >= min_interactions_user
        user_info = user_info.loc[enough_interactions_user]
        user_info.reset_index(drop=True, inplace=True)

        interaction_matrix = interaction_matrix[enough_interactions_user]

        # reassign index
        user_info = user_info.assign(userID = user_info.index)

        if np.sum(enough_interactions_item == False) == 0 \
             and np.sum(enough_interactions_user == False) == 0:
            break
            
    print("Final shape of interactions matrix is", interaction_matrix.shape)
    print("==> {} users and {} items are remaining.\n".format(*interaction_matrix.shape))
    return interaction_matrix, user_info


def print_stats(interaction_matrix):
    n_users = interaction_matrix.shape[0]
    n_items = interaction_matrix.shape[1]
    n_interactions = int(interaction_matrix.sum())
    density = n_interactions / (n_items * n_users)

    print(f"Number of interactions is {n_interactions},")
    print(f"which leads to a density of {density:.4f}.")
