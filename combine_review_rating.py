
import numpy as np
from helper.general_functions import create_and_write_csv, read_csv_file


#============================ Calulate U/I deep ===============================

def Calculate_Deep(v, z):
    """Fuse review/rating features with the FM embedding.

    The previous expression computed ``0.5 * ((v*z)^2 - v^2*z^2)``, which is
    identically zero. A Hadamard interaction preserves one value per latent
    factor and is compatible with the downstream prediction layer.
    """
    v_array = np.asarray(v, dtype=np.float32)
    z_array = np.asarray(z, dtype=np.float32)
    if v_array.shape != z_array.shape:
        raise ValueError(
            f"Cannot fuse feature vectors with shapes {v_array.shape} and {z_array.shape}"
        )
    return v_array * z_array


def mergeReview_Rating(path, filename, svd, reviewer_feature_dict, item_feature_dict, getEmbedding):
    reviewerID,_ = read_csv_file(path)
    feature_dict = {}
    review_feature_list = []
    rating_feature_list = []
    for id in reviewerID:
        if getEmbedding == "reviewer":
            A = reviewer_feature_dict[id]
            B = svd.get_user_embedding(id)
        else:
            A = item_feature_dict[id]
            B = svd.get_item_embedding(id)

        z = np.concatenate((np.array(A), np.array(B)))
        feature_dict[id] = z
        review_feature_list.append(A)
        rating_feature_list.append(B)
    create_and_write_csv(filename, feature_dict)
    return feature_dict



