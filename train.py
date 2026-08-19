import torch
import torch.nn as nn
import torch.optim as optim
import ast
import copy
import numpy as np
import pandas as pd
import tqdm
from helper.general_functions import create_and_write_csv, word_segment
from combine_review_rating import Calculate_Deep
from sklearn.metrics import accuracy_score, mean_absolute_error
from review_processing.merge_senmatic_review import (
    extract_features,
    merge_fine_coarse_features,
)
from helper.utils import setup_path
from helper.device import get_device
from rating_processing.svd import initialize_svd
from rating_processing.factorization_machine import run

def reprocess_input(data, device=None):
    rating = torch.tensor([float(x) for x in data['overall']], dtype=torch.float32)
    item_bias = torch.tensor([float(x) for x in data['item_bias']], dtype=torch.float32)
    user_bias = torch.tensor([float(x) for x in data['user_bias']], dtype=torch.float32)

    user_feature = []
    for item in data['Udeep']:
        if isinstance(item, str):
            user_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            user_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            user_feature.append(item.float())
    
    item_feature = []
    for item in data['Ideep']:
        if isinstance(item, str):
            item_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            item_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            item_feature.append(item.float())
    
    user_feature = torch.stack(user_feature)
    item_feature = torch.stack(item_feature)
    
    if device is not None:
        # Asynchronous CPU->MPS copies can expose incomplete/corrupted values
        # on some PyTorch/macOS combinations. CUDA can safely use non-blocking
        # transfers; MPS must use the synchronized default.
        non_blocking = device.type == "cuda"
        rating = rating.to(device, non_blocking=non_blocking)
        user_feature = user_feature.to(device, non_blocking=non_blocking)
        item_feature = item_feature.to(device, non_blocking=non_blocking)
        item_bias = item_bias.to(device, non_blocking=non_blocking)
        user_bias = user_bias.to(device, non_blocking=non_blocking)

    return rating, user_feature, item_feature, item_bias, user_bias

def calculate_rmse(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    squared_errors = (y_true_np - y_pred_np) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

# Define the model
class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, global_mean=0.0):
        super(FullyConnectedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.global_bias = nn.Parameter(torch.tensor([global_mean], dtype=torch.float32))

    def forward(self, user_features, item_features, item_bias, user_bias):
        interaction = user_features * item_features
        prediction = self.fc(interaction.to(dtype=torch.float32)).squeeze(-1)
        prediction += self.global_bias + item_bias + user_bias
        return prediction

    
def train_deepbert(
    train_data_loader,
    valid_data_loader,
    num_factors,
    batch_size,
    epochs,
    method_name,
    log_interval=100,
    learning_rate=0.01,
):
    print("=================== Training DeepCGSR model ============================")
    device = get_device()
    print(f"Training on device: {device}")
    feature_dim = train_data_loader.dataset.tensors[3].shape[1]
    if feature_dim == 0:
        raise ValueError("Udeep/Ideep feature vectors are empty")
    global_mean = float(train_data_loader.dataset.tensors[2].mean())
    model = FullyConnectedModel(
        input_dim=feature_dim,
        output_dim=1,
        global_mean=global_mean,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )
    best_valid_rmse = float("inf")
    best_state = None
    stale_epochs = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_data_loader):
            rating, user_feature, item_feature, item_bias, user_bias = reprocess_input({
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }, device=device)

            predictions = model(user_feature, item_feature, item_bias, user_bias)
            loss = criterion(predictions, rating)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(batch[0])}/{len(train_data_loader.dataset)} "
                      f"({100. * batch_idx / len(train_data_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        valid_rmse, valid_mae = evaluate_regression(model, valid_data_loader)
        scheduler.step(valid_rmse)
        print(
            f"Epoch {epoch + 1}: train loss={total_loss / len(train_data_loader):.6f}, "
            f"valid RMSE={valid_rmse:.6f}, valid MAE={valid_mae:.6f}"
        )
        if valid_rmse < best_valid_rmse - 1e-6:
            best_valid_rmse = valid_rmse
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            torch.save(best_state, f"chkpt/{method_name}.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"DeepBERT early stopped at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_regression(model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    targets, predictions = [], []
    with torch.no_grad():
        for batch in data_loader:
            target, udeep, ideep, item_bias, user_bias = reprocess_input(
                {
                    "overall": batch[2],
                    "Udeep": batch[3],
                    "Ideep": batch[4],
                    "item_bias": batch[5],
                    "user_bias": batch[6],
                },
                device=device,
            )
            output = model(udeep, ideep, item_bias, user_bias).clamp(1.0, 5.0)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Model produced NaN/Inf predictions")
            targets.extend(target.cpu().tolist())
            predictions.extend(output.cpu().tolist())
    return calculate_rmse(targets, predictions), mean_absolute_error(targets, predictions)

def test(model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            target, udeep, ideep, item_bias, user_bias = reprocess_input(data, device=device)
            udeep = torch.tensor(udeep, dtype=torch.float32) if isinstance(udeep, list) else udeep
            ideep = torch.tensor(ideep, dtype=torch.float32) if isinstance(ideep, list) else ideep

            y = model(udeep, ideep, item_bias, user_bias).clamp(1.0, 5.0)
            if not torch.isfinite(y).all():
                raise FloatingPointError(
                    "Validation produced NaN/Inf predictions. Check feature "
                    "values and training stability."
                )
            
            targets.extend(target.cpu().tolist())
            predicts.extend([round(float(pred)) for pred in y.flatten().cpu().numpy()])

    new_targets = [-1 if i < 4 else 1 for i in targets]
    new_predicts = [-1 if i < 4 else 1 for i in predicts]

    accuracy = accuracy_score(new_targets, new_predicts)
    print("Accuracy: ", accuracy)
    return accuracy

def test_rsme(model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            target, udeep, ideep, item_bias, user_bias = reprocess_input(data, device=device)
            y = model(udeep, ideep, item_bias, user_bias).clamp(1.0, 5.0)
            if not torch.isfinite(y).all():
                raise FloatingPointError(
                    "Evaluation produced NaN/Inf predictions. Check feature "
                    "values and training stability."
                )
            targets.extend(target.cpu().tolist())
            predicts.extend([float(pred) for pred in y.flatten().cpu().numpy()])

    new_targer = []
    new_predict = []
    new_targer = targets
    new_predict = new_predict

    print("rsme raw: ", calculate_rmse(targets, predicts))
    mae_value = mean_absolute_error(targets, predicts)
    print("MAE: ", mae_value)
    return calculate_rmse(targets, predicts), mae_value

def _interaction_frame(dataset_df):
    rating_field = "overall_new" if "overall_new" in dataset_df else "overall"
    frame = pd.DataFrame(
        {
            "reviewerID": dataset_df["reviewerID"].astype(str),
            "itemID": dataset_df["asin"].astype(str),
            "overall": pd.to_numeric(dataset_df[rating_field]),
        }
    )
    if not np.isfinite(frame["overall"]).all():
        raise ValueError("Ratings contain NaN or Inf")
    return frame


def _fit_regularized_biases(train_interactions, regularization=10.0):
    """Fit global/user/item biases using train ratings only."""
    frame = train_interactions.copy()
    global_mean = float(frame["overall"].mean())
    user_stats = (frame["overall"] - global_mean).groupby(frame["reviewerID"]).agg(
        ["sum", "count"]
    )
    user_biases = (
        user_stats["sum"] / (user_stats["count"] + regularization)
    ).to_dict()
    frame["user_bias"] = frame["reviewerID"].map(user_biases).fillna(0.0)
    residual = frame["overall"] - global_mean - frame["user_bias"]
    item_stats = residual.groupby(frame["itemID"]).agg(["sum", "count"])
    item_biases = (
        item_stats["sum"] / (item_stats["count"] + regularization)
    ).to_dict()
    return global_mean, user_biases, item_biases


def _mean_feature(features, size):
    if not features:
        return np.zeros(size, dtype=np.float32)
    return np.mean(
        np.vstack([np.asarray(value, dtype=np.float32) for value in features.values()]),
        axis=0,
    ).astype(np.float32)


def _fine_feature_diagnostics(review_rows, num_factors):
    vectors = []
    for value in review_rows["fine_feature"]:
        if isinstance(value, str):
            vector = np.fromstring(
                value.strip("[]").replace(",", " "),
                sep=" ",
                dtype=np.float32,
            )
        else:
            vector = np.asarray(value, dtype=np.float32)
        if vector.shape == (num_factors,):
            vectors.append(vector)
    if not vectors:
        return {
            "reviews": 0,
            "zeroValueRate": 1.0,
            "meanNonzeroTopics": 0.0,
            "medianNonzeroTopics": 0.0,
            "meanL2Norm": 0.0,
        }
    matrix = np.vstack(vectors)
    nonzero_topics = np.count_nonzero(matrix, axis=1)
    return {
        "reviews": int(len(matrix)),
        "zeroValueRate": float(np.mean(matrix == 0)),
        "meanNonzeroTopics": float(np.mean(nonzero_topics)),
        "medianNonzeroTopics": float(np.median(nonzero_topics)),
        "meanL2Norm": float(np.mean(np.linalg.norm(matrix, axis=1))),
    }


def _create_deep_embeddings(
    identifiers,
    review_features,
    svd,
    fm,
    *,
    entity,
    num_factors,
):
    normalized_features = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in review_features.items()
    }
    fallback_review = _mean_feature(normalized_features, num_factors)
    result = {}
    for identifier in sorted({str(value) for value in identifiers}):
        review_vector = normalized_features.get(identifier, fallback_review)
        if entity == "reviewer":
            rating_vector = svd.get_user_embedding(identifier)
            fm_name = f"reviewerID_{identifier}"
        else:
            rating_vector = svd.get_item_embedding(identifier)
            fm_name = f"itemID_{identifier}"
        combined = np.concatenate((review_vector, rating_vector)).astype(np.float32)
        result[identifier] = Calculate_Deep(combined, fm.get_embedding(fm_name))
    return result


def _create_review_only_embeddings(identifiers, review_features, num_factors):
    normalized_features = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in review_features.items()
    }
    fallback = _mean_feature(normalized_features, num_factors)
    return {
        identifier: normalized_features.get(identifier, fallback).copy()
        for identifier in sorted({str(value) for value in identifiers})
    }


def _build_final_feature_frame(
    dataset_df,
    user_embeddings,
    item_embeddings,
    user_biases,
    item_biases,
    reviewer_codes,
    item_codes,
):
    interactions = _interaction_frame(dataset_df)
    users = interactions["reviewerID"].tolist()
    items = interactions["itemID"].tolist()
    return pd.DataFrame(
        {
            "reviewerID": [reviewer_codes.get(value, -1) for value in users],
            "itemID": [item_codes.get(value, -1) for value in items],
            "overall": interactions["overall"].to_numpy(dtype=np.float32),
            "Udeep": [user_embeddings[value].tolist() for value in users],
            "Ideep": [item_embeddings[value].tolist() for value in items],
            "item_bias": [float(item_biases.get(value, 0.0)) for value in items],
            "user_bias": [float(user_biases.get(value, 0.0)) for value in users],
        }
    )


def prepare_deepbert_splits(
    train_df,
    valid_df,
    test_df,
    num_factors,
    num_words,
    cluster_method="Birch",
    bert_cache_dir="./chkpt",
    embeddings_cache_path=None,
    coarse_cache_path=None,
    max_topics_per_word=2,
    feature_mode="full",
    cluster_seed=42,
    bert_model="bert-base-uncased",
):
    """Fit every feature model on train and transform valid/test without fitting."""
    if feature_mode not in {"full", "review-only", "rating-only", "raw"}:
        raise ValueError(
            "feature_mode must be one of: 'full', 'review-only', "
            "'rating-only', 'raw'"
        )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        final_data_path,
        svd_path,
        checkpoint_path,
        sparse_matrix_path,
    ) = setup_path()

    if feature_mode in {"raw", "rating-only"}:
        # Both modes keep the original review text. Raw also restores the
        # original rating, while rating-only isolates rating adjustment by
        # retaining overall_new as both the fitted rating and ground truth.
        def select_ablation_fields(frame):
            selected_frame = frame.copy()
            selected_frame["filteredReviewText"] = selected_frame["reviewText"]
            if feature_mode == "raw":
                selected_frame["overall_new"] = selected_frame["overall"]
            return selected_frame

        train_df = select_ablation_fields(train_df)
        valid_df = select_ablation_fields(valid_df)
        test_df = select_ablation_fields(test_df)

    train_interactions = _interaction_frame(train_df)
    valid_interactions = _interaction_frame(valid_df)
    all_users = set(train_df["reviewerID"].astype(str))
    all_users.update(valid_df["reviewerID"].astype(str))
    all_users.update(test_df["reviewerID"].astype(str))
    all_items = set(train_df["asin"].astype(str))
    all_items.update(valid_df["asin"].astype(str))
    all_items.update(test_df["asin"].astype(str))

    split_data = [word_segment(text) for text in train_df["filteredReviewText"]]
    train_review_rows = extract_features(
        train_df,
        split_data,
        num_factors,
        num_words,
        "train",
        validation_df=valid_df,
        cluster_method=cluster_method,
        bert_cache_dir=bert_cache_dir,
        embeddings_cache_path=embeddings_cache_path,
        coarse_cache_path=coarse_cache_path,
        max_topics_per_word=max_topics_per_word,
        cluster_seed=cluster_seed,
        bert_model=bert_model,
    )
    feature_diagnostics = _fine_feature_diagnostics(
        train_review_rows,
        num_factors,
    )
    reviewer_features = merge_fine_coarse_features(
        train_review_rows,
        num_factors,
        groupBy="reviewerID",
    )
    item_features = merge_fine_coarse_features(
        train_review_rows,
        num_factors,
        groupBy="itemID",
    )
    create_and_write_csv("reviewer_feature_train", reviewer_features)
    create_and_write_csv("item_feature_train", item_features)

    if feature_mode in {"full", "rating-only", "raw"}:
        interaction_path = "feature/interactions_train.csv"
        train_interactions.to_csv(interaction_path, index=False)
        svd = initialize_svd(
            interaction_path,
            num_factors,
            svd_path + "train.pt",
            validation_data=valid_interactions,
        )
        fm = run(
            interaction_path,
            num_factors * 2,
            checkpoint_path + "train.pkl",
            sparse_matrix_path + "train.npz",
            validation_data=valid_interactions,
        )
        user_embeddings = _create_deep_embeddings(
            all_users,
            reviewer_features,
            svd,
            fm,
            entity="reviewer",
            num_factors=num_factors,
        )
        item_embeddings = _create_deep_embeddings(
            all_items,
            item_features,
            svd,
            fm,
            entity="item",
            num_factors=num_factors,
        )
        _, user_biases, item_biases = _fit_regularized_biases(train_interactions)
    else:
        user_embeddings = _create_review_only_embeddings(
            all_users,
            reviewer_features,
            num_factors,
        )
        item_embeddings = _create_review_only_embeddings(
            all_items,
            item_features,
            num_factors,
        )
        user_biases = {}
        item_biases = {}
    create_and_write_csv("u_deep_train", user_embeddings)
    create_and_write_csv("i_deep_train", item_embeddings)

    train_users = sorted(train_interactions["reviewerID"].unique())
    train_items = sorted(train_interactions["itemID"].unique())
    reviewer_codes = {value: index for index, value in enumerate(train_users)}
    item_codes = {value: index for index, value in enumerate(train_items)}

    output_paths = {}
    for split, frame in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        output = _build_final_feature_frame(
            frame,
            user_embeddings,
            item_embeddings,
            user_biases,
            item_biases,
            reviewer_codes,
            item_codes,
        )
        output_path = f"{final_data_path}DeepBERT_{split}.csv"
        output.to_csv(output_path, index=False)
        output_paths[split] = output_path
    output_paths["featureDiagnostics"] = feature_diagnostics
    return output_paths
