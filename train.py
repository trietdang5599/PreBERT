import torch
import torch.nn as nn
import torch.optim as optim
import ast
import copy
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler
from helper.general_functions import create_and_write_csv, word_segment
from combine_review_rating import Calculate_Deep
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
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


class FusionMLPModel(nn.Module):
    """Learn nonlinear fusion from user, item, and Hadamard features."""

    def __init__(self, feature_dim, hidden_dim=64, dropout=0.1, global_mean=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.global_bias = nn.Parameter(torch.tensor([global_mean], dtype=torch.float32))

    def forward(self, user_features, item_features, item_bias, user_bias):
        fused = torch.cat(
            (user_features, item_features, user_features * item_features), dim=1
        ).to(dtype=torch.float32)
        prediction = self.network(fused).squeeze(-1)
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
    weight_decay=1e-4,
    regressor_architecture="linear",
    mlp_hidden_dim=64,
    mlp_dropout=0.1,
):
    print("=================== Training DeepCGSR model ============================")
    device = get_device()
    print(f"Training on device: {device}")
    feature_dim = train_data_loader.dataset.tensors[3].shape[1]
    if feature_dim == 0:
        raise ValueError("Udeep/Ideep feature vectors are empty")
    global_mean = float(train_data_loader.dataset.tensors[2].mean())
    if regressor_architecture == "linear":
        model = FullyConnectedModel(
            input_dim=feature_dim,
            output_dim=1,
            global_mean=global_mean,
        )
    elif regressor_architecture == "fusion-mlp":
        model = FusionMLPModel(
            feature_dim=feature_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
            global_mean=global_mean,
        )
    else:
        raise ValueError(
            "regressor_architecture must be 'linear' or 'fusion-mlp'"
        )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
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

def test(model, data_loader, return_details=False):
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
    details = {
        "accuracy": float(accuracy),
        "balancedAccuracy": float(
            balanced_accuracy_score(new_targets, new_predicts)
        ),
        "macroF1": float(
            f1_score(new_targets, new_predicts, average="macro", zero_division=0)
        ),
        "confusionMatrix": confusion_matrix(
            new_targets, new_predicts, labels=[-1, 1]
        ).tolist(),
    }
    print(
        "Classification metrics: "
        f"accuracy={details['accuracy']:.6f}, "
        f"balanced accuracy={details['balancedAccuracy']:.6f}, "
        f"macro-F1={details['macroF1']:.6f}"
    )
    return details if return_details else accuracy

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
    rating_field = "modelRating" if "modelRating" in dataset_df else "overall"
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
    include_review_features=True,
    include_rating_features=True,
    feature_scaler=None,
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
        if not include_review_features:
            review_vector = np.zeros(num_factors, dtype=np.float32)
        if not include_rating_features:
            rating_vector = np.zeros(num_factors, dtype=np.float32)
        combined = np.concatenate((review_vector, rating_vector)).astype(np.float32)
        if feature_scaler is not None:
            combined = feature_scaler.transform(combined.reshape(1, -1))[0].astype(
                np.float32
            )
        fusion_vector = (
            fm.get_embedding(fm_name)
            if include_rating_features
            else np.ones(num_factors * 2, dtype=np.float32)
        )
        result[identifier] = Calculate_Deep(combined, fusion_vector)
    return result


def _fit_deep_feature_scaler(
    train_users,
    train_items,
    reviewer_features,
    item_features,
    svd,
    *,
    num_factors,
    include_review_features=True,
    include_rating_features=True,
):
    """Fit feature-wise scaling using train entities only.

    This scaler is deliberately fitted before FM fusion.  It keeps review and
    SVD blocks on comparable scales while avoiding validation/test leakage.
    """
    rows = []
    for identifiers, features, entity in (
        (train_users, reviewer_features, "reviewer"),
        (train_items, item_features, "item"),
    ):
        normalized = {
            str(key): np.asarray(value, dtype=np.float32)
            for key, value in features.items()
        }
        fallback = _mean_feature(normalized, num_factors)
        for identifier in sorted(map(str, identifiers)):
            review_vector = normalized.get(identifier, fallback)
            rating_vector = (
                svd.get_user_embedding(identifier)
                if entity == "reviewer"
                else svd.get_item_embedding(identifier)
            )
            if not include_review_features:
                review_vector = np.zeros(num_factors, dtype=np.float32)
            if not include_rating_features:
                rating_vector = np.zeros(num_factors, dtype=np.float32)
            rows.append(np.concatenate((review_vector, rating_vector)))
    if not rows:
        raise ValueError("Cannot fit deep feature scaler without train entities")
    return StandardScaler().fit(np.vstack(rows))


def _build_final_feature_frame(
    dataset_df,
    user_embeddings,
    item_embeddings,
    user_biases,
    item_biases,
    reviewer_codes,
    item_codes,
    include_rating_features=True,
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
            "item_bias": [
                float(item_biases.get(value, 0.0)) if include_rating_features else 0.0
                for value in items
            ],
            "user_bias": [
                float(user_biases.get(value, 0.0)) if include_rating_features else 0.0
                for value in users
            ],
        }
    )


def apply_preprocessing_mode(
    train_df, valid_df, test_df, feature_mode, ground_truth_field="overall"
):
    """Select model inputs and the configured immutable test label."""
    if feature_mode not in {"full", "review-only", "rating-only", "raw"}:
        raise ValueError(f"Unsupported preprocessing mode: {feature_mode}")
    uses_filtered_review = feature_mode in {"review-only", "full"}
    uses_adjusted_training_rating = feature_mode in {"rating-only", "full"}
    if ground_truth_field not in {"overall", "overall_new"}:
        raise ValueError("ground_truth_field must be 'overall' or 'overall_new'")

    def select_fields(frame, *, is_test=False):
        selected_frame = frame.copy()
        if not uses_filtered_review:
            selected_frame["filteredReviewText"] = selected_frame["reviewText"]
        if is_test:
            if ground_truth_field not in selected_frame:
                raise ValueError(
                    f"test split is missing configured ground truth: {ground_truth_field}"
                )
            selected_frame["modelRating"] = selected_frame[ground_truth_field]
        elif uses_adjusted_training_rating:
            if "overall_new" not in selected_frame:
                raise ValueError("train/validation split is missing overall_new")
            selected_frame["modelRating"] = selected_frame["overall_new"]
        else:
            selected_frame["modelRating"] = selected_frame["overall"]
        # Review-feature code historically consumes overall_new. Keep it as
        # an internal working label after the public input contract is applied.
        selected_frame["overall_new"] = selected_frame["modelRating"]
        return selected_frame

    return (
        select_fields(train_df),
        select_fields(valid_df),
        select_fields(test_df, is_test=True),
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
    bert_model="answerdotai/ModernBERT-base",
    bert_fine_tuning=True,
    balance_bert_classes=True,
    ground_truth_field="overall",
    rec_feature_ablation="full",
    standardize_deep_features=False,
):
    """Fit on fixed train/validation splits and evaluate the selected test rating."""
    if feature_mode not in {"full", "review-only", "rating-only", "raw"}:
        raise ValueError(
            "feature_mode must be one of: 'full', 'review-only', "
            "'rating-only', 'raw'"
        )
    if rec_feature_ablation not in {"full", "without-review", "without-rating"}:
        raise ValueError(
            "rec_feature_ablation must be one of: 'full', 'without-review', "
            "'without-rating'"
        )
    include_review_features = rec_feature_ablation != "without-review"
    include_rating_features = rec_feature_ablation != "without-rating"
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

    train_df, valid_df, test_df = apply_preprocessing_mode(
        train_df,
        valid_df,
        test_df,
        feature_mode,
        ground_truth_field=ground_truth_field,
    )

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
        bert_fine_tuning=bert_fine_tuning,
        balance_bert_classes=balance_bert_classes,
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

    # All four preprocessing modes use the same PreBERT architecture. Only
    # input text/rating preprocessing changes between modes.
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
    feature_scaler = None
    if standardize_deep_features:
        feature_scaler = _fit_deep_feature_scaler(
            train_interactions["reviewerID"].unique(),
            train_interactions["itemID"].unique(),
            reviewer_features,
            item_features,
            svd,
            num_factors=num_factors,
            include_review_features=include_review_features,
            include_rating_features=include_rating_features,
        )
        print(
            "Standardized deep features using train entities only "
            f"({feature_scaler.n_samples_seen_} entities)."
        )
    user_embeddings = _create_deep_embeddings(
        all_users,
        reviewer_features,
        svd,
        fm,
        entity="reviewer",
        num_factors=num_factors,
        include_review_features=include_review_features,
        include_rating_features=include_rating_features,
        feature_scaler=feature_scaler,
    )
    item_embeddings = _create_deep_embeddings(
        all_items,
        item_features,
        svd,
        fm,
        entity="item",
        num_factors=num_factors,
        include_review_features=include_review_features,
        include_rating_features=include_rating_features,
        feature_scaler=feature_scaler,
    )
    _, user_biases, item_biases = _fit_regularized_biases(train_interactions)
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
            include_rating_features=include_rating_features,
        )
        output_path = f"{final_data_path}DeepBERT_{split}.csv"
        output.to_csv(output_path, index=False)
        output_paths[split] = output_path
    output_paths["featureDiagnostics"] = feature_diagnostics
    return output_paths
