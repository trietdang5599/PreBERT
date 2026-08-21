import os
import tempfile
from pathlib import Path

import tqdm
import torch
import pandas as pd
import numpy as np
from helper.general_functions import create_and_write_csv, load_data_from_csv, split_text
from init import dep_parser
from review_processing.coarse_gain import (
    get_coarse_sentiment_score,
    get_vader_coarse_sentiment_score,
)
from review_processing.fine_gain import get_tbert_model, get_topic_sentiment_matrix_tbert
from helper.device import get_device, release_device_cache


def merge_fine_coarse_features(data_df, num_factors, groupBy="reviewerID"):
    feature_dict = {}
    device = get_device()
    
    for id, df in data_df.groupby(groupBy):
        feature = torch.zeros(num_factors, device=device, dtype=torch.float32)
        feature_count = 0
        list_finefeature = df['fine_feature']
        list_coarse_feature = df['coarse_feature']
        
        for fine, coarse in zip(list_finefeature, list_coarse_feature):
            try:
                if isinstance(fine, str):
                    values = np.fromstring(
                        fine.strip("[]").replace(",", " "),
                        sep=" ",
                        dtype=np.float32,
                    )
                else:
                    values = np.asarray(fine, dtype=np.float32)
                fine_feature = torch.tensor(
                    values,
                    device=device,
                    dtype=torch.float32,
                )
                coarse_feature = torch.tensor(
                    float(coarse),
                    device=device,
                    dtype=torch.float32,
                )
                feature += fine_feature * coarse_feature
                feature_count += 1
            except Exception as e:
                print("Error: ", e)
                continue
        if feature_count:
            feature /= feature_count
        feature_dict[id] = feature.cpu().numpy()
        
    return feature_dict

# Extract fine-grained and coarse-grained features
def extract_review_feature(
    data_df,
    model,
    dep_parser,
    tokenizer,
    topic_word_matrix,
    num_topics,
    coarse_cache_path=None,
    bert_fine_tuning=True,
):
    """Extract cluster-dependent fine features and reusable coarse scores.

    Coarse BERT sentiment does not depend on the clustering algorithm, so an
    optional row-aligned NumPy cache avoids repeating BERT inference during a
    clustering ablation. Fine-grained topic features are always recomputed.
    """
    device = get_device()
    model = model.to(device)
    data_df = data_df.reset_index(drop=True)
    topic_word_matrix = tuple(
        frozenset(topic_words) for topic_words in topic_word_matrix
    )

    cache_path = Path(coarse_cache_path) if coarse_cache_path else None
    coarse_scores = np.full(len(data_df), np.nan, dtype=np.float32)
    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape != coarse_scores.shape:
            raise ValueError(
                f"Cached coarse scores have shape {cached.shape}; "
                f"expected {coarse_scores.shape}"
            )
        coarse_scores = cached.astype(np.float32, copy=False)
        print(f"Loaded cached BERT coarse scores: {cache_path}")

    row_list = []
    print("data_train_size: ", data_df.shape[0])
    for asin, df in tqdm.tqdm(data_df.groupby("asin")):
        for row_index, row in df.iterrows():
            text = row["filteredReviewText"]
            try:
                # Convert text về chuỗi rỗng nếu nó là None
                if not isinstance(text, str):
                    text = ""
                fine_feature = torch.zeros(
                    num_topics,
                    device=device,
                    dtype=torch.float32,
                )
                coarse_is_cached = np.isfinite(coarse_scores[row_index])
                coarse_feature = (
                    float(coarse_scores[row_index]) if coarse_is_cached else 0.0
                )

                text_chunks = split_text(text) if text else [""]
                count_null = 0
                for chunk in text_chunks:
                    if chunk and chunk.strip():
                        try:
                            # Giữ tensor trên GPU trong quá trình tính toán
                            fine_feature_chunk = get_topic_sentiment_matrix_tbert(chunk, topic_word_matrix, dep_parser, topic_nums=num_topics)
                            if not coarse_is_cached and bert_fine_tuning:
                                coarse_feature_chunk = get_coarse_sentiment_score(
                                    model, tokenizer, chunk
                                )
                            elif not coarse_is_cached:
                                coarse_feature_chunk = (
                                    get_vader_coarse_sentiment_score(chunk)
                                )
                        except KeyError as e:
                            print(f"Skipping chunk due to missing key in vocabulary: {e}")
                            continue
                    else:
                        count_null += 1
                        continue

                    fine_feature += fine_feature_chunk
                    if not coarse_is_cached:
                        if torch.is_tensor(coarse_feature_chunk):
                            coarse_feature_chunk = (
                                coarse_feature_chunk.detach().float().cpu().item()
                            )
                        coarse_feature += float(coarse_feature_chunk)

                if not coarse_is_cached:
                    coarse_feature /= max(1, len(text_chunks) - count_null)
                    coarse_scores[row_index] = coarse_feature
                fine_feature = torch.clamp(fine_feature, min=-5, max=5)

                new_row = {
                    'reviewerID': row["reviewerID"],
                    'itemID': asin, 
                    'overall': row["overall_new"],
                    'fine_feature': fine_feature.cpu().numpy(),
                    'coarse_feature': coarse_feature
                }
                row_list.append(new_row)
            except Exception as e:
                print(f"Error: {e}, Text: {text}, fine_feature: {fine_feature}")
                continue

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.save(handle, coarse_scores, allow_pickle=False)
            temporary_path.replace(cache_path)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        print(f"Cached BERT coarse scores: {cache_path}")

    return pd.DataFrame(row_list, columns=['reviewerID', 'itemID', 'overall', 'fine_feature', 'coarse_feature'])


# Global variables to store features
reviewer_feature_dict = {}
item_feature_dict = {}
allFeatureReview = pd.DataFrame(columns=['reviewerID', 'itemID', 'overall', 'unixReviewTime', 'fine_feature', 'coarse_feature'])

def initialize_features(filename, num_factors):
    # print("Initialize features")
    global reviewer_feature_dict, item_feature_dict
    allreviews_path = "feature/allFeatureReview_"
    reviewer_path = "/feature/reviewer_feature_"
    item_path = "feature/item_feature_"
    
    # Initialize or load reviewer features
    if os.path.exists(reviewer_path + filename +".csv"):
        reviewer_feature_dict = load_data_from_csv(reviewer_path + filename +".csv")
    else:
        allFeatureReview = pd.read_csv(allreviews_path + filename +".csv")
        reviewer_feature_dict = merge_fine_coarse_features(allFeatureReview, num_factors, groupBy="reviewerID")
        create_and_write_csv("reviewer_feature_" + filename, reviewer_feature_dict)
        
    # Initialize or load item features
    if os.path.exists(item_path+ filename +".csv"):
        item_feature_dict = load_data_from_csv(item_path+ filename +".csv")
    else:
        allFeatureReview = pd.read_csv(allreviews_path+ filename +".csv")
        item_feature_dict = merge_fine_coarse_features(allFeatureReview, num_factors, groupBy="itemID")
        create_and_write_csv("item_feature_" + filename, item_feature_dict)
    return reviewer_feature_dict, item_feature_dict
        
def extract_features(
    data_df,
    split_data,
    num_topics,
    num_words,
    filename,
    validation_df=None,
    cluster_method="Birch",
    bert_cache_dir="./chkpt",
    embeddings_cache_path=None,
    coarse_cache_path=None,
    max_topics_per_word=2,
    cluster_seed=42,
    bert_model="answerdotai/ModernBERT-base",
    bert_fine_tuning=True,
):
    allreviews_path = "feature/allFeatureReview_"
    if os.path.exists(allreviews_path + filename +".csv"):
        allFeatureReview = pd.read_csv(allreviews_path + filename +".csv")
    else:
        model, tokenizer, topic_word_matrix = get_tbert_model(
            data_df,
            split_data,
            num_topics,
            num_words,
            cluster_method=cluster_method,
            validation_df=validation_df,
            bert_cache_dir=bert_cache_dir,
            embeddings_cache_path=embeddings_cache_path,
            max_topics_per_word=max_topics_per_word,
            cluster_seed=cluster_seed,
            bert_model=bert_model,
            bert_fine_tuning=bert_fine_tuning,
        )
        allFeatureReview = extract_review_feature(
            data_df,
            model,
            dep_parser,
            tokenizer,
            topic_word_matrix,
            num_topics,
            coarse_cache_path=coarse_cache_path,
            bert_fine_tuning=bert_fine_tuning,
        )
        allFeatureReview.to_csv(allreviews_path + filename +".csv", index=False)
        device = next(model.parameters()).device
        del model
        release_device_cache(device)
    return allFeatureReview
