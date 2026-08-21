import numpy as np
import torch
import os
import copy
import tempfile
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import sentiwordnet as swn
from sklearn.cluster import KMeans, Birch, DBSCAN, MeanShift, BisectingKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.utils import logging as transformers_logging
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import wordnet as wn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from helper.general_functions import preprocessed, word_segment
from helper.device import get_device


def _load_classifier_for_fine_tuning(model_name_or_path, num_labels):
    """Load a pretrained encoder with a fresh task head without a false alarm.

    Base encoder checkpoints such as ModernBERT or ``bert-base-uncased`` do not
    contain sequence-classification weights. Transformers reports that as
    a warning even though this function immediately fine-tunes the complete
    model below. Capture the loading information and replace that warning with
    an explicit training message; real loading failures still raise normally.
    """
    previous_verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            attn_implementation="eager",
            output_loading_info=True,
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)

    initialized_parameters = loading_info.get("missing_keys", [])
    if initialized_parameters:
        print(
            "Initialized a new sequence-classification head for "
            f"{num_labels} labels; it will now be fine-tuned on the training "
            "split."
        )
    return model


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,  # Cắt bớt các chuỗi dài hơn max_len
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(tokenizer):
    def collate_batch(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        labels = torch.stack(labels).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    return collate_batch

def fine_tune_bert(
    texts,
    labels,
    num_labels,
    epochs=10,
    batch_size=8,
    max_len=512,
    learning_rate=2e-5,
    save_dir="./chkpt",
    validation_texts=None,
    validation_labels=None,
    model_name_or_path="answerdotai/ModernBERT-base",
    fine_tune=True,
):
    device = get_device()
    print(f"Using device: {device}")
    checkpoint_path = Path(save_dir) / "bert_last_checkpoint.pt"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=checkpoint_path.is_file(),
    )
    if tokenizer.pad_token_id is None:
        fallback_token = tokenizer.eos_token or tokenizer.sep_token
        if fallback_token is None:
            raise RuntimeError(
                f"Tokenizer for {model_name_or_path} has no padding fallback token"
            )
        tokenizer.pad_token = fallback_token

    if not fine_tune:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            attn_implementation="eager",
        ).to(device)
        model.requires_grad_(False)
        model.eval()
        print(f"Using frozen pretrained BERT encoder: {model_name_or_path}")
        return model, tokenizer

    if checkpoint_path.is_file():
        print(f"Checkpoint found at {checkpoint_path}. Loading checkpoint.")
        # The checkpoint contains the complete fine-tuned state. Constructing
        # the architecture from config avoids loading the base checkpoint and
        # emitting a misleading warning about a newly initialized classifier.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_model = checkpoint.get("model_name_or_path")
        if checkpoint_model is not None and checkpoint_model != model_name_or_path:
            raise RuntimeError(
                f"Checkpoint encoder is {checkpoint_model}, requested "
                f"{model_name_or_path}; use a separate cache or force BERT retraining"
            )
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_config(
            config,
            attn_implementation="eager",
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(
            f"Loaded BERT checkpoint selected at epoch {checkpoint.get('epoch', '?')}"
        )
        return model, tokenizer

    dataset = CustomDataset(texts, labels, tokenizer, max_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn(tokenizer),
    )
    valid_loader = None
    if validation_texts is not None and len(validation_texts):
        valid_dataset = CustomDataset(
            validation_texts,
            validation_labels,
            tokenizer,
            max_len,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn(tokenizer),
        )
    model = _load_classifier_for_fine_tuning(model_name_or_path, num_labels)
    model = model.to(device) 

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_metric = float("inf")
    best_state = None
    stale_epochs = 0
    patience = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"Loss": total_loss / (progress_bar.n + 1)})

        train_loss = total_loss / len(loader)
        metric = train_loss
        message = f"BERT epoch {epoch + 1}: train loss={train_loss:.6f}"
        if valid_loader is not None:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    valid_loss += outputs.loss.item()
            metric = valid_loss / len(valid_loader)
            message += f", valid loss={metric:.6f}"
        print(message)

        if metric < best_metric - 1e-6:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state,
                    "validation_loss": metric,
                    "model_name_or_path": model_name_or_path,
                },
                str(checkpoint_path),
            )
        else:
            stale_epochs += 1
            if valid_loader is not None and stale_epochs >= patience:
                print(f"BERT early stopped at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"BERT best selection loss: {best_metric:.6f}")

    return model, tokenizer

def get_bert_embeddings(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.base_model(**inputs)
    attention_mask = inputs["attention_mask"].unsqueeze(-1).to(
        dtype=outputs.last_hidden_state.dtype
    )
    token_sum = (outputs.last_hidden_state * attention_mask).sum(dim=1)
    token_count = attention_mask.sum(dim=1).clamp_min(1.0)
    return token_sum / token_count


def _l2_normalize_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, np.finfo(np.float32).eps)


def _fit_birch_with_exact_cluster_count(
    embeddings,
    num_topics,
    *,
    initial_threshold=0.5,
    minimum_threshold=0.01,
):
    """Adapt BIRCH threshold until global clustering can produce k topics."""
    threshold = initial_threshold
    while threshold >= minimum_threshold:
        subcluster_model = Birch(
            threshold=threshold,
            n_clusters=None,
        ).fit(embeddings)
        subcluster_count = len(subcluster_model.subcluster_centers_)
        print(
            f"Birch threshold={threshold:.4f}: "
            f"{subcluster_count} subclusters"
        )
        if subcluster_count >= num_topics:
            model = Birch(
                threshold=threshold,
                n_clusters=num_topics,
            ).fit(embeddings)
            cluster_count = len(np.unique(model.labels_))
            if cluster_count == num_topics:
                print(
                    f"Birch selected threshold={threshold:.4f} for "
                    f"{cluster_count} clusters"
                )
                return model
        threshold *= 0.8

    raise RuntimeError(
        "Birch could not produce "
        f"{num_topics} clusters even at threshold={minimum_threshold}. "
        "Use fewer topics or inspect duplicate/degenerate embeddings."
    )


def _select_topic_words(
    split_data,
    cluster_labels,
    num_topics,
    num_words,
    max_topics_per_word,
):
    """Select discriminative TF-IDF words and limit cross-topic duplication."""
    candidate_count = max(num_words * 3, num_words)
    ranked_candidates = []
    for topic_id in range(num_topics):
        cluster_indices = np.flatnonzero(cluster_labels == topic_id)
        cluster_texts = [' '.join(split_data[index]) for index in cluster_indices]
        cluster_texts = [text for text in cluster_texts if text.strip()]
        if not cluster_texts:
            print(f"No valid texts in cluster {topic_id}, skipping.")
            ranked_candidates.append([])
            continue

        vectorizer = TfidfVectorizer(
            max_features=candidate_count,
            stop_words='english',
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        except ValueError as exc:
            print(f"Error processing cluster {topic_id}: {exc}")
            ranked_candidates.append([])
            continue
        if tfidf_matrix.shape[1] == 0:
            print(f"Cluster {topic_id} contains only stop words or empty texts.")
            ranked_candidates.append([])
            continue

        # Mean TF-IDF makes scores more comparable across differently sized
        # clusters than a raw sum.
        scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        feature_names = vectorizer.get_feature_names_out()
        order = np.argsort(scores)[::-1]
        ranked_candidates.append(
            [(str(feature_names[index]), float(scores[index])) for index in order]
        )

    # Capacity-constrained greedy assignment: every topic can receive at most
    # ``num_words`` terms and every term can belong to at most
    # ``max_topics_per_word`` topics. Sorting primarily by within-topic rank
    # avoids starving smaller clusters whose TF-IDF score scale differs.
    candidate_edges = []
    for topic_id, candidates in enumerate(ranked_candidates):
        for rank, (word, score) in enumerate(candidates):
            candidate_edges.append((rank, -score, topic_id, word))
    candidate_edges.sort()

    topic_to_words = [[] for _ in range(num_topics)]
    word_assignment_counts = defaultdict(int)
    for _, _, topic_id, word in candidate_edges:
        if len(topic_to_words[topic_id]) >= num_words:
            continue
        if word_assignment_counts[word] >= max_topics_per_word:
            continue
        topic_to_words[topic_id].append(word)
        word_assignment_counts[word] += 1

    assignment_count = sum(len(words) for words in topic_to_words)
    unique_count = len(set().union(*(set(words) for words in topic_to_words)))
    topic_sizes = np.asarray([len(words) for words in topic_to_words])
    print(
        "Topic vocabulary: "
        f"{assignment_count} assignments, {unique_count} unique words, "
        f"per-topic min/median/max={topic_sizes.min()}/"
        f"{np.median(topic_sizes):.0f}/{topic_sizes.max()}"
    )
    return topic_to_words

def get_tbert_model(
    data_df,
    split_data,
    num_topics,
    num_words,
    cluster_method='Kmeans',
    validation_df=None,
    bert_cache_dir="./chkpt",
    embeddings_cache_path=None,
    max_topics_per_word=2,
    cluster_seed=42,
    bert_model="answerdotai/ModernBERT-base",
    bert_fine_tuning=True,
):
    device = get_device()
    if max_topics_per_word <= 0:
        raise ValueError("max_topics_per_word must be greater than zero")

    cleaned_data = data_df.dropna(
        subset=['filteredReviewText', 'overall_new']
    ).copy()
    cleaned_data['overall_new'] = cleaned_data['overall_new'].apply(lambda x: x - 1)

    texts = cleaned_data['filteredReviewText'].tolist()
    labels = cleaned_data['overall_new'].tolist()

    validation_texts = None
    validation_labels = None
    if validation_df is not None:
        cleaned_valid = validation_df.dropna(
            subset=["filteredReviewText", "overall_new"]
        ).copy()
        validation_texts = cleaned_valid["filteredReviewText"].tolist()
        validation_labels = (cleaned_valid["overall_new"] - 1).tolist()

    model, tokenizer = fine_tune_bert(
        texts,
        labels,
        num_labels=5,
        epochs=10,
        save_dir=str(bert_cache_dir),
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        model_name_or_path=bert_model,
        fine_tune=bert_fine_tuning,
    )
    model = model.to(device)
    
    # Batched inference keeps Metal busy and is substantially faster than one
    # BERT forward pass per review. Move completed embeddings back to CPU since
    # scikit-learn clustering runs on CPU.
    joined_texts = [' '.join(text) for text in split_data]
    cache_path = Path(embeddings_cache_path) if embeddings_cache_path else None
    if cache_path is not None and cache_path.is_file():
        embeddings_np = np.load(cache_path, allow_pickle=False)
        if embeddings_np.shape[0] != len(joined_texts):
            raise ValueError(
                f"Cached BERT embeddings contain {embeddings_np.shape[0]} rows; "
                f"expected {len(joined_texts)}"
            )
        print(f"Loaded cached BERT embeddings: {cache_path}")
    else:
        embeddings = []
        embedding_batch_size = 16
        model.eval()
        for start in tqdm(
            range(0, len(joined_texts), embedding_batch_size),
            desc="BERT embeddings",
        ):
            text_batch = joined_texts[start:start + embedding_batch_size]
            embedding = get_bert_embeddings(text_batch, tokenizer, model, device)
            embeddings.append(embedding.detach().cpu())
        embeddings_np = _l2_normalize_embeddings(torch.vstack(embeddings).numpy())
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
                    np.save(handle, embeddings_np, allow_pickle=False)
                temporary_path.replace(cache_path)
            except BaseException:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise
            print(f"Cached BERT embeddings: {cache_path}")

    # Euclidean clustering on unit vectors is equivalent to grouping by cosine
    # proximity and prevents review length/magnitude from dominating clusters.
    embeddings_np = _l2_normalize_embeddings(embeddings_np)

    # Clustering
    method_key = cluster_method.lower()
    if method_key == 'kmeans':
        print("Kmeans")
        clustering = KMeans(
            n_clusters=num_topics,
            random_state=cluster_seed,
        ).fit(embeddings_np)
    elif method_key == 'birch':
        print("Birch")
        clustering = _fit_birch_with_exact_cluster_count(
            embeddings_np,
            num_topics,
        )
    elif method_key == 'dbscan':
        print("DBSCAN")
        clustering = DBSCAN(eps=3, min_samples=num_topics).fit(embeddings_np)
    elif method_key == 'meanshift':
        print("MeanShift")
        clustering = MeanShift(bandwidth=num_topics).fit(embeddings_np)
    elif method_key == 'bisectingkmeans':
        print("BisectingKMeans")
        clustering = BisectingKMeans(
            n_clusters=num_topics,
            random_state=cluster_seed,
        ).fit(embeddings_np)
    else:
        raise ValueError(
            "cluster_method must be one of: KMeans, Birch, DBSCAN, "
            "MeanShift, BisectingKMeans"
        )

    labels = clustering.labels_
    if method_key in {"kmeans", "birch", "bisectingkmeans"}:
        cluster_count = len(np.unique(labels))
        if cluster_count != num_topics:
            raise RuntimeError(
                f"{cluster_method} produced {cluster_count} non-empty clusters; "
                f"expected exactly {num_topics}"
            )

    topic_to_words = _select_topic_words(
        split_data,
        labels,
        num_topics,
        num_words,
        max_topics_per_word,
    )
    
    return model, tokenizer, topic_to_words


analyzer = SentimentIntensityAnalyzer()

device = get_device()

def get_word_sentiment_score_by_vader(word):
    sentiment_dict = analyzer.polarity_scores(word)
    return sentiment_dict['compound']

def get_top_synonyms(word, top_n=4):
    noun_synsets = wn.synsets(word, pos=wn.NOUN)
    adj_synsets = wn.synsets(word, pos=wn.ADJ)
    
    all_synsets = noun_synsets + adj_synsets
    synonym_scores = []
    for synset in all_synsets:
        for lemma in synset.lemma_names():
            if lemma.lower() != word.lower() and lemma not in [syn[0] for syn in synonym_scores]:
                synonym_scores.append((lemma))
    
    return synonym_scores[:top_n]

def get_word_sentiment_score(word):
    m = list(swn.senti_synsets(word))
    s = 0
    if not m:
        return s  # Trả về 0 nếu không tìm thấy synset nào cho từ này
    for synset in m:
        s += get_word_sentiment_score_by_vader(synset.synset.name().split('.')[0])
    return s

def get_synonyms_sentiment_scores(word, top_n=4):
    synonyms = get_top_synonyms(word, top_n=top_n)
    scores = 0
    
    for synonym in synonyms:
        sentiment_score = get_word_sentiment_score(synonym)
        scores += sentiment_score

    scores = scores / top_n
    return scores

def get_topic_sentiment_matrix_tbert(text, topic_word_matrix, dependency_parser, topic_nums=50):
    topic_sentiment_m = torch.zeros(
        topic_nums,
        device=device,
        dtype=torch.float32,
    )
    try:
        sentences = preprocessed(text)
        dep_parser_result_p = []
        
        for i in sentences:
            dep_parser_result = dependency_parser.raw_parse(i)
            for j in dep_parser_result:
                dep_parser_result_p.append([j[0][0], j[2][0]])
                
        review_words = word_segment(text)
        for topic_id, cur_topic_words in enumerate(topic_word_matrix):
            cur_topic_senti_word = []
            for word in review_words:
                if word in cur_topic_words:
                    cur_topic_senti_word.append(word)
                    for p in dep_parser_result_p:
                        if p[0] == word:
                            cur_topic_senti_word.append(p[1])
                        if p[1] == word:
                            cur_topic_senti_word.append(p[0])

            if cur_topic_senti_word: 
                cur_topic_sentiment = sum(get_synonyms_sentiment_scores(senti_word) for senti_word in cur_topic_senti_word)
                # np.clip returns float64 by default, which Apple MPS cannot
                # represent. new_tensor inherits float32 and the MPS device.
                clipped_sentiment = float(np.clip(cur_topic_sentiment, -5, 5))
                topic_sentiment_m[topic_id] = topic_sentiment_m.new_tensor(
                    clipped_sentiment
                )
            else:
                topic_sentiment_m[topic_id] = 0.0
                
        return topic_sentiment_m
    except Exception as e:
        print("get_topic_sentiment_matrix_tbert's error: ", e, " text: ", text)
        return topic_sentiment_m
