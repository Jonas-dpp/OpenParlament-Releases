#!/usr/bin/env python3
"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

Training script for Stylometry classification.
Extracts features from the database, trains a RandomForestClassifier,
and saves the model to models/stylometry_clf.pkl.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import json
import hashlib
import subprocess
import random
from enum import Enum
from pathlib import Path
import uuid

# Ensure project root is on sys.path when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("train_stylometry")

import pandas as pd
import numpy as np
import joblib
import sklearn
from sqlalchemy import func, select
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.base import clone

from src.database import get_session
from src.models import Rede, Redner, Sitzung
from src.nlp import StylometryEngine, _canonicalise_fraktion

MODEL_DIR = _PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "stylometry_clf.pkl"
CSV_DIR = _PROJECT_ROOT / "data" / "csv"
JSON_DIR = _PROJECT_ROOT / "data" / "json"

class TextLayer(Enum):
    RAW = "raw"
    CLEAN = "clean"
    NORMALIZED = "normalized"


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_df_hash(dataframe):
    try:
        hash_val = hashlib.sha256(pd.util.hash_pandas_object(dataframe, index=True).values).hexdigest()
        return hash_val
    except Exception:
        return "unknown"


def load_data_and_extract_features(text_layer: TextLayer):
    """Loads speeches and extracts stylometric features."""
    engine = StylometryEngine()
    # Map the text layer Enum to the corresponding model attribute
    column_attr = Rede.text
    if text_layer == TextLayer.RAW:
        column_attr = Rede.raw_text
    elif text_layer == TextLayer.CLEAN:
        column_attr = Rede.clean_text
    elif text_layer == TextLayer.NORMALIZED:
        column_attr = Rede.normalized_text

    logger.debug("Starting to load speeches from the database using text layer: %s...", text_layer.value)

    with get_session() as session:
        # Load candidate speeches sorted by rede_id for strict determinism
        # Fall back to Rede.text if chosen column is NULL (e.g. for unmigrated entries)
        stmt = (
            select(
                func.coalesce(column_attr, Rede.text),
                Redner.fraktion,
                Rede.redner_id,
                Sitzung.wahlperiode,
                Sitzung.datum,
                Rede.wortanzahl
            )
            .join(Redner, Rede.redner_id == Redner.redner_id)
            .join(Sitzung, Rede.sitzung_id == Sitzung.sitzungs_id)
            .where((func.coalesce(column_attr, Rede.text)).isnot(None))
            .where(Redner.fraktion.isnot(None))
            .order_by(Rede.rede_id)
        )
        rows = session.execute(stmt).fetchall()

    if not rows:
        logger.warning("No rows fetched!")
        return pd.DataFrame()

    logger.debug("First row sample: %s", rows[0])

    # Pre-filter rows to remove invalid/fraktionslos before extracting features to save time!
    valid_rows = []
    total_rows = len(rows)
    filter_start = time.time()
    for idx, (text, fraktion, redner_id, wahlperiode, datum, wortanzahl) in enumerate(rows, start=1):
        canonical_frac = _canonicalise_fraktion(fraktion)
        if canonical_frac not in ["fraktionslos", "N/A", ""]:
            year = datum.year if datum else 2026
            valid_rows.append((text, canonical_frac, redner_id, wahlperiode, year, wortanzahl or 0))

        # Throttled logging to prevent I/O bottlenecks when there are many rows
        if total_rows > 1000 and (idx == 1 or idx % 5000 == 0 or idx == total_rows):
            elapsed = time.time() - filter_start
            avg_time = elapsed / idx
            eta = avg_time * (total_rows - idx)
            if eta >= 3600:
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
            else:
                eta_str = time.strftime('%M:%S', time.gmtime(eta))
            logger.info(
                f"SpeechFiltering: {idx}/{total_rows} | {int((idx/total_rows)*100)}% | ETA: {eta_str}"
            )
            
    if not valid_rows:
        return pd.DataFrame()

    # Deterministic sampling of up to 50000 candidate speeches to guarantee reproducible training splits
    rng = random.Random(42)
    if len(valid_rows) > 50000:
        logger.info("Sampling 50000 speeches from %d candidates deterministically...", len(valid_rows))
        valid_rows = rng.sample(valid_rows, 50000)
    else:
        logger.info("Using all %d valid candidate speeches...", len(valid_rows))

    texts = [item[0] for item in valid_rows]
    factions = [item[1] for item in valid_rows]
    redner_ids = [item[2] for item in valid_rows]
    wahlperioden = [item[3] for item in valid_rows]
    years = [item[4] for item in valid_rows]
    wortanzahls = [item[5] for item in valid_rows]

    logger.info(
        "CHECK: Normalized faction distribution:\n%s",
        pd.Series(factions).value_counts()
    )

    logger.info("Extracting features from %d speeches in batches...", len(texts))

    batch_size = 256
    all_features = []

    total_items = len(texts)
    total_batches = (total_items + batch_size - 1) // batch_size

    start_time = time.time()

    for batch_num, start_idx in enumerate(range(0, total_items, batch_size), start=1):
        batch_texts = texts[start_idx:start_idx + batch_size]

        try:
            batch_results = engine.extract_features_batch(
                batch_texts,
                batch_size=batch_size
            )
        except Exception:
            logger.exception(
                "Feature extraction failed for batch %d/%d (items %d-%d)",
                batch_num,
                total_batches,
                start_idx,
                min(start_idx + batch_size, total_items),
            )
            raise

        all_features.extend(batch_results)

        # Progress / ETA calculation
        elapsed = time.time() - start_time
        processed = min(start_idx + batch_size, total_items)
        avg_time = elapsed / processed
        remaining = total_items - processed
        eta = avg_time * remaining
        eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))

        logger.info(
            "FeatureExtraction: %d/%d | %d%% | ETA: %s",
            processed,
            total_items,
            int((processed / total_items) * 100),
            eta_str
        )


    # Safety check
    if len(all_features) != len(factions):
        logger.warning(
            "Feature/text mismatch: extracted %d feature sets for %d speeches",
            len(all_features),
            len(factions),
        )


    records = []

    p_map = {
        "?": "punct_question",
        "!": "punct_exclamation",
        ".": "punct_period",
        ",": "punct_comma",
        ":": "punct_colon",
        ";": "punct_semicolon",
        "-": "punct_dash",
        "(": "punct_paren",
        ")": "punct_paren",
        "\"": "punct_quote",
        "...": "punct_ellipsis",
    }

    for i, features in enumerate(all_features):
        features = features.copy()
        punct = features.pop("punctuation_freq", {})

        for p_char, col_name in p_map.items():
            features[col_name] = punct.get(p_char, 0)

        features["fraktion"] = factions[i]
        features["speaker_id"] = redner_ids[i]
        features["wahlperiode"] = wahlperioden[i]
        features["year"] = years[i]
        features["speech_length"] = wortanzahls[i]
        records.append(features)

    return pd.DataFrame(records)


def train_model():
    parser = argparse.ArgumentParser(description="Train stylometry classification model.")
    parser.add_argument(
        "--text-layer",
        type=str,
        choices=[item.value for item in TextLayer],
        default="clean",
        help="The text layer to use as training source (raw, clean, or normalized)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rf", "extratrees", "histgb", "dummy", "logreg", "linearsvm", "all"],
        default="all",
        help="The classifier model to train/compare (all will run a comparison benchmark)."
    )
    args = parser.parse_args()

    valid_text_layers = [item.value for item in TextLayer]
    if args.text_layer not in valid_text_layers:
        raise ValueError(f"text-layer must be one of {valid_text_layers}")

    valid_model_types = ["rf", "extratrees", "histgb", "dummy", "logreg", "linearsvm", "all"]
    if args.model_type not in valid_model_types:
        raise ValueError(f"model-type must be one of {valid_model_types}")

    chosen_layer = TextLayer(args.text_layer)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    
    run_id = str(uuid.uuid4())
    
    start_time = time.perf_counter()
    df = load_data_and_extract_features(chosen_layer)
    if df.empty:
        logger.error("No data available for training.")
        return

    logger.debug("DataFrame shape: %s", df.shape)
    logger.debug("Missing values per column:\n%s", df.isna().sum()[df.isna().sum() > 0])

    logger.info("Splitting dataset...")

    # Speaker-independent train/test split using GroupShuffleSplit
    use_group_split = True
    if df["fraktion"].nunique() < 2:
        logger.error("Not enough classes for split.")
        return
    if df["speaker_id"].nunique() < 5 or df.shape[0] < 10:
        logger.warning("Too few samples or speakers for GroupShuffleSplit. Falling back to standard StratifiedKFold.")
        use_group_split = False

    if use_group_split:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(df, df["fraktion"], groups=df["speaker_id"]))
        except Exception as e:
            logger.warning("GroupShuffleSplit failed (%s), falling back to standard train_test_split.", e)
            use_group_split = False

    if not use_group_split:
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df["fraktion"])

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # Feature Engineering: Speaker Normalization (computed strictly on the train split to avoid leakage!)
    base_feature_cols = [col for col in df.columns if col not in ["fraktion", "speaker_id", "wahlperiode", "year", "speech_length"]]
    logger.info("Computing speaker-relative features strictly on the training split...")

    # Store speaker average values for base features calculated solely on the training split
    speaker_averages = df_train.groupby("speaker_id")[base_feature_cols].mean().to_dict(orient="index")
    
    # Vectorized computation of relative features (maps training speaker averages, unseen test speakers get 0.0 automatically!)
    train_avgs_map = {}
    for col in base_feature_cols:
        train_avgs_map[col] = df_train.groupby("speaker_id")[col].mean().to_dict()

    for col in base_feature_cols:
        col_map = train_avgs_map[col]

        # Training relative features
        train_avg_series = df_train["speaker_id"].map(col_map).fillna(df_train[col])
        df_train[f"speaker_relative_{col}"] = df_train[col] - train_avg_series

        # Test relative features (using train averages, unseen defaults to test values to make relative 0.0)
        test_avg_series = df_test["speaker_id"].map(col_map).fillna(df_test[col])
        df_test[f"speaker_relative_{col}"] = df_test[col] - test_avg_series

    logger.info("Training classifier on %d samples across %d factions.", len(df_train), df_train["fraktion"].nunique())
    
    X_train = df_train.drop(columns=["fraktion", "speaker_id", "wahlperiode", "year"])
    y_train = df_train["fraktion"]
    groups_train = df_train["speaker_id"]

    X_test = df_test.drop(columns=["fraktion", "speaker_id", "wahlperiode", "year"])
    y_test = df_test["fraktion"]
    
    logger.debug("Class distribution in train split:\n%s", y_train.value_counts())
    
    candidate_clfs = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        ),
        "LinearSVM": LinearSVC(
            class_weight="balanced",
            C=1.0,
            dual="auto",
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=12,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        )
    }

    if args.model_type != "all":
        name_map = {
            "rf": "RandomForest",
            "extratrees": "ExtraTrees",
            "histgb": "HistGradientBoosting",
            "dummy": "Dummy",
            "logreg": "LogisticRegression",
            "linearsvm": "LinearSVM"
        }
        chosen_name = name_map[args.model_type]
        candidate_clfs = {chosen_name: candidate_clfs[chosen_name]}

    logger.info("Starting Cross-Validation Benchmark on Train Split...")

    if use_group_split:
        cv_splitter = StratifiedGroupKFold(n_splits=min(5, y_train.nunique()))
    else:
        cv_splitter = StratifiedKFold(n_splits=min(5, y_train.nunique()), shuffle=True, random_state=42)

    benchmark_results = []

    total_splits = len(candidate_clfs) * cv_splitter.n_splits
    cv_start_time = time.time()
    split_count = 0

    for clf_name, clf_obj in candidate_clfs.items():
        logger.info("Evaluating %s with %d-Fold CV...", clf_name, cv_splitter.n_splits)
        cv_accs = []
        cv_precs = []
        cv_recs = []
        cv_f1s = []
        cv_weighted_f1s = []

        fit_times = []
        pred_times = []

        try:
            for fold, (f_train_idx, f_val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups=groups_train), 1):
                X_fold_train, X_fold_val = X_train.iloc[f_train_idx], X_train.iloc[f_val_idx]
                y_fold_train, y_fold_val = y_train.iloc[f_train_idx], y_train.iloc[f_val_idx]

                fold_clf = clone(clf_obj)

                fold_fit_start = time.perf_counter()
                fold_clf.fit(X_fold_train, y_fold_train)
                fit_times.append(time.perf_counter() - fold_fit_start)

                fold_pred_start = time.perf_counter()
                y_fold_pred = fold_clf.predict(X_fold_val)
                pred_times.append(time.perf_counter() - fold_pred_start)

                fold_acc = accuracy_score(y_fold_val, y_fold_pred)
                fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(
                    y_fold_val, y_fold_pred, average="macro", zero_division=0
                )
                _, _, fold_w_f1, _ = precision_recall_fscore_support(
                    y_fold_val, y_fold_pred, average="weighted", zero_division=0
                )

                cv_accs.append(fold_acc)
                cv_precs.append(fold_prec)
                cv_recs.append(fold_rec)
                cv_f1s.append(fold_f1)
                cv_weighted_f1s.append(fold_w_f1)

                split_count += 1
                elapsed = time.time() - cv_start_time
                avg_time = elapsed / split_count
                remaining_splits = total_splits - split_count
                eta = avg_time * remaining_splits
                eta_str = time.strftime('%M:%S', time.gmtime(eta)) if eta < 3600 else time.strftime('%H:%M:%S', time.gmtime(eta))

                logger.info(
                    "StylometryBenchmark: %d/%d folds | %d%% | ETA: %s",
                    split_count,
                    total_splits,
                    int((split_count / total_splits) * 100),
                    eta_str
                )

            mean_f1 = np.mean(cv_f1s)
            mean_acc = np.mean(cv_accs)
            mean_fit_time = np.mean(fit_times)
            mean_pred_time = np.mean(pred_times)

            benchmark_results.append({
                "model": clf_name,
                "mean_acc": mean_acc,
                "std_acc": np.std(cv_accs),
                "mean_macro_f1": mean_f1,
                "std_macro_f1": np.std(cv_f1s),
                "mean_weighted_f1": np.mean(cv_weighted_f1s),
                "mean_precision": np.mean(cv_precs),
                "mean_recall": np.mean(cv_recs),
                "avg_fit_time": mean_fit_time,
                "avg_pred_time": mean_pred_time
            })

            logger.info(
                "%s Benchmark Results - Acc: %.4f ± %.4f | Macro-F1: %.4f ± %.4f | Weighted-F1: %.4f | Fit: %.2fs | Pred: %.4fs",
                clf_name, mean_acc, np.std(cv_accs), mean_f1, np.std(cv_f1s), np.mean(cv_weighted_f1s), mean_fit_time, mean_pred_time
            )
        except Exception as e:
            logger.exception("Failed CV evaluation for %s: %s", clf_name, e)

    benchmark_df = pd.DataFrame(benchmark_results)
    try:
        logger.info("\n=== MODEL COMPARISON TABLE ===\n%s", benchmark_df.to_markdown(index=False))
    except Exception:
        logger.info("\n=== MODEL COMPARISON TABLE ===\n%s", benchmark_df.to_string(index=False))

    best_row = benchmark_df.sort_values(by="mean_macro_f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    logger.info("Best model selected: %s (Macro-F1: %.4f ± %.4f)", best_model_name, best_row["mean_macro_f1"], best_row["std_macro_f1"])

    logger.info("Fitting best model %s on 80%% train split...", best_model_name)
    best_clf_template = candidate_clfs[best_model_name]
    clf = clone(best_clf_template)

    fit_start_time = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_duration = time.perf_counter() - fit_start_time

    importance_df = pd.DataFrame()
    if hasattr(clf, "feature_importances_"):
        importance = pd.Series(
            clf.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        importance_df = (
            importance.rename("importance")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["timestamp"] = pd.Timestamp.now()
        importance_df["run_id"] = run_id
        importance_df["text_layer"] = chosen_layer.value
        importance_df["classifier"] = best_model_name
        importance_df["samples"] = len(df)

        feature_csv = CSV_DIR / "feature_importance_history.csv"
        importance_df.to_csv(
            feature_csv,
            mode="a",
            header=not feature_csv.exists(),
            index=False
        )

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    _, _, w_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    total_duration = time.perf_counter() - start_time

    logger.info("\n=== SCIENTIFIC EVALUATION REPORT (Layer: %s) ===", chosen_layer.value)
    logger.info("Classifier: %s", best_model_name)
    logger.info("Accuracy:    %.4f", acc)
    logger.info("Precision:   %.4f", prec)
    logger.info("Recall:      %.4f", rec)
    logger.info("Macro-F1:    %.4f", f1)
    logger.info("Weighted-F1: %.4f", w_f1)
    logger.info("Fit duration: %.2f seconds", fit_duration)
    logger.info("Total duration: %.2f seconds", total_duration)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    
    git_commit = get_git_commit()
    python_version = sys.version.split()[0]
    sklearn_version = sklearn.__version__
    dataset_hash = compute_df_hash(df)
    feature_hash = compute_df_hash(X_train)

    run_df = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "run_id": run_id,
        "text_layer": chosen_layer.value,
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "weighted_f1": w_f1,
        "samples": len(df),
        "features": X_train.shape[1],
        "classes": y_train.nunique(),
        "classifier": best_model_name,
        "fit_time_seconds": fit_duration,
        "total_time_seconds": total_duration,
        "git_commit": git_commit,
        "python_version": python_version,
        "sklearn_version": sklearn_version,
        "dataset_hash": dataset_hash,
        "feature_hash": feature_hash,
        "random_seed": 42
    }])

    run_csv = CSV_DIR / "training_history.csv"
    run_df.to_csv(
        run_csv,
        mode="a",
        header=not run_csv.exists(),
        index=False
    )

    perm_importance = pd.Series(0.0, index=X_train.columns)
    try:
        logger.info("Calculating Permutation Importance on test set (10 repeats)...")
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            clf,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        perm_importance = pd.Series(
            perm.importances_mean,
            index=X_train.columns
        ).sort_values(ascending=False)

        perm_df = (
            perm_importance.rename("importance")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        perm_df["rank"] = range(1, len(perm_df) + 1)
        perm_df["timestamp"] = pd.Timestamp.now()
        perm_df["run_id"] = run_id
        perm_df["text_layer"] = chosen_layer.value
        perm_df["classifier"] = best_model_name
        perm_df["samples"] = len(df)

        perm_csv = CSV_DIR / "permutation_importance_history.csv"
        perm_df.to_csv(
            perm_csv,
            mode="a",
            header=not perm_csv.exists(),
            index=False
        )

        logger.info(
            "\n=== TOP 30 PERMUTATION IMPORTANCE ===\n%s",
            perm_importance.head(30)
        )
    except Exception as e:
        logger.warning("Could not calculate permutation importance: %s", e)

    if not importance_df.empty:
        logger.info(
            "\n=== TOP 30 FEATURE IMPORTANCE ===\n%s",
            importance.head(30)
        )

    # Save globally/legacy for backward compatibility
    joblib.dump(clf, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # Save exact feature names to data/json/stylometry_features.json
    feature_names_list = list(X_train.columns)
    features_json_path = JSON_DIR / "stylometry_features.json"
    features_json_path.write_text(json.dumps(feature_names_list, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Feature names saved to %s", features_json_path)

    # Save metadata to data/json/stylometry_metadata.json
    metadata_payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "run_id": run_id,
        "text_layer": chosen_layer.value,
        "classifier": best_model_name,
        "sample_count": len(df),
        "feature_count": len(feature_names_list),
        "feature_names": feature_names_list,
        "metrics": {
            "accuracy": float(acc),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
            "weighted_f1": float(w_f1),
        },
        "fit_time_seconds": float(fit_duration),
        "total_time_seconds": float(total_duration),
        "git_commit": git_commit,
        "python_version": python_version,
        "sklearn_version": sklearn_version,
        "dataset_hash": dataset_hash,
        "feature_hash": feature_hash,
        "random_seed": 42
    }
    metadata_json_path = JSON_DIR / "stylometry_metadata.json"
    metadata_json_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Model metadata saved to %s", metadata_json_path)

    # Also save in the new structured directory format
    new_model_dir = MODEL_DIR / "stylometry" / best_model_name
    new_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, new_model_dir / "model.pkl")
    (new_model_dir / "features.json").write_text(json.dumps(feature_names_list, indent=2, ensure_ascii=False), encoding="utf-8")
    (new_model_dir / "metrics.json").write_text(json.dumps(metadata_payload["metrics"], indent=2, ensure_ascii=False), encoding="utf-8")
    (new_model_dir / "metadata.json").write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save speaker averages to new directory and globally
    (MODEL_DIR / "stylometry" / "speaker_averages.json").write_text(json.dumps(speaker_averages, indent=2, ensure_ascii=False), encoding="utf-8")
    (JSON_DIR / "speaker_averages.json").write_text(json.dumps(speaker_averages, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved structured files in %s", new_model_dir)


if __name__ == "__main__":
    train_model()
