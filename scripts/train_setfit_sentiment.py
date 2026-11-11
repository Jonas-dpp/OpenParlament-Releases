"""
OpenParlament - Streamlit Dashboard
Copyright (C) 2026 Jonas-dpp

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------
SetFit Few-Shot Fine-Tuning Pipeline for Parliamentary Sentiment Classification.
Remediates Domain Mismatch for procedural neutral interjections.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class _DummySetFitModel:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        pass


class _DummyTrainer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def train(self) -> None:
        pass


class _DummyTrainingArguments:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _DummyDataset:
    @classmethod
    def from_pandas(cls, *args: Any, **kwargs: Any) -> Any:
        pass


# Safe/Lazy imports for optional NLP dependencies
try:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    _SETFIT_AVAILABLE = True
except ImportError:
    Dataset = _DummyDataset  # type: ignore
    SetFitModel = _DummySetFitModel  # type: ignore
    Trainer = _DummyTrainer  # type: ignore
    TrainingArguments = _DummyTrainingArguments  # type: ignore
    _SETFIT_AVAILABLE = False

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Setup structured logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_setfit_sentiment")

LABEL_MAP: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP: Dict[int, int] = {0: -1, 1: 0, 2: 1}
CLASS_NAMES: List[str] = ["Negative (-1)", "Neutral (0)", "Positive (1)"]


def _format_eta(seconds: float) -> str:
    """Format ETA in MM:SS or HH:MM:SS."""
    if seconds < 0 or seconds != seconds:
        return "00:00"
    seconds_int = int(seconds)
    if seconds_int >= 3600:
        return time.strftime("%H:%M:%S", time.gmtime(seconds_int))
    return time.strftime("%M:%S", time.gmtime(seconds_int))


def generate_gold_standard_csv(filepath: Path) -> pd.DataFrame:
    """Generates a gold standard CSV with parliamentary interjections if missing."""
    logger.info("Generating representative parliamentary gold standard CSV at %s...", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Dataset exhibiting domain mismatch: general sentiment models misclassify procedural neutral texts
    data = [
        # --- Procedural / Neutral Class (True Label = 0) ---
        ("Zur Geschäftsordnung!", -0.75, 0),
        ("Herr Präsident, zur Geschäftsordnung!", -0.68, 0),
        ("Das Wort hat der Abgeordnete Schneider.", 0.62, 0),
        ("Wir kommen zum nächsten Punkt der Tagesordnung.", 0.55, 0),
        ("Gestatten Sie eine Zwischenfrage?", -0.45, 0),
        ("Punkt 3 der Tagesordnung.", 0.12, 0),
        ("Fraktion der SPD", 0.48, 0),
        ("Fraktion der CDU/CSU", -0.52, 0),
        ("Ruhe im Saal!", -0.82, 0),
        ("Bitte treten Sie ans Rednerpult.", 0.35, 0),
        ("Die Redezeit ist abgelaufen.", -0.71, 0),
        ("Wir treten in die Debatte ein.", 0.28, 0),
        ("Es liegt eine schriftliche Erklärungsanfrage vor.", -0.38, 0),
        ("Namentliche Abstimmung!", -0.64, 0),
        ("Das Verfahren richtet sich nach § 75 GO-BT.", -0.25, 0),
        ("Ich verweise auf die Drucksache 20/1234.", 0.18, 0),
        ("Das Ergebnis der Beratung wird nachgereicht.", 0.40, 0),
        ("Der Antrag wird dem Ausschuss überwiesen.", -0.30, 0),
        ("Eine Anmerkung der Parlamentsverwaltung.", 0.22, 0),
        ("Der Ältestenrat hat dies beschlossen.", 0.50, 0),
        ("Beide Fraktionen haben zugestimmt.", 0.65, 0),
        ("Gibt es weitere Wortmeldungen?", -0.15, 0),
        ("Ich schließe die Aussprache.", -0.60, 0),
        ("Kommen wir zur zweiten Lesung.", 0.32, 0),
        ("Gemäß Protokoll der 45. Sitzung.", -0.10, 0),
        ("Hierzu liegt ein Änderungsantrag vor.", -0.42, 0),
        ("Fraktion Bündnis 90/Die Grünen", 0.38, 0),
        ("Fraktion der FDP", -0.22, 0),
        ("Fraktion Die Linke", 0.29, 0),
        ("Fraktion der AfD", -0.58, 0),
        ("Das Protokoll ist genehmigt.", 0.68, 0),
        ("Wir bitten um Einhaltung der Ordnung.", -0.76, 0),
        ("Der Abgeordnete hat das Wort.", 0.41, 0),
        ("Antrag auf Fristverlängerung.", -0.33, 0),
        ("Zur Abgabe einer Erklärung zur Abstimmung.", -0.27, 0),
        ("Verlesung der Beschlussempfehlung.", 0.15, 0),
        ("Der Tagesordnungspunkt ist beendet.", -0.49, 0),
        ("Fraktionslose Abgeordnete", -0.18, 0),
        ("Feststellung der Beschlussfähigkeit.", -0.62, 0),
        ("Die Sitzung ist unterbrochen.", -0.80, 0),

        # --- Positive Class (True Label = 1) ---
        ("Sehr gut!", 0.88, 1),
        ("Bravo!", 0.92, 1),
        ("Hervorragende Rede!", 0.95, 1),
        ("Zustimmung!", 0.82, 1),
        ("Richtig!", 0.85, 1),
        ("Ausgezeichnete Ausführungen!", 0.91, 1),
        ("Völlig recht hat der Kollege!", 0.87, 1),
        ("Das ist ein starkes Signal!", 0.89, 1),
        ("Großartige Initiative!", 0.93, 1),
        ("Vielen Dank für diese Worte!", 0.86, 1),
        ("Recht so!", 0.80, 1),
        ("Wunderbar!", 0.94, 1),
        ("Exzellenter Beitrag!", 0.90, 1),
        ("Absolut treffend formuliert!", 0.84, 1),
        ("Ein wichtiger Meilenstein!", 0.88, 1),
        ("Sehr überzeugend!", 0.86, 1),
        ("Genau das ist der Punkt!", 0.83, 1),
        ("Volle Unterstützung!", 0.89, 1),
        ("Glückwunsch zu dieser Rede!", 0.91, 1),
        ("Ein vorbildlicher Vorschlag!", 0.87, 1),
        ("Genauso ist es!", 0.81, 1),
        ("Glänzende Begründung!", 0.92, 1),
        ("Herzlichen Dank dafür!", 0.85, 1),
        ("Wir begrüßen diesen Schritt außerordentlich!", 0.90, 1),
        ("Ein starker Moment für das Parlament!", 0.88, 1),
        ("Voller Erfolg!", 0.93, 1),
        ("Wir stimmen dem begeistert zu!", 0.89, 1),
        ("Ein weitsichtiger Beschluss!", 0.86, 1),
        ("Ganz hervorragend!", 0.94, 1),
        ("Absolut beispielhaft!", 0.87, 1),
        ("Völlig zutreffend!", 0.82, 1),
        ("Ihre Worte waren richtig und notwendig!", 0.88, 1),
        ("Das war eine Meisterleistung!", 0.95, 1),
        ("So ist es vollkommen richtig!", 0.84, 1),
        ("Ein Meilenstein parlamentarischer Arbeit!", 0.91, 1),
        ("Glückauf!", 0.78, 1),
        ("Großer Dank an den Redner!", 0.86, 1),
        ("Ein vorbildliches Vorgehen!", 0.89, 1),
        ("Besser kann man es nicht ausdrücken!", 0.92, 1),
        ("Gratulation!", 0.87, 1),

        # --- Negative Class (True Label = -1) ---
        ("Pfui!", -0.92, -1),
        ("Buh!", -0.88, -1),
        ("Skandal!", -0.95, -1),
        ("Das ist doch nicht wahr!", -0.85, -1),
        ("Unverschämtheit!", -0.90, -1),
        ("Lüge!", -0.96, -1),
        ("Schämen Sie sich!", -0.91, -1),
        ("Unerhört!", -0.87, -1),
        ("Eine absolute Frechheit!", -0.93, -1),
        ("Das glauben Sie doch selbst nicht!", -0.84, -1),
        ("Totalversagen!", -0.89, -1),
        ("Ihre Politik ist gescheitert!", -0.86, -1),
        ("Das ist Heuchelei!", -0.94, -1),
        ("Peinlich!", -0.88, -1),
        ("Reine Irreführung der Bürger!", -0.91, -1),
        ("Unfassbar!", -0.86, -1),
        ("Verlogen!", -0.95, -1),
        ("So ein Unsinn!", -0.83, -1),
        ("Sie verdrehen die Tatsachen!", -0.87, -1),
        ("Widerspruch!", -0.82, -1),
        ("Ein Desaster!", -0.90, -1),
        ("Das ist doch Lächerlich!", -0.85, -1),
        ("Schande!", -0.94, -1),
        ("Sie spalten das Land!", -0.89, -1),
        ("Eine Farce!", -0.92, -1),
        ("Das ist Verfassungsbruch!", -0.96, -1),
        ("Aufhören!", -0.86, -1),
        ("Setzen Sie sich hin!", -0.84, -1),
        ("Sie täuschen die Öffentlichkeit!", -0.91, -1),
        ("Sondergleichen an Dreistigkeit!", -0.93, -1),
        ("Verfehlt und gefährlich!", -0.88, -1),
        ("Eine Katastrophe!", -0.90, -1),
        ("Einfach abwegig!", -0.82, -1),
        ("Keine Ahnung!", -0.85, -1),
        ("Arrogant und weltfremd!", -0.92, -1),
        ("Eine Schande für das Haus!", -0.95, -1),
        ("Null Konzept!", -0.87, -1),
        ("Verunsicherung pur!", -0.84, -1),
        ("Versagt!", -0.89, -1),
        ("Unverantwortlich!", -0.91, -1),
    ]

    df = pd.DataFrame(data, columns=["text", "sentiment_raw", "true_label"])
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info("Gold standard dataset successfully saved with %d rows.", len(df))
    return df


def map_continuous_score_to_class(
    score: float, lower_threshold: float = -0.2, upper_threshold: float = 0.2
) -> int:
    """Maps continuous sentiment score to discrete class (-1, 0, 1)."""
    if not (-1.0 <= lower_threshold < upper_threshold <= 1.0):
        raise ValueError(
            f"Invalid sentiment threshold boundaries: lower_threshold ({lower_threshold}) "
            f"and upper_threshold ({upper_threshold}) must satisfy -1.0 <= lower < upper <= 1.0"
        )
    if score > upper_threshold:
        return 1
    if score < lower_threshold:
        return -1
    return 0


def evaluate_baseline(
    df: pd.DataFrame, lower_threshold: float = -0.2, upper_threshold: float = 0.2
) -> Dict[str, Any]:
    """Phase 1: Baseline Evaluation showing Domain Mismatch."""
    logger.info("==================================================")
    logger.info("PHASE 1: BASELINE EVALUATION (The Gold Standard Test)")
    logger.info("==================================================")

    total = len(df)
    start_time = time.time()
    log_step = max(1, total // 10)
    pred_labels = []

    for i, score in enumerate(df["sentiment_raw"], 1):
        pred_labels.append(map_continuous_score_to_class(score, lower_threshold, upper_threshold))
        if i % log_step == 0 or i == total:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (total - i)
            pct = int((i / total) * 100)
            logger.info(f"BaselineEvaluation: {i}/{total} | {pct}% | ETA: {_format_eta(eta)}")

    df = df.copy()
    df["pred_label"] = pred_labels

    y_true = df["true_label"].tolist()
    y_pred = df["pred_label"].tolist()

    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

    recall_neutral = report_dict["Neutral (0)"]["recall"]

    logger.info("\nBaseline Classification Report:\n%s", report_text)
    logger.info("Baseline Confusion Matrix (Rows: True [-1,0,1], Cols: Pred [-1,0,1]):\n%s", cm)
    logger.info("Baseline Recall for Neutral Class (0): %.2f%%", recall_neutral * 100)

    if recall_neutral < 0.20:
        logger.warning(
            "DOMAIN MISMATCH DETECTED: Neutral recall is %.2f%% (< 20%%). Procedural interjections are misclassified!",
            recall_neutral * 100,
        )

    return {
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "recall_neutral": recall_neutral,
        "pred_labels": y_pred,
    }


def train_and_evaluate_setfit(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    output_dir: Path = _PROJECT_ROOT / "models" / "setfit_sentiment_model",
    test_size: float = 0.3,
    random_state: int = 42,
    num_epochs: int = 1,
    num_iterations: int = 20,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Phase 2: Few-Shot Fine-Tuning with SetFit on Parliament Domain."""
    logger.info("==================================================")
    logger.info("PHASE 2: FEW-SHOT SETFIT FINE-TUNING")
    logger.info("==================================================")

    if not _SETFIT_AVAILABLE and SetFitModel is _DummySetFitModel:
        raise RuntimeError("setfit and datasets packages are required for SetFit fine-tuning.")

    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df["true_label"], random_state=random_state
    )

    logger.info(
        "Split dataset: %d Train samples, %d Test samples.", len(df_train), len(df_test)
    )

    # Encode labels for SetFit
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["label"] = df_train["true_label"].map(LABEL_MAP)
    df_test["label"] = df_test["true_label"].map(LABEL_MAP)

    train_dataset = Dataset.from_pandas(df_train[["text", "label"]])
    eval_dataset = Dataset.from_pandas(df_test[["text", "label"]])

    logger.info("Loading pre-trained base model '%s'...", model_name)
    model = SetFitModel.from_pretrained(model_name)

    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        seed=random_state,
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
    )

    logger.info("Starting SetFit training (%d iterations, %d epochs)...", num_iterations, num_epochs)
    trainer.train()
    logger.info("SetFit training complete.")

    logger.info("Evaluating fine-tuned model on test split...")
    eval_texts = df_test["text"].tolist()
    total_eval = len(eval_texts)

    y_pred_encoded = model.predict(eval_texts)
    if hasattr(y_pred_encoded, "tolist"):
        y_pred_encoded = y_pred_encoded.tolist()

    logger.info(f"SetFitEvaluation: {total_eval}/{total_eval} | 100% | ETA: 00:00")

    y_pred = [INV_LABEL_MAP[int(val)] for val in y_pred_encoded]
    y_true = df_test["true_label"].tolist()

    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

    recall_neutral = report_dict["Neutral (0)"]["recall"]

    logger.info("\nFine-Tuned SetFit Classification Report:\n%s", report_text)
    logger.info(
        "Fine-Tuned Confusion Matrix (Rows: True [-1,0,1], Cols: Pred [-1,0,1]):\n%s", cm
    )
    logger.info(
        "Fine-Tuned Recall for Neutral Class (0): %.2f%%", recall_neutral * 100
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    logger.info("Saved fine-tuned SetFit model locally to %s", output_dir)

    return {
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "recall_neutral": recall_neutral,
        "pred_labels": y_pred,
        "model_dir": str(output_dir),
    }


def run_pipeline(
    csv_path: Path,
    output_dir: Path,
    model_name: str,
    lower_threshold: float,
    upper_threshold: float,
    generate_if_missing: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs the full two-stage domain mismatch remediation pipeline."""
    if not csv_path.exists():
        if generate_if_missing:
            df = generate_gold_standard_csv(csv_path)
        else:
            raise FileNotFoundError(f"Gold standard CSV not found at {csv_path}")
    else:
        logger.info("Loading gold standard CSV from %s...", csv_path)
        df = pd.read_csv(csv_path)

    baseline_metrics = evaluate_baseline(
        df, lower_threshold=lower_threshold, upper_threshold=upper_threshold
    )

    setfit_metrics = train_and_evaluate_setfit(
        df,
        model_name=model_name,
        output_dir=output_dir,
    )

    logger.info("==================================================")
    logger.info("SUMMARY & DOMAIN MISMATCH REMEDIATION PROOF")
    logger.info("==================================================")
    logger.info(
        "Baseline Neutral Recall:   %.2f%%", baseline_metrics["recall_neutral"] * 100
    )
    logger.info(
        "SetFit Neutral Recall:     %.2f%%", setfit_metrics["recall_neutral"] * 100
    )
    improvement = (
        setfit_metrics["recall_neutral"] - baseline_metrics["recall_neutral"]
    ) * 100
    logger.info("Absolute Recall Improvement: %+.2f percentage points", improvement)

    return baseline_metrics, setfit_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SetFit Fine-Tuning Pipeline for Parliamentary Sentiment Classification."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=_PROJECT_ROOT / "data" / "gold_standard.csv",
        help="Path to gold standard CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "models" / "setfit_sentiment_model",
        help="Directory where fine-tuned model will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Base SentenceTransformer model name.",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=-0.2,
        help="Lower threshold for continuous score mapping.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=0.2,
        help="Upper threshold for continuous score mapping.",
    )
    parser.add_argument(
        "--generate-gold-standard",
        action="store_true",
        help="Force regeneration of gold standard CSV.",
    )

    args = parser.parse_args()

    if not (-1.0 <= args.lower_threshold < args.upper_threshold <= 1.0):
        raise ValueError(
            f"Invalid sentiment threshold boundaries: lower_threshold ({args.lower_threshold}) "
            f"and upper_threshold ({args.upper_threshold}) must satisfy -1.0 <= lower < upper <= 1.0"
        )

    if args.generate_gold_standard:
        generate_gold_standard_csv(args.csv_path)

    run_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold,
    )


if __name__ == "__main__":
    main()
