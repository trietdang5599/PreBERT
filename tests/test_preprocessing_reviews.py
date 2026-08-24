from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from preprocessing_reviews import (
    preprocess_batch,
    prohibited_content_labels,
    segment_review,
    write_preprocessing_audit,
)


class RejectUnknownClassifier:
    def classify_relevance(self, segments):
        return [False] * len(segments)

    def classify_polarity(self, texts):
        return [1] * len(texts)


class RecordingClassifier:
    def __init__(self):
        self.polarity_inputs = []

    def classify_relevance(self, segments):
        return [False] * len(segments)

    def classify_polarity(self, texts):
        self.polarity_inputs = list(texts)
        return [0] * len(texts)


def preprocess(text: str, domain_id: str = "item-1") -> str:
    rows = [{"asin": domain_id, "overall": 5, "reviewText": text}]
    output, _stats = preprocess_batch(
        rows,
        RejectUnknownClassifier(),
        {domain_id: 5.0},
        text_field="reviewText",
        rating_field="overall",
        item_field="asin",
        empty_review_policy="keep-original",
        adjust_ratings=False,
    )
    return output[0]["filteredReviewText"]


class PreprocessingSafetyTest(unittest.TestCase):
    def test_raw_ingredient_list_removed_but_experience_kept(self):
        filtered = preprocess(
            "It cleared my breakouts and works very well.\n\n"
            "Here are the ingredients:\n\n"
            "Water, Lactic Acid, Glycerin, Cetearyl Alcohol, "
            "Salicylic Acid, Phenoxyethanol."
        )
        self.assertIn("cleared my breakouts", filtered)
        self.assertNotIn("Here are the ingredients", filtered)
        self.assertNotIn("Cetearyl Alcohol", filtered)
        self.assertEqual([], prohibited_content_labels(filtered))

    def test_disclosure_not_restored_by_keep_original_policy(self):
        filtered = preprocess(
            "I received this toy free from the manufacturer for review."
        )
        self.assertEqual("", filtered)

    def test_music_sentiment_and_toy_defects_are_kept(self):
        examples = (
            "What can I say. I love the 80's.",
            "This beautiful song is perfect for a wedding first dance.",
            "Super disappointed. The holes were poorly drilled and the plywood splintered.",
            "I thought my 5 year old would like these but she was a little young. "
            "I would say 7 and older for a good age.",
        )
        for text in examples:
            with self.subTest(text=text):
                filtered = preprocess(text)
                for segment in segment_review(text):
                    if segment.forced_relevance is True:
                        self.assertIn(segment.text.lstrip("but "), filtered)

    def test_evaluative_ingredient_comment_is_not_a_raw_list(self):
        text = "The fragrance irritated my skin, but salicylic acid helped my acne."
        self.assertEqual([], prohibited_content_labels(text))

    def test_generic_marketing_language_is_not_forced_keep(self):
        segments = segment_review("This product is available in many fun colors.")
        self.assertEqual(1, len(segments))
        self.assertIsNone(segments[0].forced_relevance)

    def test_hybrid_segmentation_splits_coordinated_experience_clause(self):
        segments = segment_review(
            "The box lists many colors, and I found the controls difficult."
        )
        self.assertEqual(
            ["The box lists many colors,", "and I found the controls difficult."],
            [segment.text for segment in segments],
        )

    def test_rating_adjustment_receives_filtered_review_text(self):
        classifier = RecordingClassifier()
        rows = [
            {
                "asin": "item-1",
                "overall": 5,
                "reviewText": (
                    "I usually write long reviews, but it broke after two uses."
                ),
            }
        ]
        output, stats = preprocess_batch(
            rows,
            classifier,
            {"item-1": 1.0},
            text_field="reviewText",
            rating_field="overall",
            item_field="asin",
            empty_review_policy="keep-original",
            adjust_ratings=True,
        )
        self.assertEqual("it broke after two uses.", output[0]["filteredReviewText"])
        self.assertEqual(
            [output[0]["filteredReviewText"]], classifier.polarity_inputs
        )
        self.assertEqual(3.0, output[0]["overall_new"])
        self.assertEqual(1, stats["ratings_adjusted"])

    def test_audit_contains_all_heavy_removals_and_fifty_random_rows(self):
        rows = [
            {
                "reviewerID": f"r{index}",
                "asin": "a",
                "overall": 5,
                "reviewText": "one two three four",
                "filteredReviewText": "one" if index == 0 else "one two three four",
            }
            for index in range(100)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            stats = write_preprocessing_audit(
                path,
                rows,
                source_field="reviewText",
                random_sample_size=50,
                seed=42,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                audit_rows = list(csv.DictReader(handle))
        self.assertEqual(1, stats["heavy_removal_reviews"])
        self.assertGreaterEqual(len(audit_rows), 50)
        heavy = [row for row in audit_rows if row["source_row_index"] == "0"]
        self.assertEqual(1, len(heavy))
        self.assertIn("removed_at_least_25_percent", heavy[0]["audit_reason"])


if __name__ == "__main__":
    unittest.main()
