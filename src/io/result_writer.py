"""
Append per-pair experiment rows to CSV and accumulate confusion-matrix counts.
"""

from __future__ import annotations

import csv
import os


class ResultWriter:
    """
    Write detection results for one pipeline/model/dataset run and track metrics.

    CSV schema matches downstream evaluation expectations.
    """

    def __init__(self, csv_path: str) -> None:
        """
        Args:
            csv_path: Absolute path to the CSV file to create or append.
        """
        self.csv_path = csv_path

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._file_exists = os.path.isfile(csv_path)
        self._fieldnames = [
            "pair_id",
            "ll1",
            "ll2",
            "ground_truth",
            "predicted_label",
        ]


    def record_result(
        self,
        pair_id: str,
        ll1: str,
        ll2: str,
        ground_truth: int,
        predicted_label: str,
    ) -> None:
        """
        Append one row in the output csv file.

        Args:
            pair_id: Stable identifier for the pair.
            ll1: Language of codeA for this pair (always "Java" in this project).
            ll2: Language of codeB for this pair (e.g. "Python", "C++", "C#").
            ground_truth: 1 clone, 0 non-clone.
            predicted_label: CLONE, NOT_CLONE, or ERROR.
        """
        row = {
            "pair_id": pair_id,
            "ll1": ll1,
            "ll2": ll2,
            "ground_truth": ground_truth,
            "predicted_label": predicted_label,
        }

        write_header = not self._file_exists
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)

            if write_header:
                writer.writeheader()
                self._file_exists = True

            writer.writerow(row)
