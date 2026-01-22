import difflib
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from unstructured_inference.models.eval import compare_contents_as_df


class TableAlignment:
    def __init__(self, cutoff: float = 0.8):
        self.cutoff = cutoff

    @staticmethod
    def get_content_in_tables(table_data: List[List[Dict[str, Any]]]) -> List[str]:
        # Replace below docstring with google-style docstring
        """Extracts and concatenates the content of cells from each table in a list of tables.

        Args:
          table_data: A list of tables, each table being a list of cell data dictionaries.

        Returns:
          List of strings where each string represents the concatenated content of one table.
        """
        return [" ".join([d["content"] for d in td if "content" in d]) for td in table_data]

    @staticmethod
    def get_table_level_alignment(
        predicted_table_data: List[List[Dict[str, Any]]],
        ground_truth_table_data: List[List[Dict[str, Any]]],
    ) -> List[int]:
        """Compares predicted table data with ground truth data to find the best
        matching table index for each predicted table.

        Args:
          predicted_table_data: A list of predicted tables.
          ground_truth_table_data: A list of ground truth tables.

        Returns:
          A list of indices indicating the best match in the ground truth for
          each predicted table.

        """
        ground_truth_texts = TableAlignment.get_content_in_tables(ground_truth_table_data)
        matched_indices = []
        for td in predicted_table_data:
            reference = TableAlignment.get_content_in_tables([td])[0]
            matches = difflib.get_close_matches(reference, ground_truth_texts, cutoff=0.1, n=1)
            matched_indices.append(ground_truth_texts.index(matches[0]) if matches else -1)
        return matched_indices

    @staticmethod
    def _zip_to_dataframe(table_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(table_data, columns=["row_index", "col_index", "content"])
        df = df.set_index("row_index")
        df["col_index"] = df["col_index"].astype(str)
        return df

    @staticmethod
    def get_element_level_alignment(
        predicted_table_data: List[List[Dict[str, Any]]],
        ground_truth_table_data: List[List[Dict[str, Any]]],
        matched_indices: List[int],
        cutoff: float = 0.8,
    ) -> Dict[str, float]:
        """Aligns elements of the predicted tables with the ground truth tables at the cell level.

        Args:
          predicted_table_data: A list of predicted tables.
          ground_truth_table_data: A list of ground truth tables.
          matched_indices: Indices of the best matching ground truth table for each predicted table.
          cutoff: The cutoff value for the close matches.

        Returns:
          A dictionary with column and row alignment accuracies.

        """
        content_diff_cols = []
        content_diff_rows = []
        col_index_acc = []
        row_index_acc = []

        for idx, td in zip(matched_indices, predicted_table_data):
            if idx == -1:
                content_diff_cols.append(0)
                content_diff_rows.append(0)
                col_index_acc.append(0)
                row_index_acc.append(0)
                continue
            ground_truth_td = ground_truth_table_data[idx]

            # Get row and col content accuracy
            predict_table_df = TableAlignment._zip_to_dataframe(td)
            ground_truth_table_df = TableAlignment._zip_to_dataframe(ground_truth_td)

            table_content_diff = compare_contents_as_df(
                ground_truth_table_df.fillna(""),
                predict_table_df.fillna(""),
            )
            content_diff_cols.append(table_content_diff["by_col_token_ratio"])
            content_diff_rows.append(table_content_diff["by_row_token_ratio"])

            aligned_element_col_count = 0
            aligned_element_row_count = 0
            total_element_count = 0
            # Get row and col index accuracy
            ground_truth_td_contents_list = [gtd["content"].lower() for gtd in ground_truth_td]
            content_to_indices: Dict[str, List[int]] = {}
            for i, s in enumerate(ground_truth_td_contents_list):
                content_to_indices.setdefault(s, []).append(i)

            used_indices = set()
            # Cache best matching ground-truth string per predicted content to avoid repeated difflib calls
            match_cache: Dict[str, str] = {}

            for td_ele in td:
                content = td_ele["content"].lower()
                row_index = td_ele["row_index"]
                col_idx = td_ele["col_index"]

                # Retrieve cached matched string if present
                if content in match_cache:
                    matched_string = match_cache[content]
                else:
                    matches = difflib.get_close_matches(
                        content,
                        ground_truth_td_contents_list,
                        cutoff=cutoff,
                        n=1,
                    )
                    matched_string = matches[0] if matches else ""
                    # store empty string to represent no match (consistent with original matched_idx -1)
                    match_cache[content] = matched_string

                if not matched_string:
                    matched_idx = -1
                else:
                    indices_for_string = content_to_indices.get(matched_string, [])
                    # Find first unused index in indices_for_string
                    found_idx = -1
                    for candidate in indices_for_string:
                        if candidate not in used_indices:
                            found_idx = candidate
                            break
                    if found_idx == -1 and indices_for_string:
                        # If all indices are used, reset used_indices and pick the first
                        # If all indices are used, reset used_indices and use the first index
                        used_indices.clear()
                        found_idx = indices_for_string[0]
                    matched_idx = found_idx if found_idx != -1 else -1

                if matched_idx >= 0:
                    gt_row_index = ground_truth_td[matched_idx]["row_index"]
                    gt_col_index = ground_truth_td[matched_idx]["col_index"]
                    # Immediately compare and count to avoid storing tuples and iterating later
                    if row_index == gt_row_index:
                        aligned_element_row_count += 1
                    if col_idx == gt_col_index:
                        aligned_element_col_count += 1
                    total_element_count += 1
                    used_indices.add(matched_idx)

            table_col_index_acc = 0
            table_row_index_acc = 0
            if total_element_count > 0:
                table_col_index_acc = round(aligned_element_col_count / total_element_count, 2)
                table_row_index_acc = round(aligned_element_row_count / total_element_count, 2)

            col_index_acc.append(table_col_index_acc)
            row_index_acc.append(table_row_index_acc)

        not_found_gt_table_indexes = [
            id for id in range(len(ground_truth_table_data)) if id not in matched_indices
        ]
        for _ in not_found_gt_table_indexes:
            content_diff_cols.append(0)
            content_diff_rows.append(0)
            col_index_acc.append(0)
            row_index_acc.append(0)

        return {
            "col_index_acc": round(np.mean(col_index_acc), 2),
            "row_index_acc": round(np.mean(row_index_acc), 2),
            "col_content_acc": round(np.mean(content_diff_cols) / 100.0, 2),
            "row_content_acc": round(np.mean(content_diff_rows) / 100.0, 2),
        }
