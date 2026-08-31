from typing import Optional
from datasets import load_dataset, Dataset
from hugging_face_authentication import hugging_face_auth
import pandas as pd


class DatasetLoader:
    """
    Handles loading, exploration, and column selection for Hugging Face datasets.
    """

    DEFAULT_DATASET_ID: str = "mteb/tweet_sentiment_extraction"

    def __init__(
        self,
        dataset_id: str = DEFAULT_DATASET_ID,
        split: str = "train",
        trust_remote_code: bool = True,
    ) -> None:
        hugging_face_auth()
        self.dataset_id: str = dataset_id
        self.split: str = split
        self.current_column: Optional[str] = None

        try:
            self.dataset: Dataset = load_dataset(
                dataset_id, split=split, trust_remote_code=trust_remote_code
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_id}' (split: '{split}'): {exc}"
            ) from exc

    def to_pandas(self) -> pd.DataFrame:
        return self.dataset.to_pandas()

    def preview(self, start_idx: int = 0, end_idx: int = 5) -> pd.DataFrame:
        total_len = len(self.dataset)
        start = max(0, min(start_idx, total_len))
        end = max(start, min(end_idx, total_len))

        sample_slice = self.dataset[start:end]
        df_preview = pd.DataFrame(sample_slice)
        print(f"\n--- Preview [{start}:{end}] of '{self.dataset_id}' ---")
        print(df_preview)
        return df_preview

    def get_info(self) -> str:
        columns = self.dataset.column_names
        num_rows = len(self.dataset)
        information = (
            f"Dataset: '{self.dataset_id}' (Split: {self.split})\n"
            f"Rows: {num_rows:,}\n"
            f"Columns ({len(columns)}): {columns}"
        )
        print(information)
        return information

    def select_column(self, default_col: Optional[str] = None) -> str:
        columns = self.dataset.column_names

        if default_col and default_col in columns:
            self.current_column = default_col
            return default_col

        fallback = "text" if "text" in columns else (columns[0] if columns else "unknown")

        prompt_msg = (
            f"\nAvailable columns: {columns}\n"
            f"Please enter the target column name (Press Enter for default '{fallback}'): "
        )

        try:
            user_choice = input(prompt_msg).strip()
        except (EOFError, KeyboardInterrupt):
            user_choice = ""

        if user_choice and user_choice in columns:
            self.current_column = user_choice
            return user_choice

        self.current_column = fallback
        return fallback

    checking_dataset_as_pandas = to_pandas
    printing_dataset = preview
    dataset_more_information = get_info
    ask_user_to_choose_a_column_to_work_on_the_dataset = select_column

    def __repr__(self) -> str:
        return (
            f"DatasetLoader("
            f"dataset_id='{self.dataset_id}', "
            f"split='{self.split}', "
            f"rows={len(self.dataset):,}, "
            f"columns={self.dataset.column_names})"
        )


if __name__ == "__main__":
    loader = DatasetLoader()
    print(loader)
