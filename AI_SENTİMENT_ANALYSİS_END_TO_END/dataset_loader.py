from datasets import load_dataset
from hugging_face_authentication import hugging_face_auth
import pandas as pd


class DatasetLoader:
    """
    Handles loading and exploration of HuggingFace datasets.
    Provides utilities for inspecting, previewing, and selecting
    columns for downstream pipeline processing across platforms.
    """

    DEFAULT_DATASET_ID = "mteb/tweet_sentiment_extraction"

    def __init__(self, dataset_id: str = DEFAULT_DATASET_ID, split: str = "train"):
        hugging_face_auth()
        self.dataset_id = dataset_id
        self.split = split
        
        # Load dataset with trust_remote_code enabled for flexible dataset support
        self.dataset = load_dataset(dataset_id, split=split, trust_remote_code=True)
        self.current_column = None

    def checking_dataset_as_pandas(self) -> pd.DataFrame:
        """
        Converts the entire Hugging Face dataset to a Pandas DataFrame.
        """
        return self.dataset.to_pandas()

    def printing_dataset(self, min_idx: int = 0, max_idx: int = 5) -> None:
        """
        Prints a lightweight slice of the dataset without duplicating entire data in RAM.
        """
        total_len = len(self.dataset)
        start = max(0, min(min_idx, total_len))
        end = max(start, min(max_idx, total_len))
        
        sample_slice = self.dataset[start:end]
        print(pd.DataFrame(sample_slice))

    def dataset_more_information(self) -> str:
        """
        Displays summary information about the dataset structure.
        """
        columns = self.dataset.column_names
        num_rows = len(self.dataset)
        information = (
            f"Dataset: '{self.dataset_id}' (Split: {self.split})\n"
            f"Rows: {num_rows:,}\n"
            f"Columns ({len(columns)}): {columns}"
        )
        print(information)
        return information

    def ask_user_to_choose_a_column_to_work_on_the_dataset(self, default_col: str = None) -> str:
        """
        Prompts user to select a column from the dataset with automatic fallback.
        """
        columns = self.dataset.column_names
        
        if default_col and default_col in columns:
            self.current_column = default_col
            return default_col

        prompt_msg = f"Available columns: {columns}\nPlease choose a column to work with: "
        user_choice = input(prompt_msg).strip()

        if user_choice in columns:
            self.current_column = user_choice
            return user_choice

        # Fallback priority: 'text' -> first column
        fallback = "text" if "text" in columns else columns[0]
        print(f"Invalid column '{user_choice}'. Falling back to default: '{fallback}'")
        self.current_column = fallback
        return fallback

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
    loader.printing_dataset(0, 5)
    selected_column = loader.ask_user_to_choose_a_column_to_work_on_the_dataset()
    print(f"Selected column: {selected_column}")