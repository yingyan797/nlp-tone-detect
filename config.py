# config constants

train_data_path = "dataset/dontpatronizeme_pcl.tsv"
test_data_path = "dataset/task4_test.tsv"

train_split_path = "dataset/practice_splits/train_semeval_parids-labels.csv"
dev_split_path = "dataset/practice_splits/dev_semeval_parids-labels.csv"

data_col_headers = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
data_col_dtypes = {
    "par_id": str,
    "art_id": str,
    "keyword": str,
    "country_code": str,
    "text": str,
    "label": int
}