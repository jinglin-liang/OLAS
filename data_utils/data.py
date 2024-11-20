from datasets import load_dataset


DATASET_NAME_TO_PATH = {
    "imdb": "datasets/imdb"
}


def imdb_standardize_function(data_point):
    return data_point


def load_raw_data(data_name: str):
    '''
    return (train_data, test_data), standardize_function
    train_data and test_data are datasets.Dataset
    standardize_function is a function that takes a data point and returns a standardized data point {"text": str, "label": int}
    '''
    if data_name.lower() == "imdb":
        data = load_dataset(DATASET_NAME_TO_PATH[data_name])
        train_data = data["train"]
        test_data = data["test"]
        unsupervised_data = data["unsupervised"]
        return (train_data, test_data), imdb_standardize_function
    else:
        raise NotImplementedError(f"Dataset {data_name} is not supported.")