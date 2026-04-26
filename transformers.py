# from data import *
# from train_utils import *
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification

# multiclass_dataset = "jason23322/high-accuracy-email-classifier"
# train_split, test_split = load_data_splits(multiclass_dataset)

# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# label_to_name = {}

# for category_id, category in zip(train_split["category_id"], train_split["category"]):
#     label_to_name[int(category_id)] = category

# label_to_name = dict(sorted(label_to_name.items()))
# id2label = {i: name for i, name in label_to_name.items()}
# label2id = {name: i for i, name in id2label.items()}
# num_labels = len(id2label)

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=num_labels,
#     id2label=id2label,
#     label2id=label2id,
# )