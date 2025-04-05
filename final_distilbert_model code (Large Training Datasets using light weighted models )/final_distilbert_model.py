"""**TASK 1: Zero-Shot**
(Perform initial tests with untrained model)
"""

# test set: Cohere_55K

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# 1. Using HuggingFace pre-training model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. load dataset
df = pd.read_csv("final_filtered_balanced_55000.csv")
assert "text" in df.columns and "label" in df.columns

# 3. transfer to HuggingFace Dataset
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 4. Output evaluation indicators
print("\n📊 Zero-shot Evaluation results (untrained model)：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("initial_test_predictions.csv", index=False)
print("✅ result has saved as initial_test_predictions.csv")

# test set: merged_dataset (ZC real dataset)

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# 1. Using HuggingFace pre-training model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. load dataset
df = pd.read_csv("merged_dataset.csv")
assert "text" in df.columns and "label" in df.columns

# 3. transfer to HuggingFace Dataset
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 4. Output evaluation indicators
print("\n📊 Zero-shot Evaluation results on merged_dataset.csv (untrained model)：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("initial_test_predictions on merged_dataset.csv", index=False)
print("✅ result has saved as initial_test_predictions on merged_dataset.csv")

# test set: real_life_data2 (XLF real dataset)

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# 1. Using HuggingFace pre-training model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. load dataset
df = pd.read_csv("real_life_data2.csv")
assert "text" in df.columns and "label" in df.columns

# 3. transfer to HuggingFace Dataset
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 4. Output evaluation indicators
print("\n📊 Zero-shot Evaluation results on real_life_data2.csv (untrained model)：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("initial_test_predictions on real_life_data2.csv", index=False)
print("✅ result has saved as initial_test_predictions on real_life_data2.csv")

# test set: filtered_logicality_dataset_1 (Kya real dataset)

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# 1. Using HuggingFace pre-training model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. load dataset
df = pd.read_csv("filtered_logicality_dataset_1.csv")
assert "text" in df.columns and "label" in df.columns

# 3. transfer to HuggingFace Dataset
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 4. Output evaluation indicators
print("\n📊 Zero-shot Evaluation results on filtered_logicality_dataset_1.csv (untrained model)：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("initial_test_predictions on filtered_logicality_dataset_1.csv", index=False)
print("✅ result has saved as initial_test_predictions on filtered_logicality_dataset_1.csv")

# test set: GPT_general_dataset_16.5k

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# 1. Using HuggingFace pre-training model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. load dataset
df = pd.read_csv("general_16.5k.csv")
assert "text" in df.columns and "label" in df.columns

# 3. transfer to HuggingFace Dataset
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 4. Output evaluation indicators
print("\n📊 Zero-shot Evaluation results on general_16.5k.csv(untrained model)：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("initial_test_predictions on general_16.5k.csv", index=False)
print("✅ result has saved as initial_test_predictions on general_16.5k.csv")

# test set: GPT_adversarial_dataset_4950

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# 1. Using HuggingFace pre-training model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. load dataset
df = pd.read_csv("logic_adversarial_4950_labeled.csv")
assert "text" in df.columns and "label" in df.columns

# 3. transfer to HuggingFace Dataset
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 4. Output evaluation indicators
print("\n📊 Zero-shot Evaluation results on logic_adversarial_4950_labeled.csv(untrained model)：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("initial_test_predictions on logic_adversarial_4950_labeled.csv", index=False)
print("✅ result has saved as initial_test_predictions on logic_adversarial_4950_labeled.csv")

"""**TASK 2:**
Train: Cohere_Train_55K only to TEST: other datasets
"""

# test set: Cohere_test_55K

import os
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

os.environ["WANDB_DISABLED"] = "true"

# 1. load training set and test set
train_df = pd.read_csv("Cohere_train_55K.csv")
test_df = pd.read_csv("Cohere_test_55K.csv")

train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

# 2. validation set
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

# 3. transfer HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# 4. Segmentation of three data sets
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

for d in [train_dataset, val_dataset, test_dataset]:
    d.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. 加load distilbert model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

# 7. parameters setting
training_args = TrainingArguments(
    output_dir="./distilbert_cohere_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 8. training
trainer.train()

# 9. Evaluate on the test set
print("\n📊 Test set evaluation results：")
test_result = trainer.predict(test_dataset)
for k, v in test_result.metrics.items():
    print(f"{k}: {v:.4f}")

# 10. save final model
trainer.save_model("distilbert_cohere_model")

tokenizer.save_pretrained("distilbert_cohere_model")

# test set: GPT_general_16.5k

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# 1. load trained model
model_path = "./distilbert_cohere_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. load new test set
df = pd.read_csv("general_16.5k.csv")
assert "label" in df.columns,  "dataset must contains 'label' column"

# 3. transfer Dataset format
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 7. print metrics
print("\n📊 Evaluation results of the model on general_16.5k:")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# 8. save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df.to_csv("general_16.5k_with_predictions.csv", index=False)
print("✅ Prediction results saved as general_16.5k_with_predictions.csv")

# test set: merged_dataset (ZC real dataset)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# 1. load trained model
model_path = "./distilbert_cohere_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. load new test set
df = pd.read_csv("merged_dataset.csv")
assert "label" in df.columns,  "dataset must contains 'label' column"

# 3. transfer Dataset format
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 7. print metrics
print("\n📊 Evaluation results of the model on merged_dataset.csv:")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# 8. save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df.to_csv("gmerged_dataset_with_predictions.csv", index=False)
print("✅ Prediction results saved as merged_dataset_with_predictions.csv")

# test set: real_life_data2 (XLF real dataset)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# 1. load trained model
model_path = "./distilbert_cohere_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. load new test set
df = pd.read_csv("real_life_data2.csv")
assert "label" in df.columns,  "dataset must contains 'label' column"

# 3. transfer Dataset format
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 7. print metrics
print("\n📊 Evaluation results of the model on real_life_data2.csv:")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# 8. save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df.to_csv("real_life_data2_with_predictions.csv", index=False)
print("✅ Prediction results saved as real_life_data2_with_predictions.csv")

# test set: filtered_logicality_dataset_1 (Kya real dataset)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# 1. load trained model
model_path = "./distilbert_cohere_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. load new test set
df = pd.read_csv("filtered_logicality_dataset_1.csv")
assert "label" in df.columns,  "dataset must contains 'label' column"

# 3. transfer Dataset format
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 7. print metrics
print("\n📊 Evaluation results of the model on filtered_logicality_dataset_1.csv:")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# 8. save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df.to_csv("filtered_logicality_dataset_1_with_predictions.csv", index=False)
print("✅ Prediction results saved as filtered_logicality_dataset_1_with_predictions.csv")

"""**TASK 3:**
 Train: cohere + ChatGPT general dataset to TEST: other datasets
"""

# test set: GPT_general_test_16.5K

import os
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

os.environ["WANDB_DISABLED"] = "true"

# 1. load training set and test set
train_df = pd.read_csv("Cohere+GPT_train.csv")
test_df = pd.read_csv("general_test_16.5K.csv")

train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

# 2. validation set
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

# 3. transfer Dataset format
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

for d in [train_dataset, val_dataset, test_dataset]:
    d.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 4. load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

# 6. set parameters
training_args = TrainingArguments(
    output_dir="./distilbert_cohere_GPT_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 8. training
trainer.train()

# 9. Evaluate on the test set
print("\n📊 Test set evaluation results：")
test_result = trainer.predict(test_dataset)
for k, v in test_result.metrics.items():
    print(f"{k}: {v:.4f}")

# 14. save final model
trainer.save_model("distilbert_cohere_GPT_model")
tokenizer.save_pretrained("distilbert_cohere_GPT_model")

# test set: GPT_adversarial dataset_4950

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

model_path = "./distilbert_cohere_GPT_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load test data
df = pd.read_csv("logic_adversarial_4950_labeled.csv")

assert "label" in df.columns, "dataset must contains 'label' column"

dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# print metrics
print("\n📊 Evaluation results of the model on logic_adversarial_4950_labeled.csv：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("logic_adversarial_with_predictions.csv", index=False)
print("✅ Prediction results saved as logic_adversarial_with_predictions.csv")

# test set: Cohere_test_55K

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

model_path = "./distilbert_cohere_GPT_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load test data
df = pd.read_csv("Cohere_test_55K.csv")

assert "label" in df.columns, "dataset must contains 'label' column"

dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# print metrics
print("\n📊 Evaluation results of the model on Cohere_test_55K.csv：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("Cohere_test_55K_with_predictions.csv", index=False)
print("✅ Prediction results saved as Cohere_test_55K_with_predictions.csv")

# test set: merged_dataset (ZC real dataset)

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

model_path = "./distilbert_cohere_GPT_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load test data
df = pd.read_csv("merged_dataset.csv")

assert "label" in df.columns, "dataset must contains 'label' column"

dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# print metrics
print("\n📊 Evaluation results of the model on merged_dataset.csv：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("merged_dataset_with_predictions.csv", index=False)
print("✅ Prediction results saved as merged_dataset_with_predictions.csv")

# test set: real_life_data2 (XLF real dataset)

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

model_path = "./distilbert_cohere_GPT_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load test data
df = pd.read_csv("real_life_data2.csv")

assert "label" in df.columns, "dataset must contains 'label' column"

dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# print metrics
print("\n📊 Evaluation results of the model on real_life_data2.csv：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("real_life_data2_with_predictions.csv", index=False)
print("✅ Prediction results saved as Creal_life_data2_with_predictions.csv")

# test set: filtered_logicality_dataset_1 (Kya real dataset)

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

model_path = "./distilbert_cohere_GPT_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load test data
df = pd.read_csv("filtered_logicality_dataset_1.csv")

assert "label" in df.columns, "dataset must contains 'label' column"

dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# print metrics
print("\n📊 Evaluation results of the model on filtered_logicality_dataset_1.csv：")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df["prob_0"] = result.predictions[:, 0]
df["prob_1"] = result.predictions[:, 1]
df.to_csv("filtered_logicality_dataset_1_with_predictions.csv", index=False)
print("✅ Prediction results saved as filtered_logicality_dataset_1_with_predictions.csv")

"""**TASK4:**
Train: ChatGPT dataset related to real data to TEST: real dataset
"""

# test set: merged_dataset (ZC real dataset)

import os
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

os.environ["WANDB_DISABLED"] = "true"

# 1. load training set and test set
train_df = pd.read_csv("synthetic_dataset_200.csv")
test_df = pd.read_csv("merged_dataset.csv")

train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

# 2. validation set
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

# 3. transfer HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# 4. Segmentation of three data sets
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

for d in [train_dataset, val_dataset, test_dataset]:
    d.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. 加load distilbert model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

# 7. parameters setting
training_args = TrainingArguments(
    output_dir="./distilbert_cohere_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 8. training
trainer.train()

# 9. Evaluate on the test set
print("\n📊 Test set evaluation results：")
test_result = trainer.predict(test_dataset)
for k, v in test_result.metrics.items():
    print(f"{k}: {v:.4f}")

# 10. save final model
trainer.save_model("distilbert_GPTreal_model")

tokenizer.save_pretrained("distilbert_GPTreal_model")

# test set: filtered_logicality_dataset_1 (Kya real dataset)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# 1. load trained model
model_path = "./distilbert_GPTreal_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. load new test set
df = pd.read_csv("filtered_logicality_dataset_1.csv")
assert "label" in df.columns,  "dataset must contains 'label' column"

# 3. transfer Dataset format
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 7. print metrics
print("\n📊 Evaluation results of the model on general_16.5k:")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# 8. save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df.to_csv("general_16.5k_with_predictions.csv", index=False)
print("✅ Prediction results saved as general_16.5k_with_predictions.csv")

# test set: real_life_data2 (XLF real dataset)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# 1. load trained model
model_path = "./distilbert_GPTreal_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. load new test set
df = pd.read_csv("real_life_data2.csv")
assert "label" in df.columns,  "dataset must contains 'label' column"

# 3. transfer Dataset format
dataset = Dataset.from_pandas(df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": auc
    }

trainer = Trainer(model=model, compute_metrics=compute_metrics)
result = trainer.predict(dataset)

# 7. print metrics
print("\n📊 Evaluation results of the model on real_life_data2:")
for k, v in result.metrics.items():
    print(f"{k}: {v:.4f}")

# 8. save result
df["predicted_label"] = np.argmax(result.predictions, axis=1)
df.to_csv("real_life_data2_with_predictions.csv", index=False)
print("✅ Prediction results saved as real_life_data2_with_predictions.csv")