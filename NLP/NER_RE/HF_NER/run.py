from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import DataLoader
from datasets import Dataset
import evaluate
import numpy as np

# model_checkpoint = "bert-base-cased"
model_checkpoint = "./bert-finetuned-ner/checkpoint-3705"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = evaluate.load("seqeval")

label2id = {
    '[PAD]': 0,
    'O': 1,
    'B_MultiCancer': 2,
    'I_MultiCancer': 3,
    'B_Gene': 4,
    'I_Gene': 5,
    'B_Cancer': 6,
    'I_Cancer': 7,
    'B_MultiGene': 8,
    'I_MultiGene': 9,
    'B_signal-pathway': 10,
    'I_signal-pathway': 11,
    'B_GeneFunction': 12,
    'I_GeneFunction': 13,
    'B_Gene-multiFunction': 14,
    'I_Gene-multiFunction': 15
}
id2label = {v:k for k,v in label2id.items()}
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
)

import sys
sys.path.append('../../Lib/LabelStudio')

from TrainDataGenerator import TrainDataGenerator


def align_labels_with_tokens(labels, word_ids):
    """
    扩充标签到新tokenized词，某些单词分成两个
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 0:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(data):
    tokenized_inputs = tokenizer(
        data["words"], truncation=True, is_split_into_words=True
    )
    new_labels = []
    for i, labels in enumerate(data["labels"]):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    # return {
    #     "precision": all_metrics["overall_precision"],
    #     "recall": all_metrics["overall_recall"],
    #     "f1": all_metrics["overall_f1"],
    #     "accuracy": all_metrics["overall_accuracy"],
    # }
    return all_metrics


if __name__ == "__main__":
    train_file_path = r'../../Gene/LabelData/Data/project-gene-200.json'
    test_file_path = r'../../Gene/LabelData/Data/project-gene-200_250.json'

    tg = TrainDataGenerator()
    train_data = tg.process_gene_ner(train_file_path, 'max')
    train_data_set = Dataset.from_dict({
        'words': [words for words, labels in train_data],
        'labels': [[label2id[label] for label in labels] for words, labels in train_data]
    })
    test_data = tg.process_gene_ner(test_file_path, 'none')
    test_data_set = Dataset.from_dict({
        'words': [words for words, labels in test_data],
        'labels': [[label2id[label] for label in labels] for words, labels in test_data]
    })

    tokenized_train_datasets = train_data_set.map(
        tokenize_and_align_labels,
        batched=True
    )
    tokenized_test_datasets = test_data_set.map(
        tokenize_and_align_labels,
        batched=True
    )


    # words, labels = train_data[0][0], [label2id[e] for e in train_data[0][1]]
    # inputs = tokenizer(words, is_split_into_words=True)
    # word_ids = inputs.word_ids()
    # print(words, labels)
    # print(inputs.tokens(), word_ids)
    # print(align_labels_with_tokens(labels, word_ids))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_test_datasets,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()


    #
