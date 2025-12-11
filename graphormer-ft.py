from datasets import load_dataset
import numpy as np
from transformers.models.graphormer.collating_graphormer import (
    preprocess_item,
    GraphormerDataCollator,
)
from transformers import GraphormerForGraphClassification
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from typing import Dict
import evaluate
from transformers import EvalPrediction

# Add numpy globals to safe globals list for checkpoint loading
# This includes numpy array types and all numpy dtype classes
torch.serialization.add_safe_globals(
    [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.core.multiarray.scalar,
        # Add all numpy dtype classes
        np.dtypes.UInt32DType,
        np.dtypes.Int64DType,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.Int32DType,
        np.dtypes.BoolDType,
        np.dtypes.ObjectDType,
    ]
)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    logits, labels = eval_pred.predictions[0], eval_pred.label_ids
    # print(f"logits shape: {logits[0].shape}")
    # print(f"whatever length: {len(logits[1])}")
    # for i in range(len(logits[1])):
    #     print(f"whatever shape: {logits[1][i].shape}")
    # print(f"labels length: {len(labels)}")
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=predictions, references=labels)
    print(f"predictions: {predictions}")
    print(f"labels: {labels}")
    print(f"Computed metrics: {result}")

    return result


if __name__ == "__main__":
    # There is only one split on the hub
    dataset = load_dataset("OGB/ogbg-molhiv", cache_dir="./data")
    seed = 42
    dataset = dataset.shuffle(seed=seed)

    dataset_processed = dataset.map(preprocess_item, batched=False)

    model = GraphormerForGraphClassification.from_pretrained(
        "clefourrier/pcqm4mv2_graphormer_base",
        num_classes=2,  # num_classes for the downstream task
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        "graph-classification/checkpoint/molhiv",
        logging_dir="graph-classification",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,  # batch size can be changed automatically to prevent OOMs
        gradient_accumulation_steps=4,  # simulating batch size of 64
        dataloader_num_workers=4,  # 1,
        num_train_epochs=4,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",  # Save model checkpoint every epoch
        save_total_limit=3,  # Keep only the last 3 checkpoints to save disk space
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="accuracy",  # Use accuracy to determine the best model
        push_to_hub=False,
        dataloader_drop_last=True,
        seed=seed,
        label_names=["labels"],
        use_mps_device=True,  # Force MPS usage for both training and evaluation
    )

    # check first the whole pipeline works on a smaller subset of the data
    train = dataset_processed["train"]  # .shuffle(seed=10).select(range(100))
    eval = dataset_processed["validation"]  # .shuffle(seed=10).select(range(100))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        data_collator=GraphormerDataCollator(),
        compute_metrics=compute_metrics,
    )
    train_results = trainer.train(resume_from_checkpoint=True)
