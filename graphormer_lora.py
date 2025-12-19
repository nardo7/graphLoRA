from typing import Dict

import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    EvalPrediction,
    GraphormerForGraphClassification,
    TrainingArguments,
)
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator

from trainer import Trainer

if __name__ == "__main__":
    # There is only one split on the hub
    dataset = load_dataset("OGB/ogbg-molhiv", cache_dir="./data")
    seed = 42
    dataset = dataset.shuffle(seed=seed)

    model = GraphormerForGraphClassification.from_pretrained(
        "clefourrier/pcqm4mv2_graphormer_base",
        num_classes=2,  # num_classes for the downstream task
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(r=4, target_modules=["q_proj", "k_proj"])
    model.add_adapter(adapter_config=lora_config)

    training_args = TrainingArguments(
        "graph-classification/lora/molhiv",
        logging_dir="graph-classification",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,  # batch size can be changed automatically to prevent OOMs
        gradient_accumulation_steps=4,  # simulating batch size of 64
        dataloader_num_workers=8,  # 1,
        num_train_epochs=4,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",  # Save model checkpoint every epoch
        save_total_limit=3,  # Keep only the last 3 checkpoints to save disk space
        load_best_model_at_end=False,  # Disabled for LoRA compatibility
        metric_for_best_model="roc_auc",  # Use accuracy to determine the best model
        push_to_hub=False,
        use_mps_device=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,  # CRITICAL: Don't remove columns - GraphormerDataCollator needs them!
        seed=seed,
    )

    train = dataset["train"].with_format("numpy")
    eval = dataset["validation"].with_format("numpy")

    metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict:
        logits, labels = eval_pred.predictions, eval_pred.label_ids.reshape(-1)
        predictions = np.argmax(logits, axis=-1)
        predictions = np.reshape(predictions, -1)
        result = metric.compute(prediction_scores=predictions, references=labels)
        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        data_collator=GraphormerDataCollator(on_the_fly_processing=True),
        compute_metrics=compute_metrics,
    )
    train_results = trainer.train(resume_from_checkpoint=True)
