from ai.dive.saver import Saver
from ai.dive.models.vit import ViT
from ai.dive.models.resnet50 import ResNet50
from ai.dive.data.label_reader import LabelReader
from ai.dive.data.image_file_classification import ImageFileClassificationDataset

from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import AutoImageProcessor, ResNetForImageClassification

from transformers import TrainingArguments
from transformers import Trainer

from datasets import load_metric

import torch
import numpy as np
import time

import argparse
import os

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train a ViT on a dataset')
    parser.add_argument('-d', '--data', required=True, type=str, help='datasets to train/eval model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-m', '--base_model', default="google/vit-base-patch16-224-in21k", type=str, help='The base model to use')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use the GPU')
    args = parser.parse_args()

    start_time = time.time()

    labels_file = os.path.join(args.data, "labels.txt")
    label_reader = LabelReader(labels_file)
    labels = label_reader.labels()

    model_name_or_path = args.base_model
    # processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)

    print("Preparing train dataset...")
    train_file = os.path.join(args.data, "train.csv")
    ds = ImageFileClassificationDataset(
        data_dir=args.data,
        file=train_file,
        label_reader=label_reader,
        img_processor=processor
    )
    train_dataset = ds.to_hf_dataset()

    print(train_dataset[0])
    print(train_dataset[0]['pixel_values'].shape)

    print("Preparing eval dataset...")
    train_file = os.path.join(args.data, "test.csv")
    ds = ImageFileClassificationDataset(
        data_dir=args.data,
        file=train_file,
        label_reader=label_reader,
        img_processor=processor,
        num_samples=100
    )
    eval_dataset = ds.to_hf_dataset()

    # model = ViTForImageClassification.from_pretrained(
    #     model_name_or_path,
    #     num_labels=len(labels),
    #     id2label={str(i): c for i, c in enumerate(labels)},
    #     label2id={c: str(i) for i, c in enumerate(labels)}
    # )
    
    print(f"Loading model... {model_name_or_path}")
    model = ResNetForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )
    
    training_args = TrainingArguments(
        output_dir=args.output, # directory to save the model
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4, # loop through the data N times
        no_cuda=(not args.gpu), # use the GPU or not
        save_steps=10, # save the model every N steps
        eval_steps=10, # evaluate the model every N steps
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2, # only keep the last N models
        remove_unused_columns=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )
    
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
    
    metric = load_metric("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
    )

    print("Training model...")
    train_results = trainer.train()
    print("Training complete!")
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")
    # write elapsed time to the model dir
    with open(f"{args.output}/elapsed_time.txt", "w") as f:
        f.write(str(elapsed_time))
        
    # NOTES:
    # GPU: 2.8 it/sec vs M1 CPU 4.5 sec/it
    # GPU Memory: 4GB (TODO: Do math on model size)
    # Total training time: ??



if __name__ == '__main__':
    main()