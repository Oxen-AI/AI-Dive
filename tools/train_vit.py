from ai.dive.saver import Saver
from ai.dive.models.vit import ViT
from ai.dive.models.resnet50 import ResNet50
from ai.dive.data.file_classification import FileClassification
from ai.dive.data.label_reader import LabelReader
from ai.dive.data.directory_classification import DirectoryClassification

from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import AutoImageProcessor, ResNetForImageClassification

from transformers import TrainingArguments
from transformers import Trainer

from datasets import Dataset
from datasets import load_metric

from PIL import Image
from tqdm import tqdm

import torch
import numpy as np
import time

import argparse

def load_dataset(dataset, label_reader, processor, num_samples=-1):
    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = processor([x for x in example_batch['images']], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['labels']
        return inputs
    
    data_dict = {
        "images": [],
        "labels": []
    }
    for data in tqdm(dataset):
        # Format is {"filepath": "images/shiba_inu_1.jpg", "class_name": "shiba_inu"}
        # read in pixel values from filepath
        filepath = data["filepath"]
        image = Image.open(filepath)
        # convert grayscale image to rgb
        image = image.convert("RGB")
        # convert image to pixel values
        # pixel_values = processor(images=image, return_tensors="pt")
        data_dict["images"].append(image)
        
        # read in label index from class_name
        label = data["class_name"]
        label_idx = label_reader.index_of(label)
        data_dict["labels"].append(label_idx)
        if num_samples > 0 and len(data_dict["labels"]) >= num_samples:
            break

    # Create dataset object
    dataset = Dataset.from_dict(data_dict)
    print("Dataset created from dict!")

    dataset = dataset.with_transform(transform)
    print("Dataset prepared!")
    return dataset

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train a ViT on a dataset')
    parser.add_argument('-t', '--train_dataset', required=True, type=str, help='dataset to train model on')
    parser.add_argument('-e', '--eval_dataset', required=True, type=str, help='dataset to eval model on')
    parser.add_argument('-l', '--labels', required=True, type=str, help='The dynamic labels file to use')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-m', '--base_model', default="google/vit-base-patch16-224-in21k", type=str, help='The base model to use')
    parser.add_argument('-n', '--num-samples', default=-1, type=int, help='Number of samples to train model on')
    args = parser.parse_args()

    start_time = time.time()

    label_reader = LabelReader(labels_file=args.labels)
    labels = label_reader.labels()

    model_name_or_path = args.base_model
    # processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)

    print("Preparing train dataset...")
    train_dataset = DirectoryClassification(data_dir=args.train_dataset)
    train_dataset.build()
    train_dataset = load_dataset(train_dataset, label_reader, processor, num_samples=args.num_samples)

    print("Preparing eval dataset...")
    eval_dataset = DirectoryClassification(data_dir=args.eval_dataset)
    eval_dataset.build()
    eval_dataset = load_dataset(eval_dataset, label_reader, processor, num_samples=100)

    metric = load_metric("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


    # model = ViTForImageClassification.from_pretrained(
    #     model_name_or_path,
    #     num_labels=len(labels),
    #     id2label={str(i): c for i, c in enumerate(labels)},
    #     label2id={c: str(i) for i, c in enumerate(labels)}
    # )
    
    print(f"Loading model... {model_name_or_path}")
    model = ResNetForImageClassification.from_pretrained(model_name_or_path)
    
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        # fp16=True,
        fp16=False,
        no_cuda=True,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )
    
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