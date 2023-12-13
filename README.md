# 🤿 AI Dive

AI `D`ata `I`ngestor, `V`erifier and `E`ncoder.

This library is meant to be practical examples of how to run real world AI Models given data.

## Installation

```
pip install ai-dive
```

# Why build AI Dive?

In the age of the [AI Engineer](https://www.latent.space/p/ai-engineer), it is more likely that you will start by grabbing an off the shelf model as a starting point than training your own from scratch. That is not to say you will never train a model. It is just to say, let's verify state of the art before we go building.

> A wide range of AI tasks that used to take 5 years and a research team to accomplish in 2013, now just require API docs and a spare afternoon in 2023.

🤿 AI-Dive let's you easily dive into the results of a model to decide whether it is worth building upon. It also gives a simple and consistent interface to run in your app or implement new models.

# Main Components

There are a few main components that make up a DIVE.

1) `Model` - The machine learning model you want to analyze (for example an Image Classifier)
2) `Dataset` - The dataset you want to evaluate (for example Cats vs Dogs)
3) `Diver` - The code to run the model given the data
4) `Saver` - Saves the outputs in a consistent format to analyze

## Model

AI-Dive provides a wrapper around existing models to make them easy to run on your own data. We are not writing models from scratch, we are simply wrapping them with a consistent interface so they can be evaluated in a consistent way.

```python
from ai.dive.models.vit import ViT

model = ViT()
data = {"full_path": "images/shiba_inu_1.jpg"}
output = model.predict(data)
print(output)
```

There are a few models implemented already, we are looking to extend this list to new models as the come out, or allow this interface to be implemented in your package to save you time evaluating.

## Dataset

Models are worthless without the data to run and evaluate them on. Sure, you can poke your model with a stick by running on a single example, but the real insights come from running your model given a dataset.

There are a few datasets implemented already, including a `DirectoryClassification` dataset and an `ImageFileClassificationDataset` dataset.

The datasets follow some conventions laid out in the [🧼 SUDS blogpost](https://blog.oxen.ai/suds-a-guide-to-structuring-unstructured-data/). They should be re-usable interfaces so we can evaluate multiple models on the same dataset, and use multiple dataset for the same model.

```python
from ai.dive.models.vit import ViT
from ai.dive.data.directory_classification import DirectoryClassification

# Instantiate the model and dataset
model = ViT()
dataset = DirectoryClassification(data_dir="/path/to/images")
```

## Diver

The Diver's job is to run the model, given the dataset. It will return a list of json blobs as a results set that you can save to a csv file or any other format to analyze later.

```python
from ai.dive.models.vit import ViT
from ai.dive.data.directory_classification import DirectoryClassification

# Instantiate the model and dataset
model = ViT()
dataset = DirectoryClassification(data_dir="/path/to/images")

# Run the model on the dataset
diver = Diver(model, dataset)
results = diver.run()

# The output will be a list of all the predictions
print(results)
```

## Saver

Since these models tend to take some time to run on a dataset, there is a `Saver` class that helps save data as we go. You can specify how often you want to save the data.

By default the Saver will save csvs in a 🧼 SUDS compatible format to make it easy to debug later.

```python
from ai.dive.models.vit import ViT
from ai.dive.data.directory_classification import DirectoryClassification

# Instantiate the model and dataset
model = ViT()
dataset = DirectoryClassification(data_dir="/path/to/images")

# Use a Saver to write the results to a csv
saver = Saver(
    "output.csv",
    output_keys=['filename', 'class_name', 'prediction', 'probability'],
    save_every=10
)

# Run the model on the dataset, and save the results as we go
diver = Diver(model, dataset, saver=saver)
results = diver.run()

# The output will be a list of all the predictions
print(results)
```

All together, the `Diver` object saves you the work of processing each row in the dataframe and the `Saver` takes care of writing all the results to disk so you can compare them across runs.

With plug and play models and datasets, the hope is anyone can evaluate a model against any dataset and share the results quickly and effectively.

## Model Interface

Models should be as easy to run as:

```python
prediction = model.predict(data)
```

There are a few models implemented already, we are looking to extend this list to new models as the come out, or allow this interface to be implemented in your package to save you time evaluating.

The interface is very simple.

```python
from ai.dive.models.model import Model

class ImageClassification(Model):
    def __init__(self):
        super().__init__()

    # Function to run the model on a single example
    def _predict(self, data):
        # Do your work, return dictionary of results
        return {
            "prediction": "dog",
            "probability": 0.9
        }
```

We would love help building out interfaces to these models! Or suggestions for ones that you find interesting.

Each week during our [Practical ML Dives](https://lu.ma/practicalml) series we will be knocking out a couple models live for people to re-use.

* [x] ResNet50
* [x] Vision Transformer (ViT)
* [x] CLIP (Zero-Shot Image Classification)
* [ ] Llama-2
* [ ] Mistral-7b
* [ ] Dalle-3
* [ ] Stable Diffusion
* [ ] Magic Animate

## Dataset Interface

A dataset has to implement two methods `__len__` and `__getitem__` so that we can iterate over it. If it implements `_build`, you can load everything into memory to make the other calls faster.

Here is an example dataset that iterates over a directory of images with the folder names as classnames.

Example directory structure:

```
images/
    cat/
        1.jpg
        2.jpg
    dog/
        1.jpg
        2.jpg
        3.jpg
```

Example data loader:

```python
from ai.dive.data.dataset import Dataset
import os

class DirImageClassification(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = data_dir

    # For iterating over the dataset
    def __len__(self):
        return len(self.filepaths)

    # For iterating over the dataset
    def __getitem__(self, idx):
        return {
            "filepath": self.filepaths[idx],
            "class_name": self.labels[idx]
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        # iterate over files in directory, taking the directory name as the label
        labels = []
        filepaths = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    labels.append(os.path.basename(root))
                    filepaths.append(os.path.join(root, file))
        self.labels = labels
        self.filepaths = filepaths
```

# 🐂 Join the Oxen.ai Community

[Oxen.ai](https://oxen.ai) is a community of builders, who have respect for real world models trained and evaluated on real world data.

Feel free to join the [Discord](https://discord.com/invite/s3tBEn7Ptg) or join any of our [Practical ML Dives](https://lu.ma/practicalml)

Best & Moo
~ The Oxen.ai Herd