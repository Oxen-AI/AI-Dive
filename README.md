# 🤿 AI Dive

AI `D`ata `I`ngestor, `V`erifier and `E`ncoder.

This library is meant to be practical examples of how to run real world AI Models given data.

## Installation

```
pip install ai-dive
```

## Why build AI-Dive

In the age of the [AI Engineer](https://www.latent.space/p/ai-engineer), it is more likely that you will start by grabbing an off the shelf model as a starting point than training your own from scratch. That is not to say you will never train a model. It is just to say, let's verify state of the art before we go building.

> A wide range of AI tasks that used to take 5 years and a research team to accomplish in 2013, now just require API docs and a spare afternoon in 2023.

🤿 AI-Dive let's you easily dive into the results of a model to decide whether it is worth building upon. It also gives a simple and consistent interface to run in your app or implement new models.

# Model

TODO: Breakup below into each part

1) Model
2) Dataset
3) Diver
4) Saver

# Dataset

TODO

# Dive

TODO

# Save

TODO

# All Together Now

TODO

# Model & Dataset

There are only a two interfaces to implement to get up and running on any model or dataset.

1) `Dataset` - How to iterate over data
2) `Model` - How to predict given each data point

# Dive & Save

There are two helper classes to run your model given a dataset

1) `Diver` - How to run each datapoint from your dataset through the model.
2) `Saver` - How to save off the results of the run. Running the model and not saving the results can cost time and money.

## Models

AI-Dive provides a wrapper around existing models to make them easy to run on your own data. We are not writing models from scratch, we are simply wrapping them with a consistent interface so they can be evaluated in a consistent way.

```python
from ai.dive.models.vit import ViT

model = ViT()
data = {"filepath": "images/shiba_inu_1.jpg"}
output = model.predict(data)
print(output)
```

There are a few models implemented already, we are looking to extend this list to new models as the come out, or allow this interface to be implemented in your package to save you time evaluating.

HELP US BUILD OUT OUR MODEL LIBRARY OR IMPLEMENT YOUR OWN
TODO: Show how to do either

* [x] Vision Transformer (ViT)
* [ ] Llama-2
* [ ] Mistral-7b
* [ ] Dalle-3
* [ ] Stable Diffusion
* [ ] Magic Animate

## Datasets

Models are worthless without the data to run and evaluate them on. Sure you can poke your model with a stick by running on a single example, but the real insights come from running your model given a dataset.

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

The `Diver` object saves you the work of processing each row in the dataframe and the `Saver` takes care of writing all the results to disk so you can compare them across runs.

With plug and play models and datasets, the hope is anyone can evaluate a model against any dataset and share the results quickly and effectively.

## Model Interface

TODO

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

