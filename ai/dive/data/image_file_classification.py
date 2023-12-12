


from ai.dive.data.file_classification import FileClassification
from ai.dive.data.label_reader import LabelReader

from datasets import Dataset as HFDataset
from transformers import AutoImageProcessor

from tqdm import tqdm
from PIL import Image


class ImageFileClassificationDataset(FileClassification):
    def __init__(
        self,
        data_dir: str,
        file: str,
        label_reader: LabelReader,
        img_processor: AutoImageProcessor,
        path_key: str = 'file',
        label_key: str = 'label',
        num_samples=-1
    ):
        self.label_reader = label_reader
        self.img_processor = img_processor
        self.num_samples = num_samples
        
        super().__init__(
            data_dir=data_dir,
            file=file,
            path_key=path_key,
            label_key=label_key
        )
        self.build()


    def to_hf_dataset(self):
        def transform(example_batch):
            # Take a list of PIL images and turn them to pixel values
            inputs = self.img_processor([x for x in example_batch['images']], return_tensors='pt')

            # Don't forget to include the labels!
            inputs['labels'] = example_batch['labels']
            return inputs
        
        data_dict = {
            "images": [],
            "labels": []
        }
        for data in tqdm(self):
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
            label = data[self.label_key]
            label_idx = self.label_reader.index_of(label)
            data_dict["labels"].append(label_idx)
            if self.num_samples > 0 and len(data_dict["labels"]) >= self.num_samples:
                break

        # Create dataset object
        dataset = HFDataset.from_dict(data_dict)
        dataset = dataset.with_transform(transform)

        return dataset
