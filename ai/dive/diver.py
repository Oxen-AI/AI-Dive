# Iterates over a dataset and runs a model
from tqdm import tqdm

class Diver:
    def __init__(self, model, dataset, saver=None):
        self.model = model
        self.dataset = dataset
        self.saver = saver
    
    def run(self, args):
        # {
        #     'dataset': args.dataset,
        #     'output': args.output,
        #     'num_samples': args.num_samples
        # }
        
        # Load the dataset
        self.dataset.load()
        
        # TODO: can abstract and parallelize if need be
        outputs = []
        num_items = self.dataset.len()
        for i in tqdm(range(num_items)):
            try:
                row = self.dataset.item_at(i)
                output = self.model.process(row)
                outputs.append(output)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
            
            if self.saver is not None and self.saver.save_every is not None:
                if i % self.saver.save_every == 0:
                    self.saver.save(outputs)
        
        # Save at the end
        if self.saver is not None:
            self.saver.save(outputs)
        
        return outputs