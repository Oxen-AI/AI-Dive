
from ai.dive.cleaners.cleaner import Cleaner

class QACleaner(Cleaner):
    def clean(self, dataset):
        print(f'Cleaning dataset {dataset}')
