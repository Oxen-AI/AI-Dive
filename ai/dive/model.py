# Base class for all models

class Model:
    def __init__(self):
        self.build()

    def build(self):
        raise NotImplementedError("Model must implement build()")
    
    def process(self, data):
        raise NotImplementedError("Model must implement process()")

