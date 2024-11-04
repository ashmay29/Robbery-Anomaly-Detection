import torch

class AnomalyDetector:
    def __init__(self, model_path):
        self.model = torch.load(model_path)

    def detect_anomaly(self, trajectory):
        # Assuming the model returns a 0 for normal and 1 for anomaly
        result = self.model(trajectory)
        return 'anomaly' if result == 1 else 'normal'
