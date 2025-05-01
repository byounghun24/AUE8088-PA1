from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds, target):
        pred_labels = torch.argmax(preds, dim=1)

        for t, p in zip(target.view(-1), pred_labels.view(-1)):
            self.confmat[p, t] += 1

    def compute(self):
        f1_scores = []

        for c in range(self.num_classes):
            TP = self.confmat[c, c].item()
            FP = self.confmat[c, :].sum().item() - TP
            FN = self.confmat[:, c].sum().item() - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        return torch.tensor(f1_scores)

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        pred_max = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if pred_max.shape != target.shape:
            raise ValueError(f"Pred&target shape mismatch, pred: {pred_max.shape}, target: {target.shape}")

        # [TODO] Cound the number of correct prediction
        correct = (pred_max == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
