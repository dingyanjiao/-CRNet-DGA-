from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class EvaluationMetrics:
    
    def __init__(self,  labels, predictions):
        self.predictions = predictions 
        self.labels = labels
        
    def accuracy(self):
        return accuracy_score(self.labels, self.predictions)
    
    def precision_recall_f1(self):
        precision, recall, f1, _ = precision_recall_fscore_support(self.labels, self.predictions, average='binary')
        return precision, recall, f1