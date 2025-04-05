import torch
from tqdm import tqdm
import datetime
from Data_Preprocessing import DataLoader
from utils import EvaluationMetrics
from model import CRNet
from loguru import logger
from collections import Counter

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f'/home/dingyj/malicious_domain_name_detection/dingyj/pytorch_malicious_url/log/train_{current_time}.log'
logger.add(log_file, level="INFO", encoding="utf-8")

data = DataLoader('/home/dingyj/malicious_domain_name_detection/data/dingyj/majestic_million.csv', '/home/dingyj/malicious_domain_name_detection/data/dingyj/dga.csv')
train_sets, test_sets = data.get_data()
#train_sets, test_sets = torch.load('/home/hanly/examples_1.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE=512
best_val_acc=0.
epochs = 30

model=CRNet()
print(model)
model.to(device)

loss=torch.nn.NLLLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)

for epoch in range(epochs):
    train_accuracy=0.
    train_loss=0.
    counter=0
    train_precision_accumulated=0.
    train_recall_accumulated=0.
    train_f1_accumulated=0.

    logger.info("__________________Start_Training___________________")
    model.train()
    num_batches=len(train_sets)//BATCH_SIZE

    for it in tqdm(range(num_batches)):
        urls=[url[0] for url in train_sets[it*BATCH_SIZE:(it+1)*BATCH_SIZE]]
        labels=[url[1] for url in train_sets[it*BATCH_SIZE:(it+1)*BATCH_SIZE]]

        counter+=1
        if counter%100==0:
          logger.info(('Epoch: %d, Train accuracy: %s loss: %s precession: %s recall: %s f1: %s' % (epoch, train_accuracy / counter, train_loss /counter, train_precision_accumulated / counter, train_recall_accumulated / counter, train_f1_accumulated / counter)))
          #logger.info(('Epoch: %d, Train accuracy: %s loss: %s ' % (epoch, train_accuracy / counter, train_loss /counter)))

        optimizer.zero_grad()
        urls=torch.tensor(urls).to(device)
        scores=model(urls)
        preds=torch.max(scores.cpu(),1)[1].detach().numpy()
        accuracy = EvaluationMetrics(labels,preds)
        train_accuracy+=accuracy.accuracy()
        train_precision,train_recall,train_f1 =accuracy.precision_recall_f1()
        train_precision_accumulated += train_precision
        train_recall_accumulated += train_recall
        train_f1_accumulated += train_f1

        labels=torch.tensor(labels).to(device)

        #loss_val = loss(scores.squeeze(dim=1), labels.float())
        loss_val=loss(scores,labels)
        train_loss+=loss_val.item()

        loss_val.backward()
        optimizer.step()

    logger.info("__________________Start_Validation___________________")

    model.eval()
    validation_accuracy=0.
    validation_preds = []
    validation_labels = []

    num_batches=len(test_sets)//BATCH_SIZE

    for it in range(num_batches):
        urls=[url[0] for url in test_sets[it*BATCH_SIZE:(it+1)*BATCH_SIZE]]
        labels=[url[1] for url in test_sets[it*BATCH_SIZE:(it+1)*BATCH_SIZE]]
        with torch.no_grad():
            urls = torch.tensor(urls).to(device)
            scores = model(urls)
            preds = torch.max(scores.cpu(), 1)[1].detach().numpy()
            
        validation_preds.extend(preds)
        validation_labels.extend(labels)

    accuracy = EvaluationMetrics(validation_labels, validation_preds)
    validation_accuracy = accuracy.accuracy()
    validation_precision,validation_recall,validation_f1 =accuracy.precision_recall_f1()
    #logger.info(('Epoch: %d, Validation accuracy: %s' % (epoch, validation_accuracy))) 
    logger.info(('Epoch: %d, Validation accuracy: %s precision: %s recall: %s f1: %s' % (epoch, validation_accuracy, validation_precision, validation_recall, validation_f1)))
    if best_val_acc<validation_accuracy:
        best_val_acc=validation_accuracy
        torch.save(model.state_dict(), 'forenzika_model'+str(round(validation_accuracy,5)))
    logger.info("第%s个epoch结束"%(epoch))

logger.info("Best result for train: %s"%(best_val_acc))
