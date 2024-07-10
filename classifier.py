import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import gradio as gr

class NewsDataset(Dataset):
    def __init__(
        self,
        csv_file,
        device,
        model_name_or_path="bert-base-uncased",
        max_length=250
    ):
        self.device = device
        self.df = pd.read_csv(csv_file)
        self.df["content"] = self.df["content"].astype(str)
        self.labels = self.df.category.unique()
        labels_dict = {label: i for i, label in enumerate(self.labels)}
        
        self.df["category"] = self.df["category"].map(labels_dict)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        content = self.df.content[index]
        label = self.df.category[index]

        inputs = self.tokenizer(
            content,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = torch.tensor(label)

        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "labels": labels.to(self.device),
        }
    

class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=5):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)

        return x
    
def training_step(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    
    return total_loss / len(data_loader.dataset)
    

def evaluation(model, test_dataloader, loss_fn):
    model.eval()

    correct_predictions = 0
    losses = []

    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = output.max(1)
        correct_predictions += torch.sum(pred == labels)

        loss = loss_fn(output, labels)
        losses.append(loss.item())

    return np.mean(losses), correct_predictions / len(test_dataloader.dataset)
    
def predict(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = CustomBert(n_classes=len(train_dataset.labels_dict))
    model.load_state_dict(torch.load("bbc_news_classifier.pth", map_location=device))
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=100,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        _, preds = torch.max(outputs, dim=1)
    
    label = list(train_dataset.labels_dict.keys())[preds.item()]
    return label

def main():
    global train_dataset, device

    N_EPOCHS = 8
    LR = 2e-5
    BATCH_SIZE = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NewsDataset(csv_file="./data/train_news.csv", device=device, max_length=100)
    test_dataset = NewsDataset(csv_file="./data/test_news.csv", device=device, max_length=100)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    model = CustomBert()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_eval, accuracy = evaluation(model, test_dataloader, loss_fn)

        print(
            f"Train Loss : {loss_train} | Eval Loss : {loss_eval} | Accuracy : {accuracy}"
        )

    torch.save(model.state_dict(), "bbc_news_classifier.pth")

    iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="News Category Classifier")
    iface.launch()

    
if __name__ == "__main__":
    main()
