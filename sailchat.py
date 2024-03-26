import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

# Preprocess data
texts = []
labels = []
classes = set()

for intent in intents['intents']:
    tag = intent['tag']
    if isinstance(tag, list):
        tag = ' '.join(tag)  # Convert the list to a string
    classes.add(tag)
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(tag)

# Encode labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, numeric_labels, test_size=0.2, random_state=42)

# Create PyTorch dataset
class ChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = 50  # Adjust this value based on your data

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load tokenizer and create datasets
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_dataset = ChatDataset(X_train, y_train, tokenizer)
test_dataset = ChatDataset(X_test, y_test, tokenizer)

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(classes))

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
epochs = 45

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# Save the trained model
model.save_pretrained('chat_model')
tokenizer.save_pretrained('chat_model')

