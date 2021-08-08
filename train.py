from data_load import *
from tqdm import tqdm 
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering
from transformers import AdamW
from torch.utils.data import DataLoader

train_contexts, train_questions, train_answers = read_squad('squad/train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('squad/dev-v2.0.json')

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)


add_token_positions(tokenizer, train_encodings, train_answers)
add_token_positions(tokenizer, val_encodings, val_answers)


train_dataset = Dataset(train_encodings)
val_dataset = Dataset(val_encodings)



model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    print("Epoch has startes")
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

with open('QandA_model.pt', 'wb') as f:
    torch.save(model, f)