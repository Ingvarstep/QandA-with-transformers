from data_load import *
from tqdm import tqdm 
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering
from transformers import AdamW
from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open('documents.txt', 'r') as f:
    contexts = f.readlines()
with open('questions.txt', 'r') as f:
    questions = f.readlines()

with open('QandA_model.pt', 'rb') as f:
    model  = torch.load(f).to(device)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

encodings = tokenizer(contexts, questions, truncation=True, padding=True)

dataset = Dataset(encodings)
batch_size = 16
loader = DataLoader(dataset, batch_size)

with torch.no_grad():
    start_positions = []
    end_positions = []
    counter = -batch_size
    for batch in loader:
        counter+=batch_size
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        start_preds = outputs['start_logits'].argmax(dim=-1)
        end_preds = outputs['end_logits'].argmax(dim=-1)
        for id, pos in enumerate(start_preds):
            batch_id = counter+id
            start_pos = encodings.token_to_chars(batch_id, start_preds[id].item()).start
            end_pos = encodings.token_to_chars(batch_id, end_preds[id].item()).end
            print('\n', contexts[batch_id][start_pos:end_pos])
            start_positions.append(start_pos)
            end_positions.append(end_pos)