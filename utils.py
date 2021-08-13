import torch
import transformers
from nltk.tokenize import sent_tokenize 
import nltk
nltk.download('punkt')
from data_load import *
from transformers import AutoTokenizer, AutoModel

model_checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast = True)
transformer = AutoModel.from_pretrained(model_checkpoint)

def vectorize(sent):
    encoding = tokenizer(sent, truncation=True)
    with torch.no_grad():
        output = transformer(input_ids = torch.tensor([encoding['input_ids']]), attention_mask = torch.tensor([encoding['attention_mask']]))
    return output


def find_sent(contexts, questions, answers):
    add_end_idx(answers, contexts)
    sent_vectors = []
    question_vectors = []
    
    for id, text in enumerate(contexts[:100]):
        sents = sent_tokenize(text)
        start = 0
        end = 0
        answer_start = answers[id]['answer_start']
        question =  questions[id]
        for sent_id, sent in enumerate(sents):
            end += len(sent)
            if answer_start>=start and answer_start<=end:
                break
        
        sent_output = vectorize(sent)[0].sum(dim=1)
        question_output = vectorize(question)[0].sum(dim=1)
        sent_vectors.append(sent_output)
        question_vectors.append(question_output)

    sent_vectors = torch.cat(sent_vectors, dim=0)
    question_vectors = torch.cat(question_vectors, dim=0)

    return sent_vectors, question_vectors