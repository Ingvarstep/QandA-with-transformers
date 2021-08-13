from data_load import *
from utils import *
from networks import V2V

train_contexts, train_questions, train_answers = read_squad('squad/train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('squad/dev-v2.0.json')

train_sent_vectors, train_question_vectors = find_sent(train_contexts, train_questions, train_answers)
val_sent_vectors, val_question_vectors = find_sent(train_contexts, train_questions, train_answers)

q2s_model = V2V(input_dim = 768, n_epochs = 300, batch_size = 32)
q2s_model.train(train_question_vectors, train_sent_vectors)