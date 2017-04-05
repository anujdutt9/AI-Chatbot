# Input Story and Get Answer as output using Trained Model.

from Text_Preprocessing import *
from train import *
import numpy as np


memory_network = memoryNetwork()

while True:
    print('Use this Vocabulary to form Questions:\n' + ' , '.join(memory_network.word_id.keys()))
    story = read_text()
    print('Story: ' + ' '.join(story))
    question = input('q:')
    if question == '' or question == 'exit':
        break
    story_vector, query_vector = vectorize_ques([(story, tokenize(question))],
                                                  memory_network.word_id, 68, 4)
    prediction = memory_network.model.predict([np.array(story_vector), np.array(query_vector)])
    prediction_word_index = np.argmax(prediction)
    for word, index in memory_network.word_id.items():
        if index == prediction_word_index:
            print('Answer: ',word)

# ----------------------- EOC -----------------------