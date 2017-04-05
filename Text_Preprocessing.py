import re
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Tokenize the Input Sentences
# def tokenize(sentence):
#	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
def tokenize(sentence):
    return word_tokenize(sentence)


# Vectorize the text
# Convert Subtext, Questions to Vector Form
def vectorize_ques(data, word_id, test_max_length, ques_max_length):
    X = []
    Xq = []
    for subtext, question in data:
        x = [word_id[w] for w in subtext]
        xq = [word_id[w] for w in question]
        # let's not forget that index 0 is reserved
        X.append(x)
        Xq.append(xq)
    return (pad_sequences(X, maxlen=test_max_length),
            pad_sequences(Xq, maxlen=ques_max_length))


# Vectorize the text
# Convert Subtext, Questions, Answers to Vector Form
# Y: array[] of zero's with "1" corresponding to word representing correct answer
def vectorize_text(data, word_id, text_max_length, ques_max_length):
    X = []
    Xq = []
    Y = []
    for subtext, question, answer in data:
        x = [word_id[w] for w in subtext]
        # Save the ID of Questions using SubText
        xq = [word_id[w] for w in question]
        # Save the answers for the Questions in "Y" as "1"
        y = np.zeros(len(word_id) + 1)
        y[word_id[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=text_max_length),
            pad_sequences(Xq, maxlen=ques_max_length),
            np.array(Y))


# Read the text files
def read_text():
    text = []
    input_line = input('Story, Empty to stop: ')
    while input_line != '':
        # for now, lines have to be a full sentence
        if not input_line.endswith('.'):
            input_line += '.'
        text.extend(tokenize(input_line))
        input_line = input('Story, Empty to stop: ')
    return text

# -------------------- EOC ------------------------