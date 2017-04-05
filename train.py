# AI Chatbot
# Dataset from Facebook AI Research Page

import os
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from functools import reduce
from nltk.tokenize import word_tokenize
import tarfile
from Text_Preprocessing import *
import numpy as np
import pickle
import keras
import re
import random



# Parse the text data from bAbI tasks format.
# If only_supporting is True, only the sentences supporting the answer are kept
# Data format: id Text
def parse_text(lines, only_supporting=False):
    # Make two new Lists/Arrays to store data and text
    data = []
    text = []
    # Read the text from bAbI Dataset
    for line in lines:
        # Lines input from Text
        # Format: ID Line
        line = line.decode('utf-8').strip()
        # Separate ID and Text from Input Lines
        # Lines contain both Text as well as Question Answers
        id, line = line.split(' ', 1)
        # Convert ID to int type
        id = int(id)
        # If ID = 1, it is the text/story
        if id == 1:
            text = []
        # If there is a tab space in the input lines, it contains Question, Answer, Supporting Text ID
        # and the Supporting Line Number in the Text
        # Format: Question? Answer Line_Number
        if '\t' in line:
            ques, ans, supporting = line.split('\t')
            # Take in the Question and Tokenize it into words
            ques = tokenize(ques)
            subtext = None
            # Keep only the supporting text from Question; only_supporting = True
            if only_supporting:
                # Map the Supporting Text ID as int
                supporting = list(map(int, supporting.split()))
                # subtext: List of the sentences supporting the Questions
                subtext = [text[i - 1] for i in supporting]
            else:
                # Contains all the related Text Lines in the file
                # Relation using Supporting ID
                subtext = [x for x in text if x]

            # Data containes tokenized first two sentences, then the answers.
            # All tokenized words in form of arrays
            # Tokenized text in form of array of array
            # All this in a List
            # data: array of all such Lists
            # Format: [([[First sentence Tokeinized],[Second Sentence Tokenized]],[Question with Answer Tokenized]), ....]
            data.append((subtext, ques, ans))
            text.append('')
        else:
            sent = tokenize(line)
            text.append(sent)
    return data


# Read the file, retrieve the stories and convert sentences into a single story
def get_stories(file, only_supporting=False, max_length=None):
    # Data containes tokenized first two sentences, then the answers.
    # All tokenized words in form of arrays
    # Tokenized text in form of array of array
    # All this in a List
    # data: array of all such Lists
    # Format: [([[First sentence Tokeinized],[Second Sentence Tokenized]],[Question with Answer Tokenized]), ....]
    data = parse_text(file.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    # flatten: Takes two sentences and makes one array, 2nd array of Question answer in a list
    # Format: [([First sentence Tokeinized, Second Sentence Tokenized],[Question with Answer Tokenized]), ....]
    data = [(flatten(text), question, answer) for text, question, answer in data if
            not max_length or len(flatten(text)) < max_length]
    return data



class memoryNetwork(object):
    FILE_NAME = 'model'
    VOCAB_FILE_NAME = 'model_vocab'

    def __init__(self):
        if (os.path.exists(memoryNetwork.FILE_NAME) and
                os.path.exists(memoryNetwork.VOCAB_FILE_NAME)):
            self.load()
        else:
            self.train()
            self.store()


    def load(self):
        self.model = keras.models.load_model(memoryNetwork.FILE_NAME)
        with open(memoryNetwork.VOCAB_FILE_NAME, 'rb') as file:
            self.word_id = pickle.load(file)


    def store(self):
        self.model.save(memoryNetwork.FILE_NAME)
        with open(memoryNetwork.VOCAB_FILE_NAME, 'wb') as file:
            pickle.dump(self.word_id, file)


    def train(self):
        # Load the bAbI Dataset
        try:
            dataPath = get_file('babi-tasks-v1-2.tar.gz',
                                origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise

        tar = tarfile.open(dataPath)

        # Load the Single Supporting Fact and Two Supporting Fact files
        challenges = {
            # QA1 with 10,000 samples
            'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
            # QA2 with 10,000 samples
            'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
        }

        challenge_type = 'single_supporting_fact_10k'

        challenge = challenges[challenge_type]

        # Extract the Text from single_supporting_fact_10k file
        print('Extracting stories for the challenge:', challenge_type)

        # Load the Testing and Training Text Data
        train_stories = get_stories(tar.extractfile(challenge.format('train')))
        test_stories = get_stories(tar.extractfile(challenge.format('test')))


        # Initialize vocabulary as as Set
        # Create a Vocabulary list with all words occuring only once
        vocab = set()
        for text, ques, answer in train_stories + test_stories:
            vocab |= set(text + ques + [answer])

        # Sort the words in Vocabulary List
        vocab = sorted(vocab)

        # Get the max length of the Vocabulary, text and Questions
        vocab_size = len(vocab) + 1

        # text_max_length: length of th subtext; no. of subtexts
        text_max_length = max(list(map(len, (x for x, _, _ in train_stories + test_stories))))

        # ques_max_length: length of questions in input.
        ques_max_length = max(list(map(len, (x for _, x, _ in train_stories + test_stories))))


        print('-')

        print('Vocab size:', vocab_size, 'unique words')
        print('Story max length:', text_max_length, 'words')
        print('Query max length:', ques_max_length, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))

        print('-')

        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])

        print('-')

        print('Vectorizing the word sequences...')


        # Vectorize the Training and Testing Data
        self.word_id = dict((c, i + 1) for i, c in enumerate(vocab))

        # inputs_train: Matrix of Arrays; Arrays containing vectorized sentences
        # ques_train: Matrix of Arrays; Each array has 4 values; Each value corresponds to a character.
        # answers_train: Matrix of Arrays; Each array contains a single "1", index corresponding to answer
        inputs_train, ques_train, answers_train = vectorize_text(train_stories,
                                                                 self.word_id,
                                                                 text_max_length,
                                                                 ques_max_length)

        inputs_test, ques_test, answers_test = vectorize_text(test_stories,
                                                              self.word_id,
                                                              text_max_length,
                                                              ques_max_length)



        # Dataset Analysis
        print('-')

        print('inputs: integer tensor of shape (samples, max_length)')
        print('inputs_train shape:', inputs_train.shape)
        print('inputs_test shape:', inputs_test.shape)

        print('-')

        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', ques_train.shape)
        print('queries_test shape:', ques_test.shape)

        print('-')

        print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        print('answers_train shape:', answers_train.shape)
        print('answers_test shape:', answers_test.shape)

        print('-')

        print('Compiling...')



        # Define Placeholders
        input_sequence = Input((text_max_length,))
        question = Input((ques_max_length,))

        # ---------------------------------- Encode the Data ----------------------------------------
        # Embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()

        input_encoder_m.add(Embedding(input_dim=vocab_size,
                                      output_dim=64))

        input_encoder_m.add(Dropout(0.3))
        # Output: (samples, text_maxlen, embedding_dim)


        # Embed the input into a sequence of vectors of size ques_max_length
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=vocab_size,
                                      output_dim=ques_max_length))

        input_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)


        # Embed the question into a sequence of vectors
        question_encoder = Sequential()

        question_encoder.add(Embedding(input_dim=vocab_size,
                                       output_dim=64,
                                       input_length=ques_max_length))

        question_encoder.add(Dropout(0.3))
        # output: (samples, query_maxlen, embedding_dim)


        # Encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(32)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(0.3)(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        self.model = Model([input_sequence, question], answer)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the Model
        self.model.fit([inputs_train, ques_train], answers_train,
                  batch_size=32,
                  epochs=120,
                  validation_data=([inputs_test, ques_test], answers_test))

# -------------------- EOC ----------------------