import numpy as np
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add, concatenate
from keras.callbacks import ModelCheckpoint

from nltk.translate.bleu_score import corpus_bleu
# define the captioning model

from icg.processing.text import word_for_id


def define_model(vocab_size, max_length):

    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    # decoder1 = concatenate([fe2,se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    # decoder2 = Dense(512, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


def sample(preds, temperature=1.0):
    preds = preds.ravel()
    preds += 1e-07  # epsilon
    preds = preds / np.max(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def model_injection(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)

    # Add feature extractor model and sequence model
    add2model = add([fe2, se2])

    # Inject both output into LSTM
    Inj_model = LSTM(256)(add2model)

    # decoder1 = concatenate([fe2,se3])
    decoder1 = Dense(256, activation='relu')(Inj_model)
    # decoder2 = Dense(512, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder1)
    # tie it together [image, seq] [word]

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    # return model
    return model


def generate_desc(model, tokenizer, photo, max_length, mode):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer

        if mode == 'sample':
            yhat = sample(yhat)
        else:
            yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual,
                                     predicted,
                                     weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual,
                                     predicted,
                                     weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual,
                                     predicted,
                                     weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' %
          corpus_bleu(actual,
                      predicted,
                      weights=(0.25, 0.25, 0.25, 0.25)))
