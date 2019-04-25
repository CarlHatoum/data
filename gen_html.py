import nltk
from nltk import FreqDist

import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy


df = pd.read_json('Automotive_5.json', lines=True)
# remove unwanted characters, numbers and symbols
df['reviewText'] = df['reviewText'].str.replace("[^a-zA-Z#]", " ")
# remove short words (length < 3)
df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

reviews = [remove_stopwords(r.split()) for r in df['reviewText']]

# make entire text lowercase
reviews = [r.lower() for r in reviews]


nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
#print(tokenized_reviews[1])

def lemmatization(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

reviews_2 = lemmatization(tokenized_reviews)

reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

df['reviews'] = reviews_3

def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()
  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms)
  return d.to_json(orient='records')

def fit_model(ParCh):

    #processed_x = pre_processing(sex, title, age, Pclass, cabin, SibSp, ParCh, fare, embarked, scaler)
    #pred_class = model.predict(processed_x)

    #print(tokenized_reviews[1])

    words = freq_words(df['reviews'], ParCh)

    # HTML formatting
    html = ''
    # Hard binary classifier
    '''
    html = addContent(html, header('Oh...', color='darkblue'))
    if pred == 1:
        html = addContent(html, box('You would have survived!'))
    else:
        html = addContent(html, box('You would have died.'))
    '''

    # Soft binary classifier
    html = addContent(html, box('Most popular words :'))
    color = '#6EA1D1'

    html = addContent(html, header(words))

    return f'<div>{html}</div>'


# Create an HTML header
def header(text, color='black', gen_text=None):

    if gen_text:
        raw_html = f'<h1 style="margin-top:16px;color: {color};font-size:54px"><center>' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</center></h1>'
    else:
        raw_html = f'<h1 style="margin-top:12px;color: {color};font-size:54px"><center>' + str(
            text) + '</center></h1>'
    return raw_html

# Create an HTML box of text
def box(text, gen_text=None):

    if gen_text:
        raw_html = '<div style="padding:8px;font-size:28px;margin-top:28px;margin-bottom:14px;">' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</div>'

    else:
        raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 28px;">' + str(
            text) + '</div>'
    return raw_html

# Concatenate html content together
def addContent(old_html, raw_html):

    old_html += raw_html
    return old_html
