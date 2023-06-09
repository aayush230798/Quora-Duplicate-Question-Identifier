import pickle
import numpy as np

cv = pickle.load(open("cv.pkl", "rb"))
stop_words = pickle.load(open("stop_words.pkl", "rb"))
# cv = load('cv.joblib')
SAFE_DIV = 0.001


def preprocess(q):
    # make the question lowercase and remove whitespace from both ends if any
    q = str(q).lower().strip()

    # remove certain very common punctuations
    punc = [',', '.', '?']
    q = "".join(c for c in str(q) if c not in punc)

    # replace some special characters
    replace_char = {'₹': ' rupee ', '$': ' dollar ', '€': ' euro ', '@': ' at ', '%': ' percent '}
    q = "".join(c if c not in replace_char.keys() else replace_char[c] for c in str(q))

    # replace contractions
    contractions = {
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }
    q = " ".join([c if c not in contractions.keys() else contractions[c] for c in str(q).split()])

    return q


def common_tokens_array(q1, q2):
    A = set(q1.split())
    B = set(q2.split())
    return len(A & B)


def total_tokens_array(q1, q2):
    A = set(q1.split())
    B = set(q2.split())
    return len(A) + len(B)


def cwc_min_array(q1, q2):
    A = set([x for x in q1.split() if x not in stop_words])
    B = set([x for x in q2.split() if x not in stop_words])
    return len(A & B) / (min(len(A), len(B)) + SAFE_DIV)


def cwc_max_array(q1, q2):
    A = set([x for x in q1.split() if x not in stop_words])
    B = set([x for x in q2.split() if x not in stop_words])
    return len(A & B) / (max(len(A), len(B)) + SAFE_DIV)


def csc_min_array(q1, q2):
    A = set([x for x in q1.split() if x in stop_words])
    B = set([x for x in q2.split() if x in stop_words])
    return len(A & B) / (min(len(A), len(B)) + SAFE_DIV)


def csc_max_array(q1, q2):
    A = set([x for x in q1.split() if x in stop_words])
    B = set([x for x in q2.split() if x in stop_words])
    return len(A & B) / (max(len(A), len(B)) + SAFE_DIV)


def ctc_min_array(q1, q2):
    A = set(q1.split())
    B = set(q2.split())
    return len(A & B) / (min(len(A), len(B)) + SAFE_DIV)


def ctc_max_array(q1, q2):
    A = set(q1.split())
    B = set(q2.split())
    return len(A & B) / (max(len(A), len(B)) + SAFE_DIV)


def first_word_same_array(q1, q2):
    return int(q1.split()[0] == q2.split()[0])


def last_word_same_array(q1, q2):
    return int(q1.split()[-1] == q2.split()[-1])


def input_array(q1, q2):
    input_query = []
    # heuristic features
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))

    input_query.append(common_tokens_array(q1, q2))
    input_query.append(total_tokens_array(q1, q2))
    input_query.append(round(common_tokens_array(q1, q2) / total_tokens_array(q1, q2), 2))

    # Advanced features

    input_query.append(cwc_min_array(q1, q2))
    input_query.append(cwc_max_array(q1, q2))
    input_query.append(csc_min_array(q1, q2))
    input_query.append(csc_max_array(q1, q2))
    input_query.append(ctc_min_array(q1, q2))
    input_query.append(ctc_max_array(q1, q2))
    input_query.append(first_word_same_array(q1, q2))
    input_query.append(last_word_same_array(q1, q2))

    # bow features
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 15), q1_bow, q2_bow))
