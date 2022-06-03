import numpy as np
import re
import nltk
from tqdm import tqdm
import pickle
import multiprocessing


stopwords = nltk.corpus.stopwords.words('english')


def get_sentences(article):
    '''
        pipline
    '''
    # TODO Firstly, convert Chinese punctuations into English ones

    # remove "." in abbreviation, "U.K" -> "UK"
    text = re.sub(r'(?<!\w)([A-Z])\.', r'\1', article)

    # remove extra space
    text = re.sub(" +", " ", text)

    # remove "," in numbers
    p = re.compile("\d+,\d\d\d")
    for com in p.finditer(text):
        mm = com.group()
        text = text.replace(mm, mm.replace(",", ""))

    # nltk cut sentences
    text = [s for s in nltk.tokenize.sent_tokenize(text)]

    # Replace periods and commas with spaces
    blocklist = ["'s", "’s", "news record", "\n", "\xa0"]
    res = []
    for s in text:
        s = s.lower()
        for i in blocklist:
            s = re.sub(f'{i}', ' ', s)
        s = re.sub(r'[\W]',' ',s)
        s = re.sub(" +", " ", s).strip()  # extra space
        if len(s)>20:  # save sentences which have more than 20 words
            s = s.split(' ')
            s_new = []
            for w in s:
                if w not in stopwords:  # stop words
                    s_new.append(w)
            res.append(s_new)

    return res


def one_process(article):
    return get_sentences(article)


if __name__ == '__main__':

    data = np.load('data/corpus_CleanNewsFilter.pkl', allow_pickle=True)

    # multiprocess
    pool = multiprocessing.Pool(processes=8)
    pbar = tqdm(total=len(data))
    all_sentences = []
    def update(res):
        if len(res)>0:
            all_sentences.extend(res)
        pbar.update()
    for article in data:
        pool.apply_async(one_process, args=(article,), callback=update)
    pool.close()
    pool.join()
    # multiprocess end

    # # single kernel
    # all_sentences = []
    # for article in tqdm(data):
    #     all_sentences.append(get_sentences(article))
    # # single kernel end

    with open('data/words_CleanNewsFilter.pkl', 'wb') as f:
        pickle.dump(all_sentences, f)