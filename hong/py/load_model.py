##### 정리


##### import Library #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import re
import random

from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 경고표시 지우기
import warnings

warnings.filterwarnings(action="ignore")


##### import Library #####

##### function #####
def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'{2,}", "\'", string)
    string = re.sub(r"\'", "", string)

    return string.lower()


def model_load(file_path):
    print("모델 불러오는 중...")
    # model load
    filepath = file_path
    model = load_model(filepath, custom_objects=None, compile=True)

    return model


def data_load(dataframe):
    print("데이터 불러오는 중...")
    df = pd.read_csv(dataframe, encoding="utf-8-sig", index_col=0)
    review_df = df.dropna(axis=0)

    return review_df


def data_preprocessing(review_df):
    print("데이터 정제 중...")
    # clean_str 함수를 통해서 X데이터를 정제해주어야함.
    train_test_X = [clean_str(sentence) for sentence in review_df["review"]]
    # 문장을 띄어쓰기 단위로 단어 분리
    sentences = [sentence.split(' ') for sentence in train_test_X]
    for i in range(5):
        print(sentences[i])

    # 문장의 길이를 시각화
    # padding의 기준을 잡는 부분
    sentence_len = [len(sentence) for sentence in sentences]
    sentence_len.sort()
    plt.plot(sentence_len)
    plt.show()

    # 총 3956974행의 데이터 중 30문장이내의 데이터는 3401248으로 85.9%에 육박.
    # 따라서, padding의 기준을 30로 잡겠음.
    print(sum([int(l <= 30) for l in sentence_len]))

    sentence_new = []
    for sentence in sentences:
        sentence_new.append([word[:5] for word in sentence][:30])

    sentences = sentence_new

    for i in range(5):
        print(sentences[i])

    tokenizer = Tokenizer(num_words=3000)
    tokenizer.fit_on_texts(sentences)

    return sentences, tokenizer


def model_random_test(tokenizer, mango):
    print("모델 테스트 중...")
    # 망고 플레이트에서 가게 랜덤으로 뽑아서 점수 예측.
    cnt = 0
    mango_store_list = []
    try:
        for store in mango["이름"].unique():
            mango_store_list.append(store)

        choiceList = random.choice(mango_store_list)

        data = mango[mango["이름"] == choiceList]
        print("뽑은 가게는 {}입니다.".format(str(data["이름"].unique())))

        score = 0
    except:
        print("망고플레이트 Database안에 없는 가게입니다.")

    # 망고플레이트 리뷰 테스트
    for review in data.iloc[:, -1]:
        test_sentence = review
        test_sentence = test_sentence.split(' ')
        test_sentences = []
        now_sentence = []
        for word in test_sentence:
            now_sentence.append(word)
            test_sentences.append(now_sentence[:])

        test_X_1 = tokenizer.texts_to_sequences(test_sentences)
        test_X_1 = pad_sequences(test_X_1, padding='post', maxlen=30)
        prediction = model.predict(test_X_1)
        for idx, sentence in enumerate(test_sentences):
            review_score = np.argmax((prediction[idx]).round(3))

        print("리뷰\n", review)
        print("실제 점수 : ", data.iloc[cnt, 2])
        print("예측한 점수 : ", review_score)
        print("-" * 70)
        cnt += 1


def model_test(tokenizer, mango):
    print("모델 테스트 중...")
    print("리뷰 모델 테스트.")
    store = input("가게 이름을 입력하시오 : ")

    data = mango[mango["이름"] == store]
    if data.empty is True:
        print("Database에 해당 가게가 없습니다.")

    else:
        score_sum = 0
    cnt = 0

    for review in data.iloc[:, -1]:
        test_sentence = review
        test_sentence = test_sentence.split(' ')
        test_sentences = []
        now_sentence = []
        for word in test_sentence:
            now_sentence.append(word)
            test_sentences.append(now_sentence[:])

        test_X_1 = tokenizer.texts_to_sequences(test_sentences)
        test_X_1 = pad_sequences(test_X_1, padding='post', maxlen=30)
        prediction = model.predict(test_X_1)
        for idx, sentence in enumerate(test_sentences):
            review_score = np.argmax((prediction[idx]).round(3))
        print("리뷰\n", review)
        print("실제 점수 : ", data.iloc[cnt, 2])
        print("예측한 점수 : ", review_score)
        print("-" * 70)
        score_sum += review_score
        cnt += 1
    print("예측한 평균 평점 : {}".format((score_sum / cnt + 1).round(2)))


##### function #####

##### Main #####
if __name__ == "__main__":

    path = "./data/model/model_lstm_220110.h5"
    model = model_load(file_path=path)

    data = "./data/review_data/total_review.csv"
    df = data_load(data)

    # 망고플레이트 리뷰 데이터
    mango = pd.read_excel("./data/망고플레이트 리뷰 종합.xlsx")

    _, token = data_preprocessing(df)
    try:
        opinion = input("어느 기능을 사용하시겠습니까(1:작동테스트 / 2:가게입력) : ")
        if opinion == 1:
            model_random_test(tokenizer=token, mango=mango)
        else:
            model_test(tokenizer=token, mango=mango)
    except:
        print("번호를 잘못 입력하셨습니다. 1번과 2번중 골라주세요.")