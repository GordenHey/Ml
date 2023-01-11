import re
import jieba
import numpy as np
import operator
from gensim.models import FastText
import pickle
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.models import Model
from keras import optimizers
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def stopwordslist():
    filepath = 'D:\\early fake event detection\\stop_words.txt'
    stopwords = {}
    for line in open(filepath, 'r').readlines():
        line = line.strip()
        stopwords[line] = 1
    return stopwords


def clean_str_sst(string):
    string = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', string)
    return string.strip().lower()


def readdata(file_name):
    print('The input data is: ', file_name)
    fobj = open(file_name, 'r')
    contents = []
    words = []
    word_list = []
    stopwords = stopwordslist()
    for i, line in enumerate(fobj):
        if line != '\n':
            contents.append(line)
            line_clean = clean_str_sst(line)
            line_jieba = list(jieba.cut(line_clean, cut_all=False))
            words.append(' '.join(line_jieba))
            real_words = []
            for word in line_jieba:
                if word not in stopwords:
                    real_words.append(word)
            word_list.append(real_words)
    fobj.close()
    return contents, words, word_list


class News(object):
    def __init__(self, contents, label_str=None, ids=None, author='', stopwords={}, language='ch'):
        self.ids = ids
        self.contents = contents
        self.label = label_str
        self.author = author.strip().lower()
        self.language = language
        self.event_id = []
        self.vector = None
        self.weight = None
        self.event_num = None
        self.author_news_num = None
        self.vote_value = None
        self.vote_sort = None
        self.words = None
        self.word_list = None
        self.analyzContents(stopwords)
        return

    def __str__(self):
        return self.contents

    def analyzContents(self, stopwords):
        words = []
        words_split = []
        real_words = []
        line_clean = self.clean_str_sst()
        if self.language == 'ch':
            words_split = list(jieba.cut(line_clean, cut_all=False))
        words.append(' '.join(words_split))
        for word in words_split:
            if word not in stopwords:
                real_words.append(word)
        self.words = words
        self.word_list = real_words
        return 0

    def clean_str_sst(self):
        string = self.contents
        string = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", string)
        return string.strip()


def getrain():  # 得到训练集的向量和对应标签
    f1 = open('D:\\early fake event detection\\train_data.txt', 'rb')
    tmp_list = pickle.load(f1)
    labels = []
    for i in tmp_list:
        labels.append(i.label)
    contents_list, words, word_list = readdata(
        "D:\\early fake event detection\\train_label.txt")
    ids_list = np.arange(len(contents_list))  # 为新闻创建索引,形如[1,2,3,4]是一个nddary数组,下标从0开始
    stopwords = stopwordslist()
    news_list = [News(contents_list[i], labels[i], ids_list[i], '', stopwords=stopwords) for i in
                 range(len(contents_list))]  # id从0开始
    return news_list, labels


def getest():
    f2 = open('D:\\early fake event detection\\test_data.txt', 'rb')
    tmp_list = pickle.load(f2)
    test_label = []
    for i in tmp_list:
        test_label.append(i.label)
    contents_list, words, word_list = readdata(
        "D:\\early fake event detection\\test_label.txt")
    ids_list = np.arange(len(contents_list))  # 为新闻创建索引,形如[1,2,3,4]是一个nddary数组,下标从0开始
    stopwords = stopwordslist()
    news_list = [News(contents_list[i], test_label[i], ids_list[i], '', stopwords=stopwords) for i in
                 range(len(contents_list))]  # id从0开始
    return news_list, test_label


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

    def __str__(self):
        return self.text + ' : ' + str(self.vector)

    def __repr__(self):
        return self.__str__()


class Sentence:
    def __init__(self, word_list, label):
        self.word_list = word_list
        self.label = label

    def len(self) -> int:
        return len(self.word_list)

    def __str__(self):
        word_str_list = [word.text for word in self.word_list]
        return ' '.join(word_str_list)  # 返回一个字符串,该字符串是新闻中所有词汇组成的

    def __repr__(self):
        return self.__str__()


class Event:
    def __init__(self, news_list, vector, event_id, keyword, old):
        self.news_list = news_list
        self.vector = vector
        self.id = event_id
        self.keyword = [keyword]
        self.old = old
        self.represent = None
        self.label = None
        self.update = None


def padding_sentences(input_sentences, padding_token, padding_sentence_length):
    sentences = [sentence for sentence in input_sentences]
    max_sentence_length = padding_sentence_length
    l = []
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
            l.append(sentence)
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
            l.append(sentence)
    return l


def sentence_to_vec(sentence_list):
    sentence_vec = []
    for sentence in sentence_list:
        vectors = []
        for word in sentence:
            if word == 0:
                vectors.append([0] * 10)
            else:
                vectors.append(word.vector)
        sentence_vec.append(vectors)
    return sentence_vec


def gensim_func(news_list):
    all_sentences = []
    for news in news_list:
        all_sentences.append(news.word_list)
    model = FastText(all_sentences, min_count=1, vector_size=10)
    sentence_list = []
    for i, news in enumerate(news_list):
        word_list = []
        for word in news.word_list:
            word_list.append(Word(word, model.wv[word].tolist()))
        sentence_list.append(Sentence(word_list,
                                      news.label))
    sentence_word_list = []
    for sentence in sentence_list:
        sentence_word_list.append(sentence.word_list)
    sentence_word_list = padding_sentences(sentence_word_list, 0, 60)
    sentence_vec = sentence_to_vec(sentence_word_list)
    return sentence_vec


def preprocess():
    train_news, y_train = getrain()
    test_news, y_test = getest()
    x_train = gensim_func(train_news)  # 注意一下输入的问题,输入的文本是一个60*10的矩阵
    x_test = gensim_func(test_news)
    return x_train, x_test, y_train, y_test


def real_met(a, b):  # a是预测值,b是真实值
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(a)):
        if a[i] == 1 and b[i] == 1:
            TN = TN + 1
        if a[i] == 1 and b[i] == 0:
            FN = FN + 1
        if a[i] == 0 and b[i] == 1:
            FP = FP + 1
        if a[i] == 0 and b[i] == 0:
            TP = TP + 1
    return TP, FN, FP, TN


def fake_met(a, b):  # a是预测值,b是真实值
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(a)):
        if a[i] == 1 and b[i] == 1:
            TP = TP + 1
        if a[i] == 1 and b[i] == 0:
            FP = FP + 1
        if a[i] == 0 and b[i] == 1:
            FN = FN + 1
        if a[i] == 0 and b[i] == 0:
            TN = TN + 1
    return TP, FN, FP, TN


def cal_precision(a, b, c):
    if c == 0:
        TP, FN, FP, TN = fake_met(a, b)
    else:
        TP, FN, FP, TN = real_met(a, b)
    if TP + FP == 0:
        return 0
    else:
        return round(TP / (TP + FP), 4)


def cal_recall(a, b, c):
    if c == 0:
        TP, FN, FP, TN = fake_met(a, b)
    else:
        TP, FN, FP, TN = real_met(a, b)
    if TP + FN == 0:
        return 0
    else:
        return round(TP / (TP + FN), 4)


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


def cal_f1(a, b, c):  # c=0表示是fake news为正样本,c=1表示real news为正样本
    precision = cal_precision(a, b, c)
    recall = cal_recall(a, b, c)
    if precision + recall == 0:
        return 0
    else:
        return round(2 * precision * recall / (precision + recall), 4)


def cal_auc(a, b):
    y_true = []
    y_pred = []
    for i in range(len(a)):
        y_pred.append(a[i])
        y_true.append(b[i])
    auc_score = roc_auc_score(y_true, y_pred)
    return round(auc_score, 4)


def cal_loss_func(a, b):
    count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            count = count + 1
    result = round((float(count) / len(a)), 4)
    return result


def cal_metrics(a, b):
    pred = []
    truth = []
    for i in range(len(a)):
        pred.append(a[i])
        truth.append(b[i])
    y_accuracy = accuracy_score(truth, pred)
    y_roc = roc_auc_score(truth, pred)
    y_precision = precision_score(truth, pred)
    y_recall = recall_score(truth, pred)
    y_f1 = f1_score(truth, pred)
    return y_accuracy, y_roc, y_precision, y_recall, y_f1


class threshold():
    def __init__(self, value, index):
        self.value = value
        self.index = index


def text_cnn(x_train, y_train, x_test, y_test):
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    maxlen = 60
    embed_size = 10
    tensor_input = Input(shape=(maxlen, embed_size))
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=10, kernel_size=fsz, activation='relu')(
            tensor_input)  # kernel_size中只用说明窗口大小就行,不用说宽度是多少,因为宽度默认是embedding size
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)  # 此时是100个1
        l_pool = Flatten()(l_pool)  # 将数据压成1维的数据,也就是将数据变成1x100的形式
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    output = Dense(30, activation='relu')(merge)
    output = Dense(20, activation='relu')(output)
    output = Dense(5, activation='relu')(output)
    output = Dense(units=2, activation='softmax')(output)
    model = Model(tensor_input, output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model=multi_gpu_model(model, gpus=2)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['binary_accuracy'])
    model.fit(x_train, y_train, batch_size=30 * 2, epochs=20, validation_split=0.2)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    y_predict = []
    cal_train = []
    train_list = []
    for i in range(len(y_train)):
        if y_train[i][1] == 1:
            cal_train.append(1)
        else:
            cal_train.append(0)
    for a in range(1, 10):
        r = []
        a = a / 10
        for i in range(len(train_pred)):
            if train_pred[i][1] > a:
                r.append(1)
            else:
                r.append(0)
        train_loss = cal_loss_func(r, cal_train)
        train_list.append(threshold(train_loss, a))
    cmpfun1 = operator.attrgetter('value')  # 参数为排序依据的属性，可以有多个，这里优先id，使用时按需求改换参数即可
    train_list.sort(key=cmpfun1)
    threshold_a = train_list[-1].index
    print(threshold_a)
    for i in range(len(test_pred)):
        if test_pred[i][1] > threshold_a:
            y_predict.append(1)
        else:
            y_predict.append(0)
    y_predict = np.array(y_predict)
    count1 = 0
    count2 = 0
    for i in y_predict:
        if i == 0:
            count1 = count1 + 1
        else:
            count2 = count2 + 1
    print(count1)
    print(count2)
    acc, auc, pre, recall, f1 = cal_metrics(y_predict, y_test)
    print('textcnn-test')
    print("acc:", acc)
    print("auc:", auc)
    print("pre:", pre)
    print("recall:", recall)
    print("f1:", f1)
    return model


if __name__ == '__main__':
    x_train, x_test, y_train, y_test1 = preprocess()
    for i in range(len(x_train)):
        x_train[i] = np.array(x_train[i])
        y_train[i] = np.array(y_train[i])
    for i in range(len(x_test)):
        x_test[i] = np.array(x_test[i])
    y_test = []
    for i in range(len(y_test1)):
        if y_test1[i][1] == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    for i in range(len(x_train)):
        x_train[i] = preprocessing.scale(x_train[i])
        for j in range(len(x_train[i])):
            x_train[i][j] = x_train[i][j] * 10
    for i in range(len(x_test)):
        x_test[i] = preprocessing.scale(x_test[i])
        for j in range(len(x_test[i])):
            x_test[i][j] = x_test[i][j] * 10
    model = text_cnn(x_train, y_train, x_test, y_test)

