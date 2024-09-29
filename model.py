import re
import codecs
import numpy as np
import pickle
from numpy import zeros, int8, log
from pyvi import ViTokenizer
from pylab import random

class PLSA:
    def __init__(self, K, maxIteration=30, threshold=10.0, topicWordsNum=5):
        self.K = K  # Số chủ đề
        self.maxIteration = maxIteration  # Số lần lặp tối đa
        self.threshold = threshold  # Ngưỡng dừng
        self.topicWordsNum = topicWordsNum  # Số từ hàng đầu trong mỗi chủ đề
        self.word2id = {}  
        self.id2word = {}  
        self.Pz = zeros(self.K)  # Xác suất của các chủ đề
        self.lamda = None  # Ma trận lambda P(d|z)
        self.theta = None  # Ma trận theta P(w|z)
        self.p = None  # Ma trận xác suất trung gian P(z|d,w)
        self.stopwordsFilePath='stopwords.dic'  
    
    def preprocessing(self, dataset):
        """
        Xử lý trước tài liệu, bao gồm tokenization, loại bỏ stopwords và xây dựng
        word2id, id2word và ma trận tần suất từ.
        """
        # Load stopwords
        file = codecs.open(self.stopwordsFilePath, 'r', 'utf-8')
        self.stopwords = [line.lower().strip() for line in file]
        file.close()

        documents = dataset
        N = len(documents)  # Số tài liệu
        wordCounts = []
        currentId = 0

        # Xây dựng word2id và id2word, đếm số lượng từ
        for document in documents:
            segList = ViTokenizer.tokenize(document).split()
            wordCount = {}
            for word in segList:
                word = word.lower().strip()
                if len(word) > 1 and not re.search('[0-9]', word) and word not in self.stopwords:
                    if word not in self.word2id:
                        self.word2id[word] = currentId
                        self.id2word[currentId] = word
                        currentId += 1
                    if word in wordCount:
                        wordCount[word] += 1
                    else:
                        wordCount[word] = 1
            wordCounts.append(wordCount)

        M = len(self.word2id)  # Số từ
        X = zeros([N, M], int8)
        for word in self.word2id.keys():
            j = self.word2id[word]
            for i in range(0, N):
                if word in wordCounts[i]:
                    X[i, j] = wordCounts[i][word]

        return N, M, X

    def initializeParameters(self, N, M):
        self.lamda = random([N, self.K])  # P(d|z)
        self.theta = random([self.K, M])  # P(w|z)
        self.Pz = random([self.K])  # P(z)
        self.Pz /= sum(self.Pz)  # Chuẩn hóa Pz để tổng bằng 1

        self.p = np.zeros((N, M, self.K))  # Ma trận xác suất P(z|d,w)

        # Chuẩn hóa lambda và theta
        for i in range(0, N):
            self.lamda[i, :] /= sum(self.lamda[i, :])

        for i in range(0, self.K):
            self.theta[i, :] /= sum(self.theta[i, :])

    def EStep(self, N, M, X):
        """
        Thực hiện bước E: Tính toán ma trận xác suất P(z|d,w).
        """
        for i in range(0, N):
            for j in range(0, M):
                denominator = np.sum(self.Pz[k] * self.lamda[i,k] * self.theta[k,j] for k in range(self.K))
                if denominator == 0:
                    self.p[i,j,:] = 1.0/self.K
                else:
                    for k in range(self.K):
                        self.p[i,j,k] = (self.Pz[k] * self.lamda[i,k] * self.theta[k,j]) / denominator    

    def MStep(self, N, M, X):
        # Cập nhật theta (P(w|z))
        for k in range(0, self.K):
            for j in range(0, M):
                self.theta[k,j] = np.sum(X[i,j] * self.p[i,j,k] for i in range(N))
            self.theta[k, :] /= np.sum(self.theta[k, :])

        # Cập nhật lambda (P(d|z))
        for k in range(self.K):
            for i in range(N):
                self.lamda[i,k] = np.sum(X[i,j] * self.p[i,j,k] for j in range(M))
            self.lamda[:, k] /= np.sum(self.lamda[:, k])

        for k in range(self.K):
            self.Pz[k] = np.sum(self.p[i,j,k] * X[i,j] for i in range(N) for j in range(M))

        # Chuẩn hóa Pz: chia cho tổng số lần xuất hiện của tất cả các từ và tài liệu
        total = np.sum(X)
        self.Pz /= total


        
    def LogLikelihood(self, N, M, X):
        loglikelihood = 0
        for i in range(0, N):
            for j in range(0, M):
                tmp = 0
                for k in range(0, self.K):
                    tmp += self.theta[k, j] * self.lamda[i, k] * self.Pz[k]
                if tmp > 0:
                    loglikelihood += X[i, j] * log(tmp)
        return loglikelihood

    def train(self, dataset):
        N, M, X = self.preprocessing(dataset)
        self.initializeParameters(N, M)

        oldLoglikelihood = 1
        newLoglikelihood = 1
        # print("training")
        for i in range(0, self.maxIteration):
            self.EStep(N, M, X)
            self.MStep(N, M, X)
            # print(self.Pz)
            newLoglikelihood = self.LogLikelihood(N, M, X)
            if oldLoglikelihood != 1 and newLoglikelihood - oldLoglikelihood < self.threshold:
                break
            oldLoglikelihood = newLoglikelihood
            # print(newLoglikelihood)

        topic = self.get_top_words()
        return self.p, self.Pz, self.lamda, self.theta, topic, self.id2word

    def get_top_words(self):
        topic = []
        for i in range(0, self.K):
            topicword = []
            ids = self.theta[i, :].argsort()
            for j in ids:
                topicword.insert(0, self.id2word[j])
            tmp = ''
            for word in topicword[0:min(self.topicWordsNum, len(topicword))]:
                tmp += word + ' '
            topic.append(tmp)
        return topic


    def test(self, data_test):
        segList= ViTokenizer.tokenize(data_test[0])    
        xtest = zeros(len(self.id2word))
        for word in segList.split(' '):
            word = word.lower().strip()
            # print(word)
            for i in range(len(self.id2word)):
                if word == self.id2word[i]:
                    xtest[i] += 1

        z = []
        for k in range(self.K):
            p_new = 1
            # print("Theta[",k,":]")
            for i in range(len(xtest)):
                if xtest[i] != 0:
                    if self.theta[k,i] > 1e-6:
                        # print(self.theta[k,i], " * ", xtest[i])
                        p_new *= self.theta[k,i] * xtest[i]
                        # print(p_new)
            p_new *= self.Pz[k]
            # print("Pzk|D", p_new)
            z.append(p_new)
        # print(z)        
        main_topic = z.index(max(z))
        return main_topic

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'Pz': self.Pz,
                'theta': self.theta,
                'lamda': self.lamda,
                'id2word': self.id2word,
                'word2id': self.word2id,
            }, f)
        # print(f'Model saved to {filename}')

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            self.Pz = params['Pz']
            self.theta = params['theta']
            self.lamda = params['lamda']
            self.id2word = params['id2word']
            self.word2id = params['word2id']
        # print(f'Model loaded from {filename}')
        topic = self.get_top_words()
        self.K = len(self.Pz)
        return self.p, self.Pz, self.lamda, self.theta, topic, self.id2word


# file = codecs.open('data_train.txt', 'r', 'utf-8')
# train_data = [document.strip() for document in file] 
# file.close()


# model = PLSA(K=3, maxIteration=70,threshold=5.0, topicWordsNum=4)

# m,n,x = model.preprocessing(dataset=train_data)
# p, Pz, lamda, theta, wordTop, id2w = model.train(dataset=train_data)



# print("Pz \n")
# print(Pz)
# print(wordTop)

# file = codecs.open('data_test.txt', 'r', 'utf-8')
# test_data = [document.lower().strip() for document in file] 
# file.close()

# m= model.test(test_data)
# # print(z)
# print("Topic: ",m)
