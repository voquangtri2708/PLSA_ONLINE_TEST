import re
import codecs
import numpy as np
import pickle
from numpy import zeros, int8, log, float32
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
        word2id, id2word và ma trận TF-IDF.
        """
        # Load stopwords
        file = codecs.open(self.stopwordsFilePath, 'r', 'utf-8')
        self.stopwords = [line.lower().strip() for line in file]
        file.close()

        documents = dataset
        N = len(documents)  # Số tài liệu
        wordCounts = []  # Lưu tần suất từ trong từng tài liệu
        df = {}  # Lưu số lượng tài liệu chứa từ
        currentId = 0

        # Xây dựng word2id, id2word và đếm số lượng từ trong mỗi tài liệu
        for document in documents:
            segList = ViTokenizer.tokenize(document).split()
            wordCount = {}
            seen_words = set()  # Để theo dõi từ đã xuất hiện trong tài liệu

            for word in segList:
                word = word.lower().strip()
                if len(word) > 1 and not re.search('[0-9]', word) and word not in self.stopwords:
                    if word not in self.word2id:
                        self.word2id[word] = currentId
                        self.id2word[currentId] = word
                        currentId += 1
                    
                    # Tính TF
                    if word in wordCount:
                        wordCount[word] += 1
                    else:
                        wordCount[word] = 1

                    # Tính DF (chỉ tăng nếu từ chưa xuất hiện trong tài liệu này)
                    if word not in seen_words:
                        df[word] = df.get(word, 0) + 1
                        seen_words.add(word)

            wordCounts.append(wordCount)

        M = len(self.word2id)  # Số từ (số cột trong ma trận)
        X = zeros([N, M], float32)  # Ma trận TF-IDF, sử dụng float32 để lưu giá trị TF-IDF

        # Tính IDF và TF-IDF cho mỗi từ trong mỗi tài liệu
        for word, j in self.word2id.items():
            idf = log(N / (1 + df[word]))  # Tính IDF cho từ

            for i in range(N):
                if word in wordCounts[i]:
                    tf = wordCounts[i][word]  # Tần suất từ trong tài liệu
                    X[i, j] = tf * idf  # Tính TF-IDF và lưu vào ma trận

        return N, M, X  # Trả về số tài liệu, số từ, và ma trận TF-IDF


    def initializeParameters(self, N, M):
        self.lamda = random([N, self.K])  # P(d|z)
        self.theta = random([M, self.K])  # P(w|z)
        self.Pz = random([self.K])  # P(z)
        self.Pz /= sum(self.Pz)  # Chuẩn hóa Pz để tổng bằng 1

        self.p = np.zeros((N, M, self.K))  # Ma trận xác suất P(z|d,w)

        # Chuẩn hóa lambda và theta
        for k in range(0, self.K):
            self.lamda[:, k] /= sum(self.lamda[:, k])
            self.theta[:, k] /= sum(self.theta[:, k])
            

    def EStep(self, N, M, X):
        """
        Thực hiện bước E: Tính toán ma trận xác suất P(z|d,w).
        """
        for i in range(0, N):
            for j in range(0, M):
                denominator = np.sum(self.Pz[k] * self.lamda[i,k] * self.theta[j,k] for k in range(self.K))
                if denominator == 0:
                    self.p[i,j,:] = 1.0/self.K
                else:
                    for k in range(self.K):
                        self.p[i,j,k] = (self.Pz[k] * self.lamda[i,k] * self.theta[j,k]) / denominator    

    def MStep(self, N, M, X):
        # Cập nhật theta (P(w|z))
        for k in range(0, self.K):
            for j in range(0, M):
                self.theta[j,k] = np.sum(X[i,j] * self.p[i,j,k] for i in range(N))
            self.theta[:, k] /= np.sum(self.theta[:, k])

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
                    tmp += self.theta[j, k] * self.lamda[i, k] * self.Pz[k]
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
            ids = self.theta[:, i].argsort()
            for j in ids:
                topicword.insert(0, self.id2word[j])
            tmp = ''
            for word in topicword[0:min(self.topicWordsNum, len(topicword))]:
                tmp += word + ' '
            topic.append(tmp)
        return topic


    # def test(self, data_test):
    #     segList= ViTokenizer.tokenize(data_test[0])    
    #     xtest = zeros(len(self.id2word))
    #     for word in segList.split(' '):
    #         word = word.lower().strip()
    #         # print(word)
    #         for i in range(len(self.id2word)):
    #             if word == self.id2word[i]:
    #                 xtest[i] += 1

    #     z = []
    #     for k in range(self.K):
    #         p_new = 1
    #         for i in range(len(xtest)):
    #             if xtest[i] != 0:
    #                 # if self.theta[k,i] > 1e-6:
    #                     p_new *= self.theta[i, k] * xtest[i]
    #                     # print(p_new)
    #         if p_new == 1:
    #             z.append(0)
    #         else:
    #             p_new *= self.Pz[k]
    #             z.append(p_new)
    #     # print(z)

    #     if all(i == 0 for i in z):
    #         return "Khong thuoc chu de nao trong cac chu de tren"  
    #     else:
    #         main_topic = z.index(max(z))
    #         return 'Pz'+str(main_topic)


    def test(self, data_test):
        segList = ViTokenizer.tokenize(data_test[0])    
        xtest = np.zeros(len(self.id2word))
        
        for word in segList.split(' '):
            word = word.lower().strip()
            for i in range(len(self.id2word)):
                if word == self.id2word[i]:
                    xtest[i] += 1

        z = []
        for k in range(self.K):
            log_p_new = 0  # Initialize log probability
            for i in range(len(xtest)):
                if xtest[i] != 0:
                    log_p_new += xtest[i] * np.log(self.theta[i, k] + 1e-10)  # Use log and prevent log(0)
            
            log_p_new += np.log(self.Pz[k] + 1e-10)  # Add log of prior probability Pz[k]
            z.append(log_p_new)

        if all(i == float('-inf') for i in z):
            return "Khong thuoc chu de nao trong cac chu de tren"  
        else:
            main_topic = z.index(max(z))
            return 'Pz' + str(main_topic)


    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'Pz': self.Pz,
                'theta': self.theta,
                'lamda': self.lamda,
                'id2word': self.id2word,
                'word2id': self.word2id,
            }, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            self.Pz = params['Pz']
            self.theta = params['theta']
            self.lamda = params['lamda']
            self.id2word = params['id2word']
            self.word2id = params['word2id']
        topic = self.get_top_words()
        self.K = len(self.Pz)
        return self.p, self.Pz, self.lamda, self.theta, topic, self.id2word





# file = codecs.open('data_train.txt', 'r', 'utf-8')
# train_data = [document.strip() for document in file] 
# file.close()


# model = PLSA(K=3, maxIteration=70,threshold=5.0, topicWordsNum=4)

# # m,n,x = model.preprocessing(dataset=train_data)
# p, Pz, lamda, theta, wordTop, id2w = model.train(dataset=train_data)



# print("Pz \n")
# print(Pz)
# print(wordTop)

# file = codecs.open('data_test.txt', 'r', 'utf-8')
# test_data = [document.lower().strip() for document in file] 
# file.close()

# m,z= model.test(test_data)
# print(z)
# print("Topic: ",m)
