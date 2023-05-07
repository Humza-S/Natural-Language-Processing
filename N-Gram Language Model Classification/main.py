import os
from nltk import word_tokenize
import copy
import math
import random
from nltk.stem import PorterStemmer
from scipy.optimize import basinhopping
import numpy as np
import pandas as pd



class n_grams:
    
    def __init__(self,fpath, n = 1, k = 0, unkfreq = 0.05) -> None:
        
        self._n = n
        self._k = k
        self._unkfreq = unkfreq
        self._review = []
        self._lowercase_review = []
        self._tokenized_review = []
        self._ngram = {}
        self._prior_ngram = {}
        self._ps = PorterStemmer()
        
        with open(fpath) as f:
            self._review = f.read().splitlines() 

        for review in self._review:
            self._lowercase_review.append(review.lower())
        
        self.tokenization()
        self.create_unigram()
        self.unkwown_word()
        self.create_unigram()
        self.train_ngrams()

    def tokenization(self):
        for review in self._lowercase_review:
            words = [ self._ps.stem(word) for word in word_tokenize(review)  ]
            #words = [ self._lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in words ]
            self._tokenized_review.append(['<s>']  * max(1, (self._n - 1)) + words + ['</s>'])
        return
    
    def create_unigram(self):
        self._total_word_count = 0
        self._unigram = {}
        for review in self._tokenized_review:
            for word in review:
                if word not in self._unigram:
                    self._unigram[word] = 1
                else:
                    self._unigram[word] += 1
                self._total_word_count += 1
        self._V = len(self._unigram)
        return
    
    def unkwown_word(self):
                        
        if self._unkfreq >= 1:
            self._unktarget = self._unkfreq
            self._unk_list = [i for i in self._unigram if (self._unigram[i] <= self._unktarget and i != "<s>" and i != "</s>")]
        #elif self._unkfreq > 0:
        #    self._unktarget = int(max(math.ceil(self._total_word_count * self._unkfreq), 1))
        else:
            #unigram_list = list(self._unigram.keys())
            #rand_list = np.random.uniform(0,1,len(unigram_list))
            #print(rand_list)
            #self._unk_list = [unigram_list for i in range(len(unigram_list)) if rand_list[i] < self._unkfreq]
            #self._unk_list = [i for i in self._unk_list if i != "<s>" and i != "</s>"]
            self._unk_list = [i for i in self._unigram if (random.uniform(0,1) < self._unkfreq and i != "<s>" and i != "</s>")]
            
        
        
        for review in self._tokenized_review:
            for index in range(len(review)):
                if review[index] in self._unk_list:
                    review[index] = "<UNK>"                  
        return 
    
    
    def train_ngrams(self):
        
        for review in self._tokenized_review:
            for i in range(0, len(review) - self._n + 1):
                ngram = " ".join(review[i:(i+self._n)])
                if ngram in self._ngram:
                    self._ngram[ngram] += 1
                else:
                    self._ngram[ngram] = 1
                    
        if self._n == 1:
            self._ngram_count_table = {' ': { j : self._ngram[j]  for j in self._unigram}}  

            
        elif self._n > 1:
            for review in self._tokenized_review:
                for i in range(0, len(review) - self._n + 2):
                    ngram = " ".join(review[i:(i+self._n - 1)])
                    if ngram in self._prior_ngram:
                        self._prior_ngram[ngram] += 1
                    else:
                        self._prior_ngram[ngram] = 1
                
            self._ngram_count_table = {i : { j : self._ngram[str(i + ' ' + j)] if str(i + ' ' + j) in self._ngram else 0 for j in self._unigram if j != '<s>'} for i in self._prior_ngram}   
            
        self._ngram_prob_table_raw = copy.deepcopy(self._ngram_count_table)
        self._ngram_prob_table_smoothed = copy.deepcopy(self._ngram_count_table)

        for i in self._ngram_prob_table_raw:
            for j in self._ngram_prob_table_raw[i]:
                if i != '</s>':
                    self._ngram_prob_table_raw[i][j] = self._ngram_count_table[i][j]  / (self._prior_ngram[i] if self._n > 1 else self._total_word_count)
                else:
                    self._ngram_prob_table_raw[i][j] = 0
    
            
        for i in self._ngram_prob_table_smoothed:
            for j in self._ngram_prob_table_smoothed[i]:
                if i != '</s>':
                    self._ngram_prob_table_smoothed[i][j] = (self._ngram_count_table[i][j] + self._k)  / ((self._prior_ngram[i] if self._n > 1 else self._total_word_count) + self._k * (self._V - 1)) #self._V - 1 because _V includes <s>
                else:
                    self._ngram_prob_table_smoothed[i][j] = 0
            
        return
    
    def likelihood(self, txt):
        #words = [word for word in word_tokenize(txt.lower())  if word not in self._stop_words ]
        words = [self._ps.stem(word) for word in word_tokenize(txt.lower()) ] 
        words = [ word if (word in self._unigram and word not in self._unk_list) else "<UNK>" for word in words ]
        tokenized_txt = ['<s>'] * max(1, (self._n - 1))  + words + ['</s>']
        prob = 0
        for i in range(len(tokenized_txt) - self._n + 1):
            if self._n > 1:
                prior_ngrams = " ".join(tokenized_txt[i:(i + self._n-1)])
            else:
                prior_ngrams = " "
                
            next_word = tokenized_txt[i + self._n - 1]
            
            if prior_ngrams in self._ngram_prob_table_smoothed and next_word in self._ngram_prob_table_smoothed[prior_ngrams]:
                prob += -math.log(self._ngram_prob_table_smoothed[prior_ngrams][next_word])
                
            else:
                prob += -math.log(self._k / (self._k * (self._V - 1) ))
           
            
           
                               
        return math.exp(prob/len(tokenized_txt))
    
    
    def get_ngram_count(self):
        return self._ngram_count_table
    
    def get_raw_ngram_prob(self):
        return self._ngram_prob_table_raw
    
    def get_smoothed_ngram_prob(self):
        return self._ngram_prob_table_smoothed
            

class optimize_model:
    
    def __init__(self, parms, data, target, unkfreq):
        self._parms = parms
        self._unkfreq = unkfreq
        self._data = data
        self._target = target
        
    def predict(self, model, data):
        return [model.likelihood(i) for i in data]
        
    
    def acc(self, parms):
        
        if parms[0] > parms[1]:
            return 1
        
        tmodel = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = parms[0],  unkfreq = self._unkfreq)     
        dmodel = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = parms[1],  unkfreq = self._unkfreq)
        
        dmodel_pp = self.predict(dmodel, self._data)
        tmodel_pp = self.predict(tmodel, self._data)
        return -sum([1 if (self._target[i] == '1' and tmodel_pp[i] > dmodel_pp[i]) or (self._target[i] == '0' and tmodel_pp[i] < dmodel_pp[i]) else 0  for i in range(len(self._target))]) / len(self._target)
        
    def optimize(self):
        bnds = ((0.25, 2), (0.25, 2))
        minimizer_kwargs = {"method":"L-BFGS-B", "bounds":bnds}
        return basinhopping(self.acc, self._parms, minimizer_kwargs=minimizer_kwargs, stepsize=0.1, niter=300, disp = True)          
   


class ensemble_lm:
    
    def __init__(self, parms, n) -> None:
        self._ensemble_lm = []
        self._n = n
        for i in range(n):
            print(f"Building {i} models")
            tmodel = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = parms[0],  unkfreq = parms[2])     
            dmodel = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = parms[1],  unkfreq = parms[3])
            self._ensemble_lm.append([tmodel, dmodel])
            
    def predict(self, txt):
        
        temp = []
        for m in self._ensemble_lm:
            t_pp = m[0].likelihood(txt)
            f_pp = m[1].likelihood(txt)
            
            if t_pp > f_pp:
                temp.append(1)
            else:
                temp.append(0)

        if sum(temp) > self._n / 2:
            return '1'
        else:
            return '0'
    
       
   





os.chdir(r"A1")


##
##
##  Unsmoothed Probabilities
##
##


# Unigram Word Probability Computation
tmodel_unigram = n_grams(r"A1_DATASET/train/truthful.txt", n = 1, k = 0,  unkfreq = 0)
dmodel_unigram = n_grams(r"A1_DATASET/train/deceptive.txt", n = 1, k = 0,  unkfreq = 0)

# Freq
print(tmodel_unigram.get_ngram_count())   
print(dmodel_unigram.get_ngram_count())   

#Prob
print(tmodel_unigram.get_raw_ngram_prob())   
print(dmodel_unigram.get_raw_ngram_prob())   




# Bigram Word Probability Computation
tmodel_bigram = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = 0,  unkfreq = 0)
dmodel_bigram = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = 0,  unkfreq = 0)

# Freq
truthful_bigram_cnt = pd.DataFrame.from_dict(tmodel_bigram.get_ngram_count(), orient='index')
truthful_bigram_cnt_top10 = truthful_bigram_cnt.head(10)

deceptive_bigram_cnt = pd.DataFrame.from_dict(dmodel_bigram.get_ngram_count(), orient='index')
deceptive_bigram_cnt_top10 = deceptive_bigram_cnt.head(10)



#Prob
truthful_bigram_prob = pd.DataFrame.from_dict(tmodel_bigram.get_raw_ngram_prob(), orient='index')
truthful_bigram_prob_top10 = truthful_bigram_prob.head(10)

deceptive_bigram_prob = pd.DataFrame.from_dict(dmodel_bigram.get_raw_ngram_prob(), orient='index')
deceptive_bigram_prob_top10 = deceptive_bigram_prob.head(10)




##
##
##  Smoothing - Laplace
##
##

tmodel_bigram = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = 1,  unkfreq = 0)
dmodel_bigram = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = 1,  unkfreq = 0)

#Smoothed Probabilities
truthful_bigram_prob = pd.DataFrame.from_dict(tmodel_bigram.get_smoothed_ngram_prob(), orient='index')
truthful_bigram_prob_top10 = truthful_bigram_prob.head(10)

deceptive_bigram_prob = pd.DataFrame.from_dict(dmodel_bigram.get_smoothed_ngram_prob(), orient='index')
deceptive_bigram_prob_top10 = deceptive_bigram_prob.head(10)



##
##
##  Smoothing - Add K
##
##
tmodel_bigram = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = 1.033,  unkfreq = 0)
dmodel_bigram = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = 1.460,  unkfreq = 0)

#Smoothed Probabilities
truthful_bigram_prob = pd.DataFrame.from_dict(tmodel_bigram.get_smoothed_ngram_prob(), orient='index')
truthful_bigram_prob_top10 = truthful_bigram_prob.head(10)

deceptive_bigram_prob = pd.DataFrame.from_dict(dmodel_bigram.get_smoothed_ngram_prob(), orient='index')
deceptive_bigram_prob_top10 = deceptive_bigram_prob.head(10)





##
##
##  Perplexity - Add 1
##
##


tmodel_bigram = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = 1,  unkfreq = 1)
dmodel_bigram = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = 1,  unkfreq = 1)

   
with open("A1_DATASET/validation/truthful.txt") as f:
    treviews = f.read().splitlines() 

with open("A1_DATASET/validation/deceptive.txt") as f:
    dreviews = f.read().splitlines() 



perplexity_validation = []
for review in treviews:
    perplexity_validation.append([0, tmodel_bigram.likelihood(review), dmodel_bigram.likelihood(review)])
    
for review in dreviews:
    perplexity_validation.append([1, tmodel_bigram.likelihood(review), dmodel_bigram.likelihood(review)])


perplexity_validation_df = pd.DataFrame(perplexity_validation, columns = ["Type", "Tmodel", "Fmodel"])



##
##
##  Base Models
##
##


with open("A1_DATASET/train/truthful.txt") as f:
    treviews_t = f.read().splitlines()
    
with open("A1_DATASET/train/deceptive.txt") as f:
    dreviews_t = f.read().splitlines() 

with open("A1_DATASET/validation/truthful.txt") as f:
    treviews_v = f.read().splitlines() 

with open("A1_DATASET/validation/deceptive.txt") as f:
    dreviews_v = f.read().splitlines() 
    
with open("A1_DATASET/test/test.txt") as f:
    test = f.read().splitlines() 

with open("A1_DATASET/test/test_labels.txt") as f:
    test_labels = f.read().splitlines() 
    
result = []

for n in [1, 2]:
    
    for k in [[0.1, 0.1],[0.1,0.3],[0.1, 0.5],[1, 1],[1, 1.3],[1, 1.5]]:
        
        for unkfreq in [0.005, 0.0075, 0.01, 1, 2, 3]:
            
            tmodel_ngram = n_grams(r"A1_DATASET/train/truthful.txt", n = n, k = k[0],  unkfreq = unkfreq)
            dmodel_ngram = n_grams(r"A1_DATASET/train/deceptive.txt", n = n, k = k[1],  unkfreq = unkfreq)
               
            
            ngram_train_result_t = []
            
            t_pp_total_train_t = []
            d_pp_total_train_t = []
            
            for txt in treviews_t:
                t_pp = tmodel_ngram.likelihood(txt)
                d_pp = dmodel_ngram.likelihood(txt)
                
                t_pp_total_train_t.append(t_pp)
                d_pp_total_train_t.append(d_pp)
                
                if t_pp <= d_pp:
                    ngram_train_result_t.append(1)
                else:
                    ngram_train_result_t.append(0)
    
      
            ngram_train_result_d = []
            
            t_pp_total_train_d = []
            d_pp_total_train_d = []
            
            for txt in dreviews_t:
                t_pp = tmodel_ngram.likelihood(txt)
                d_pp = dmodel_ngram.likelihood(txt)
                
                t_pp_total_train_d.append(t_pp)
                d_pp_total_train_d.append(d_pp)
                
                if t_pp > d_pp:
                    ngram_train_result_d.append(1)
                else:
                    ngram_train_result_d.append(0)
          
    
     
            ngram_dev_result_t = []
            
            t_pp_total_dev_t = []
            d_pp_total_dev_t = []
            
            for txt in treviews_v:
                t_pp = tmodel_ngram.likelihood(txt)
                d_pp = dmodel_ngram.likelihood(txt)
                
                t_pp_total_dev_t.append(t_pp)
                d_pp_total_dev_t.append(d_pp)
                
                if t_pp <= d_pp:
                    ngram_dev_result_t.append(1)
                else:
                    ngram_dev_result_t.append(0)
    
      
            ngram_dev_result_d = []
            
            t_pp_total_dev_d = []
            d_pp_total_dev_d = []
            
            for txt in dreviews_v:
                t_pp = tmodel_ngram.likelihood(txt)
                d_pp = dmodel_ngram.likelihood(txt)
                
                t_pp_total_dev_d.append(t_pp)
                d_pp_total_dev_d.append(d_pp)
                
                if t_pp > d_pp:
                    ngram_dev_result_d.append(1)
                else:
                    ngram_dev_result_d.append(0)
                    
                    
            ngram_test_result = []
            
            t_pp_total_test_d = []
            d_pp_total_test_d = []
            
            for i in range(len(test)):
                t_pp = tmodel_ngram.likelihood(test[i])
                d_pp = dmodel_ngram.likelihood(test[i])
                
                if test_labels[i] == '0':
                    t_pp_total_test_d.append(t_pp)
                else:
                    d_pp_total_test_d.append(d_pp)
                
                if (t_pp > d_pp and test_labels[i] == '1') or (t_pp <= d_pp and test_labels[i] == '0'):
                    ngram_test_result.append(1)
                else:
                    ngram_test_result.append(0)
                    
            trainingacc = np.mean(ngram_train_result_t+ngram_train_result_d)
            trainingacc_t = np.mean(ngram_train_result_t)
            trainingacc_d = np.mean(ngram_train_result_d)
            validationacc = np.mean(ngram_dev_result_t+ngram_dev_result_d)
            validationacc_t = np.mean(ngram_dev_result_t)
            validationacc_d = np.mean(ngram_dev_result_d)
            testingacc = np.mean(ngram_test_result)
        
            
            
            print(n, k, unkfreq, trainingacc,trainingacc_t, trainingacc_d, validationacc, validationacc_t, validationacc_d, testingacc)
       
            result.append([n, k, unkfreq, trainingacc,trainingacc_t, trainingacc_d , validationacc, validationacc_t, validationacc_d, testingacc ])
            
            


result_df = pd.DataFrame(result, columns = ['n','K','UNK','Training Acc','Training Acc T','Training Acc D', 'Validation Acc','Validation Acc T', 'Validation Acc D','Testing Acc'])




##
##
##  Optimization
##
##

with open("A1_DATASET/validation/truthful.txt") as f:
    treviews = f.read().splitlines() 

with open("A1_DATASET/validation/deceptive.txt") as f:
    dreviews = f.read().splitlines() 

data = treviews + dreviews
target = ['0' for _ in range(len(treviews))] + ['1' for _ in range(len(dreviews))]

print(optimize_model([1.06, 1.460], data, target, 0.0075).optimize())
print(optimize_model([1.0, 1.38], data, target, 2).optimize())


##
##
##  Optimized Model
##
##


result_2 = []

for unkfreq in [0.005,0.0075,0.01]:
    
    tmodel_ngram = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = 1.06,  unkfreq = unkfreq)
    dmodel_ngram = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = 1.46,  unkfreq = unkfreq)
       
    ngram_dev_result_t = []
    
    t_pp_total_dev_t = []
    d_pp_total_dev_t = []
    
    for txt in treviews_v:
        t_pp = tmodel_ngram.likelihood(txt)
        d_pp = dmodel_ngram.likelihood(txt)
        
        t_pp_total_dev_t.append(t_pp)
        d_pp_total_dev_t.append(d_pp)
        
        if t_pp <= d_pp:
            ngram_dev_result_t.append(1)
        else:
            ngram_dev_result_t.append(0)
    
    
    ngram_dev_result_d = []
    
    t_pp_total_dev_d = []
    d_pp_total_dev_d = []
    
    for txt in dreviews_v:
        t_pp = tmodel_ngram.likelihood(txt)
        d_pp = dmodel_ngram.likelihood(txt)
        
        t_pp_total_dev_d.append(t_pp)
        d_pp_total_dev_d.append(d_pp)
        
        if t_pp > d_pp:
            ngram_dev_result_d.append(1)
        else:
            ngram_dev_result_d.append(0)
    
    
    validationacc = np.mean(ngram_dev_result_t+ngram_dev_result_d)
    validationacc_t = np.mean(ngram_dev_result_t)
    validationacc_d = np.mean(ngram_dev_result_d)
    
    result_2.append([unkfreq, validationacc, validationacc_t, validationacc_d])
    
    
for unkfreq in [1,2,3]:
    
    tmodel_ngram = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = 1.00,  unkfreq = unkfreq)
    dmodel_ngram = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = 1.38,  unkfreq = unkfreq)
       
    ngram_dev_result_t = []
    
    t_pp_total_dev_t = []
    d_pp_total_dev_t = []
    
    for txt in treviews_v:
        t_pp = tmodel_ngram.likelihood(txt)
        d_pp = dmodel_ngram.likelihood(txt)
        
        t_pp_total_dev_t.append(t_pp)
        d_pp_total_dev_t.append(d_pp)
        
        if t_pp <= d_pp:
            ngram_dev_result_t.append(1)
        else:
            ngram_dev_result_t.append(0)
    
    
    ngram_dev_result_d = []
    
    t_pp_total_dev_d = []
    d_pp_total_dev_d = []
    
    for txt in dreviews_v:
        t_pp = tmodel_ngram.likelihood(txt)
        d_pp = dmodel_ngram.likelihood(txt)
        
        t_pp_total_dev_d.append(t_pp)
        d_pp_total_dev_d.append(d_pp)
        
        if t_pp > d_pp:
            ngram_dev_result_d.append(1)
        else:
            ngram_dev_result_d.append(0)
    
    
    validationacc = np.mean(ngram_dev_result_t+ngram_dev_result_d)
    validationacc_t = np.mean(ngram_dev_result_t)
    validationacc_d = np.mean(ngram_dev_result_d)
    
    result_2.append([unkfreq, validationacc, validationacc_t, validationacc_d])


result_2_df = pd.DataFrame(result_2, columns = ['UNK','Validation Acc','Validation Acc T', 'Validation Acc D'])



#
#
#   Emsemble Model using classes
#
#



en_bigram = ensemble_lm([1.06,1.46,0.0075,0.0075], 25)

random.seed(101)
with open("A1_DATASET/test/test.txt") as f:
    tests = f.read().splitlines() 
    
with open("A1_DATASET/test/test_labels.txt") as f:
    test_labels = f.read().splitlines() 
    
cnt = 0
for i in range(len(tests)):
    if en_bigram.predict(tests[i]) == test_labels[i]:
        cnt += 1
        
print(cnt / len(tests))



#
#
#  Ensemble model with 50 sets of bigrams - For error analysis
#
#


ensemble_metric_t = [[0,0] for _ in range(70)]
ensemble_metric_f = [[0,0] for _ in range(70)]
test_result = [[0, 0] for _ in range(len(test))]


with open("A1_DATASET/validation/truthful.txt") as f:
    treviews = f.read().splitlines() 
    
with open("A1_DATASET/validation/deceptive.txt") as f:
    dreviews = f.read().splitlines() 
    
with open("A1_DATASET/test/test.txt") as f:
    test = f.read().splitlines() 

with open("A1_DATASET/test/test_labels.txt") as f:
    test_labels = f.read().splitlines() 


for _ in range(50):
    
    tmodel = n_grams(r"A1_DATASET/train/truthful.txt", n = 2, k = 1.06,  unkfreq = 0.0075)
    dmodel = n_grams(r"A1_DATASET/train/deceptive.txt", n = 2, k = 1.46,  unkfreq = 0.0075)
    cnt = 0
    tcnt = 0
    for treview in treviews:
        t_pp = tmodel.likelihood(treview)
        d_pp = dmodel.likelihood(treview)
        if t_pp < d_pp:
            tcnt += 1
            ensemble_metric_t[cnt][0] += 1
        else:
            ensemble_metric_t[cnt][1] += 1
            
        cnt += 1
        
    dcnt = 0
    cnt = 0
    for treview in dreviews:
        t_pp = tmodel.likelihood(treview)
        d_pp = dmodel.likelihood(treview)
        if t_pp > d_pp:
            dcnt += 1
            ensemble_metric_f[cnt][1] += 1
        else:
            ensemble_metric_f[cnt][0] += 1        
        cnt += 1
        
        
    acc_cnt = 0
    cnt = 0
    for i in range(len(test)):
        t_pp = tmodel.likelihood(test[i])
        d_pp = dmodel.likelihood(test[i])   
        if t_pp > d_pp:
            test_result[cnt][1] += 1
        else:
            test_result[cnt][0] += 1   
            
        if (t_pp > d_pp and test_labels[i] == '1') or (t_pp <= d_pp and test_labels[i] == '0'):
            acc_cnt += 1
            
        cnt += 1        
        
    
testing_df = pd.DataFrame([[test_labels[i]] + test_result[i]  for i in range(len(test_labels))], columns = ['Test_Labels', 'Truthful_Votes', 'Deceptive_Votes'])
truthful_validation_df = pd.DataFrame(ensemble_metric_t, columns = ['Truthful_Votes', 'Deceptive_Votes'])
deceptive_validation_df = pd.DataFrame(ensemble_metric_f, columns = ['Truthful_Votes', 'Deceptive_Votes'])



