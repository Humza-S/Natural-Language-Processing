import pandas as pd
import copy
import math
import random
import os
""" Contains the part of speech tagger class. """

def load_data(sentence_file, tag_file=None):
    
    if tag_file is not None:
        sentence_df = pd.read_csv(sentence_file)
        tag_df = pd.read_csv(tag_file)
        final_df = sentence_df.merge(tag_df, how = "inner")
        final_list = []
        tmp = []
        for i in range(len(final_df)):
            if final_df['word'][i] != '-DOCSTART-' :
                tmp.append([final_df['word'][i], final_df['tag'][i]])
            elif final_df['word'][i] == '-DOCSTART-' and len(tmp) == 0:
                tmp = [['-DOCSTART-', 'O']]
            elif len(tmp) > 0:
                tmp.append(['-DOCEND-', 'END'])
                final_list.append(tmp)
                tmp = [['-DOCSTART-', 'O']]
        tmp.append(['-DOCEND-', 'END'])
        final_list.append(tmp)             
        return final_list
    
class POSTagger():
    def __init__(self, n):
        """Initializes the tagger model parameters and anything else necessary. """
        self._n = n
        self._pos_unigram_cnt = {}
        self._emission_unigram_cnt = {}
        self._prior_ngram_cnt = {}
        self._ngram_cnt = {}
        
    def train(self, data, unk = 0.0075):
        
        #Create Word List
        word_list = []
        for sen in data:
            for word in sen:
                word_list.append(word[0])
                
        self._word_list = set(word_list)
        
        self._unk_list = [i for i in self._word_list if (random.uniform(0,1) < unk and i != "-DOCSTART-" and i != "-DOCEND-")]
        
        word_list = ['<UNK>']
        
        for sen in data:
            for word in sen:
                if word[0] in self._unk_list:
                    word[0] = '<UNK>'
                else:
                    word_list.append(word[0])
                    
        self._word_list = set(word_list)        
        
        
        word_list = []
        #Create Unigram - it is used in any n-grams
        for sen in data:
            for word in sen:
                
                # word[0]: word, word[1]: POS tag
                word_list.append(word[0])
                
                # POS Tag Unigram
                if word[1] not in self._pos_unigram_cnt:
                    self._pos_unigram_cnt[word[1]] = 1
                else:
                    self._pos_unigram_cnt[word[1]] += 1
                    
                # Emission Unigram
                if word[1] not in self._emission_unigram_cnt:
                    self._emission_unigram_cnt[word[1]] = {i : 0 for i in self._word_list}
                
                if word[0] not in self._emission_unigram_cnt[word[1]]:
                    self._emission_unigram_cnt[word[1]][word[0]] = 1
                else:
                    self._emission_unigram_cnt[word[1]][word[0]] += 1
                    
        
                    
        if self._n == 1:
            self._total_tag_cnt = sum([self._pos_unigram_cnt[i] for i in self._pos_unigram_cnt])
            self._prior_ngram_cnt = {i: self._total_tag_cnt for i in self._pos_unigram_cnt}
            self._ngram_cnt = {i : {j : self._pos_unigram_cnt[j] for j in self._pos_unigram_cnt} for i in self._pos_unigram_cnt}
            
        else:
            # Create n-1 gram and ngram
            for sen in data:
                
                sen = [('-DOCSTART-', 'O') for _ in  range(self._n - 2)] + sen
                
                for i in range(0, len(sen)- self._n + 1):
                    
                    # Create n-1 gram
                    prior_ngram = " ".join([ngram[1] for ngram in sen[i:(i+self._n-1)]])
                    if prior_ngram not in self._prior_ngram_cnt:
                        self._prior_ngram_cnt[prior_ngram] = 1
                    else:
                        self._prior_ngram_cnt[prior_ngram] += 1
                        
                    # Create ngram
                    if prior_ngram not in self._ngram_cnt:
                        self._ngram_cnt[prior_ngram] = {j : 0 for j in self._pos_unigram_cnt}
                        
                    if sen[i+self._n-1][1] not in self._ngram_cnt[prior_ngram]:
                        self._ngram_cnt[prior_ngram][sen[i+self._n-1][1]] = 1
                    else:
                        self._ngram_cnt[prior_ngram][sen[i+self._n-1][1]] += 1


    def smoothing_k(self, k1, k2):
        # Transform n-gram count to probability
        self._k1 = k1
        self._k2 = k2
        
        self._ngram_prob = copy.deepcopy(self._ngram_cnt)
        self._emission_prob = copy.deepcopy(self._emission_unigram_cnt)

        for i in self._prior_ngram_cnt:
            for j in self._ngram_prob[i]:
                self._ngram_prob[i][j] = (self._ngram_prob[i][j] + k1) / (self._prior_ngram_cnt[i] + k1 * len(self._pos_unigram_cnt))
        
        for i in self._emission_prob:
            for j in self._emission_prob[i]:
                self._emission_prob[i][j] = (self._emission_prob[i][j] + k2) / (self._pos_unigram_cnt[i] + k2 * len(self._word_list))
                
    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        sequence = ['-DOCSTART-' for _ in  range(self._n - 2)] + sequence + ['-DOCEND-']
        tags = ['O' for _ in  range(self._n - 2)] + tags + ['END']
        
        prob = 1
        for t in range(len(sequence)-self._n + 1):
            prior_ngram = " ".join(tags[t:(t + self._n - 1)])
            prob = prob * self._ngram_prob[prior_ngram][tags[t+self._n-1]] * self._emission_prob[tags[t+self._n-1]][sequence[t+self._n-1]]
        
        return prob
        
 
    def inference(self, sequence):
        
        # Vit_Matrix 
        # Dictionary Key : POS Tag 
        # Each Value in Dictionary is a list - each element is the list is also a list, representing the status at position t
        # List is consisted of tuple - 1st : Prob 2nd : Prior Tag
        Vit_Matrix = {i: [ [float("-inf"), None] for _ in range(len(sequence) + max(0, self._n - 2)) ]   for i in self._pos_unigram_cnt}

        sequence = ['O' for _ in  range(self._n - 2)] + [ word if (word in self._word_list and word not in self._unk_list) else "<UNK>"  for word in sequence]
        
        for t in range(max(1, self._n - 1)):
            Vit_Matrix['O'][t][0] = 0
            Vit_Matrix['O'][t][1] = 'O'
        
        for t in range(self._n - 2, len(sequence) - 1): # Check each position in the sequence
            
            for tag in self._pos_unigram_cnt: #For each position in the sequence, check if there is any non-zero prob for each pos tag 
                
                if Vit_Matrix[tag][t][0] != float("-inf"): 
                    prior_ngram = [tag]
                    tmp_tag = Vit_Matrix[tag][t][1]
                    
                    for tag_reverse in range(t-1, t - self._n + 1, -1):
                        prior_ngram = [tmp_tag] + prior_ngram
                        tmp_tag = Vit_Matrix[tmp_tag][tag_reverse][1]
                 
                    prior_ngram = " ".join(prior_ngram) 
                    #print(prior_ngram)     
                    
                            
                    if prior_ngram in self._ngram_prob:       
                        for next_tag in self._ngram_prob[prior_ngram]:
                            
                            if sequence[t+1] in self._emission_prob[next_tag]:
                                Emission_Prob = self._emission_prob[next_tag][sequence[t+1]]
                            else:
                                Emission_Prob = self._k2 / (self._pos_unigram_cnt[next_tag] + self._k2 * len(self._word_list))
                                
                            Next_Prob = Vit_Matrix[tag][t][0] + math.log(self._ngram_prob[prior_ngram][next_tag]) + math.log10(Emission_Prob)
                            if Next_Prob > Vit_Matrix[next_tag][t+1][0]:
                                Vit_Matrix[next_tag][t+1][0] = Next_Prob
                                Vit_Matrix[next_tag][t+1][1] = tag
                            

        #Trace Back the best route
        cur_max = float("-inf")
        cur_tag = None
        for tag in Vit_Matrix:
            if Vit_Matrix[tag][len(sequence)-1][0] > cur_max:
                cur_max = Vit_Matrix[tag][len(sequence)-1][0]
                cur_tag = tag
                
        result = [cur_tag]
        for t in range(len(sequence)-1, 0, -1):
            result = [Vit_Matrix[cur_tag][t][1]] + result
            cur_tag = Vit_Matrix[cur_tag][t][1]
            
        return result[self._n - 2 :-1]                            
                    
         



train_data = load_data("data/train_x.csv", "data/train_y.csv")
pos_tagger = POSTagger(n = 3)
pos_tagger.train(train_data, unk = 0.01)
pos_tagger.smoothing_k(0.001, 0.2)

'''
#test_data = load_data("data/trial_x.csv", "data/trial_y.csv")
test_data = load_data("data/dev_x.csv", "data/dev_y.csv")
print(len(test_data))


cnt = 0
predicted = []
observed = []

for i in test_data:
    seq = [word[0] for word in i]
    print(cnt, len(seq))
    predicted += pos_tagger.inference(seq)
    observed += seq[:-1]
    cnt += 1
    



final = [ [predicted[i], observed[i]] for i in range(len(predicted))]
predicted = pd.DataFrame(final).reset_index()
predicted.columns = ['id','tag','word']
predicted.to_csv("prediction.csv", index=False)

'''