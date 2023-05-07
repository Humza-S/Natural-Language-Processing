import pandas as pd
import copy
import math
""" Contains the part of speech tagger class. """

def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    
    if tag_file is not None:
        sentence_df = pd.read_csv(sentence_file)
        tag_df = pd.read_csv(tag_file)
        final_df = sentence_df.merge(tag_df, how = "inner")
        final_list = []
        tmp = []
        for i in range(len(final_df)):
            if final_df['word'][i] != '-DOCSTART-' :
                tmp.append((final_df['word'][i], final_df['tag'][i]))
            elif final_df['word'][i] == '-DOCSTART-' and len(tmp) == 0:
                tmp = [('-DOCSTART-', 'O')]
            elif len(tmp) > 0:
                tmp.append(('-DOCEND-', 'END'))
                final_list.append(tmp)
                tmp = [('-DOCSTART-', 'O')]
        tmp.append(('-DOCEND-', 'END'))
        final_list.append(tmp)             
        return final_list
  

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy

    You might want to refactor this into several different evaluation functions.
    
    """
    pass

class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self._n = 1
        self._unigram_cnt = {}
        self._bigram_cnt = {}
        self._emission_cnt = {}

        pass

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        for sen in data:
            for i in range(0, len(sen)):
                if sen[i][1] not in self._unigram_cnt:
                    self._unigram_cnt[sen[i][1]] = 1
                else:
                    self._unigram_cnt[sen[i][1]] += 1
 
                if sen[i][1] not in self._emission_cnt:
                    self._emission_cnt[sen[i][1]] = {}
                
                if sen[i][0] not in self._emission_cnt[sen[i][1]]:
                    self._emission_cnt[sen[i][1]][sen[i][0]] = 1
                else:
                    self._emission_cnt[sen[i][1]][sen[i][0]] += 1
                    
        self._emission_prob = copy.deepcopy(self._emission_cnt)

        for i in self._emission_prob:
            for j in self._emission_prob[i]:
                self._emission_prob[i][j] = self._emission_prob[i][j] / self._unigram_cnt[i]
               
        
        for sen in data:
            for i in range(0, len(sen)-1):
                if sen[i][1] not in self._bigram_cnt:
                    self._bigram_cnt[sen[i][1]] = {}
                  
                if sen[i+1][1] not in self._bigram_cnt[sen[i][1]]:
                    self._bigram_cnt[sen[i][1]][sen[i+1][1]] = 1
                else:
                    self._bigram_cnt[sen[i][1]][sen[i+1][1]] += 1                    
        
        self._bigram_prob = copy.deepcopy(self._bigram_cnt)
        
        for i in self._bigram_prob:
            for j in self._bigram_prob[i]:
                self._bigram_prob[i][j] = self._bigram_cnt[i][j] / self._unigram_cnt[i]
        
        
        
        return

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        return 0.

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.
        """
        Vit_Matrix = {i: [ [float("-inf"), None] for _ in range(len(sequence)) ]   for i in self._unigram_cnt}
        
        Vit_Matrix['O'][0][0] = 0        
        for t in range(len(sequence)-1):
            for tag in Vit_Matrix:
                if Vit_Matrix[tag][t][0] != float("-inf"):
                    for next_tag in self._bigram_prob[tag]:
                        if sequence[t+1] in self._emission_prob[next_tag] and Vit_Matrix[tag][t][0] +  math.log(self._bigram_prob[tag][next_tag]) + math.log(self._emission_prob[next_tag][sequence[t+1]]) >= Vit_Matrix[next_tag][t+1][0]:
                            Vit_Matrix[next_tag][t+1][0] = Vit_Matrix[tag][t][0] +  math.log(self._bigram_prob[tag][next_tag]) + math.log(self._emission_prob[next_tag][sequence[t+1]]) 
                            Vit_Matrix[next_tag][t+1][1] = tag
                            
        cur_max = float("-inf")
        cur_tag = None
        for tag in Vit_Matrix:
            if Vit_Matrix[tag][len(sequence)-1][0] > cur_max:
                cur_max = Vit_Matrix[tag][len(sequence)-1][0]
                cur_tag = tag
                
        result = [cur_tag]
        #print(Vit_Matrix[cur_tag][len(sequence)-1])
        for t in range(len(sequence)-1, 0, -1):
            result = [Vit_Matrix[cur_tag][t][1]] + result
            cur_tag = Vit_Matrix[cur_tag][t][1]
            
            
        return result[:-1]
        


if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/trial_x.csv", "data/trial_y.csv")
    pos_tagger.train(train_data)
    predicted_trial = ['-DOCSTART-','Pierre','Vinken',',','61','years','old',',','will','join','the','board','as','a','nonexecutive','director','Nov.','29','.','Mr.','Vinken','is','chairman','of','Elsevier','N.V.',',','the','Dutch','publishing','group','.','-DOCEND-']
    predicted_trial = ['-DOCSTART-','Pierre','Vinken',',','61','years','old',',','will','join','the','board','as','a','nonexecutive','director','Nov.','29','.','Mr.','Vinken','is','chairman','of','Elsevier','N.V.',',','the','Dutch','publishing','group','.','-DOCEND-']
    x = pos_tagger.inference(predicted_trial)
    
    print(x)