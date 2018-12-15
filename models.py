import numpy as np
import pandas as pd
import io
import re
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import argparse
parser = argparse.ArgumentParser(description='very deep convolutional neural networks for text classification')

parser.add_argument('--model_name', type=str, default='content-words',
					help='type of bag-of-words model to use')

def create_dataset(data, chapter, author):
    tempdata = data[data.chapter != chapter]
    gps = tempdata.groupby(['Character']).groups
    minnum = []
    for key, item in gps.items():
        minnum.append(len(gps[key]))
    minnum = min(minnum)
    samplelist = []
    for group in gps.items():
        #samplelist.extend(random.sample(list(group[1]), minnum))
        #lenlist = len(list(group[1]))
        #samplelist.extend(list(group[1])[lenlist-minnum:])
        samplelist.extend(list(group[1])[0:minnum])
    return samplelist
def main():
	data = pd.read_csv('proc_data.csv', encoding = "ISO-8859-1")
	stops = stopwords.words('french')
	cat_map = {'_cécile volanges': 'CV', '_la marquise de merteuil': 'MM', '_le vicomte de valmont': 'VV',
           	'_la présidente de tourvel':'PT','_madame de volanges':'MV','_le chevalier danceny':'CD','_madame de rosemonde':'MR'}
	args = parser.parse_args()
	if args.model_name == 'basic':
		text_clf = Pipeline([('vect', CountVectorizer()),
	                         ('clf', LinearSVC())])
	if args.model_name == 'content-words':
	    text_clf = Pipeline([('vect', CountVectorizer(stop_words = stops)),
	                         ('tfidf', TfidfTransformer()),
	                         ('clf', LinearSVC())])
	if args.model_name == 'topk':
	    text_clf = Pipeline([('vect', CountVectorizer(stop_words = stops, max_features = 1000)),
	                         ('tfidf', TfidfTransformer()),
	                         ('clf', LinearSVC())])
	if args.model_name == 'char_3':
	    text_clf = Pipeline([('vect', CountVectorizer(analyzer = 'char', ngram_range = (3,3))),
	                         ('clf', LinearSVC())])
	if args.model_name == 'char_5':
	    text_clf = Pipeline([('vect', CountVectorizer(analyzer = 'char', ngram_range = (5,5))),
	                         ('clf', LinearSVC())])
	if args.model_name == '2_gram':
	    text_clf = Pipeline([('vect', CountVectorizer(ngram_range = (2,2))),
	    					 ('tfidf', TfidfTransformer()),
	                         ('clf', LinearSVC())])
	accuracy_arr = {k:0 for k in cat_map.values()}
	tot_vals = {k:0 for k in cat_map.values()}
	#clf =  LinearSVC(random_state=0, tol = 0.00001)
	pred_list = []
	 
	for i in range(len(data)):

	    sampleid = create_dataset(data, data['chapter'][i], data['Character'][i] )
	    datatemp = data.iloc[sampleid]
	    train_data_features = text_clf.fit(datatemp.chapter, datatemp.Character)
	    pred = text_clf.predict([data['chapter'][i]])[0]
	    pred_list.append(pred)
	    if pred == data['Character'][i]:
	        accuracy_arr[pred] += 1
	        tot_vals[pred] += 1
	    else:
	        tot_vals[data['Character'][i]] +=  1
	accuracy_arr = {i: accuracy_arr[i]/tot_vals[i] for i in accuracy_arr.keys()}
	print(accuracy_arr)
	print('Average Accuracy: {0}'.format(sum(accuracy_arr.values())/7))
	from sklearn.metrics import confusion_matrix
	print(list(cat_map.values()))
	print(confusion_matrix(list(data.Character), pred_list, labels = list(cat_map.values())))
if __name__ == "__main__":
	main()