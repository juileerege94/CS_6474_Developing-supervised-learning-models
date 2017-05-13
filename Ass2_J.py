# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 15:32:47 2016

@author: Juilee Rege
"""

import json
import os
import pandas as pd
import numpy as np
import re
import random
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

path = os.getcwd()

def match(line,w):
    
    result = []
    my_regex = r"(^" + re.escape(w) + r")"
    for s in line.split(" "):
        result += re.compile(my_regex,re.IGNORECASE).findall(s)
    res = len(result)
    return res
    
def assignment():

    with open(path+'\data\pizza_request_dataset.json') as json_data:
        d = json.load(json_data)

    list_random = random.sample(range(5671),567)
    
    with open(path+'\post_train.txt', 'w') as outfile:
        with open(path+'\post_test.txt', 'w') as outfile1:
            with open(path+'\\target_train.txt', 'w') as outfile2:
                with open(path+'\\target_test.txt', 'w') as outfile3:
                    for i in range(0,5671):
                        if(i in list_random):
                            json.dump(d[i]["request_text"], outfile1, indent=4)
                            outfile1.write('\n')
                            json.dump(d[i]["requester_received_pizza"], outfile3, indent=4)
                            outfile3.write('\n')
                        else:
                            json.dump(d[i]["request_text"], outfile, indent=4)
                            outfile.write('\n')
                            json.dump(d[i]["requester_received_pizza"], outfile2, indent=4)
                            outfile2.write('\n')
                        
    #pre-processing the posts file and storing to post.txt
    mydata = pd.read_csv(path + '/post_train.txt',header=None,sep='delimiter',engine='python')
    X_train=[]
    for i in range(0,len(mydata)):
        mydata[0][i] = mydata[0][i].lower()
        mydata[0][i]= mydata[0][i].replace('\\n',' ')
        mydata[0][i]= mydata[0][i].replace('"','')
        w=mydata[0][i].split(' ')
        words=[]
        for j in range(0,len(w)):
            words.append(w[j])
        s=""
        s = " ".join(str(x) for x in words)
        X_train.append(s)
    
    mydata = pd.read_csv(path + '/post_test.txt',header=None,sep='delimiter',engine='python')
    X_test=[]
    for i in range(0,len(mydata)):
        mydata[0][i] = mydata[0][i].lower()
        mydata[0][i]= mydata[0][i].replace('\\n',' ')
        mydata[0][i]= mydata[0][i].replace('"','')
        w=mydata[0][i].split(' ')
        words=[]
        for j in range(0,len(w)):
            words.append(w[j])
        s=""
        s = " ".join(str(x) for x in words)
        X_test.append(s)

    y_train = pd.read_csv(path + '/target_train.txt',header=None,sep='delimiter',engine='python')
    y_test = pd.read_csv(path + '/target_test.txt',header=None,sep='delimiter',engine='python')        
    print len(X_train)
    print len(X_test)
    print len(y_train)
    print len(y_test)    
    
####################################################first model###########################################
    
    print "Running part 1"
    my_words = ['just','im']
    sw = text.ENGLISH_STOP_WORDS.union(my_words)
    vectorizer = CountVectorizer(input='content',analyzer=u'word',tokenizer=None,ngram_range=(1,1), stop_words=sw, lowercase=True)
    vectorizer1 = CountVectorizer(input='content',analyzer=u'word',tokenizer=None,ngram_range=(2,2), stop_words=sw, lowercase=True)
    
    X = vectorizer.fit_transform(X_train)
    X1 = vectorizer1.fit_transform(X_train)
    word_freq_df = pd.DataFrame({'term':vectorizer.get_feature_names(),'occurences':pd.np.array(X.sum(axis=0)).ravel().tolist()})
    word_freq_df1 = pd.DataFrame({'term':vectorizer1.get_feature_names(),'occurences':pd.np.array(X1.sum(axis=0)).ravel().tolist()})
    vocab = word_freq_df.sort("occurences",ascending=False).head(500)
    vocab1 = word_freq_df1.sort("occurences",ascending=False).head(500)
    vocab.to_csv(path + "/TopFrequencyUnigramWords.txt", index=False, cols=('term','occurrences'),encoding='utf-8')
    vocab1.to_csv(path + "/TopFrequencyBigramWords.txt", index=False, cols=('term','occurrences'),encoding='utf-8')
    
    v = vocab['term'].tolist()
    v1=vocab1['term'].tolist()
    v = v + v1
    vectorizer = CountVectorizer(input='content',vocabulary = v,analyzer=u'word',tokenizer=None, stop_words=sw, lowercase=True)
    input11 = vectorizer.transform(X_train).toarray()
    input2 = vectorizer.transform(X_test).toarray()

    clf = svm.LinearSVC()
    clf.fit(input11,y_train)
    answer1 = clf.predict(input2)
    p1 = precision_score(y_test, answer1)
    r1 = recall_score(y_test, answer1)
    f1 = f1_score(y_test, answer1)
    a1 = accuracy_score(y_test, answer1)
    mat = confusion_matrix(y_test, answer1)
    sp = float(mat[0][0])/float(mat[0][0]+mat[0][1])
    auc1 = roc_auc_score(y_test, answer1)
    print "accuracy, precision, recall, f1, specificity, auc for first model: ", a1, p1, r1, f1, sp, auc1

###################################################second model###########################################

############       making a dictionary and storing into list.txt..DO NOT DELETE LIST.TXT###

#    with open(path+'\part3.txt', 'w') as outfile:
#        for i in range(0,5671):
#            for j in range(0,len(d[i]["requester_subreddits_at_request"])):
#                s = d[i]["requester_subreddits_at_request"][j]
#                json.dump(s, outfile, indent=4)
#                outfile.write('\n')
#    
#    dict1 = {}
#    mydata1 = pd.read_csv(path + '/part3.txt',header=None,sep='delimiter',engine='python')
#    for i in range(0,len(mydata1)):
#        mydata1[0][i]= mydata1[0][i].replace('\\n',' ')
#        mydata1[0][i]= mydata1[0][i].replace('"','')
#        w=mydata1[0][i].split(' ')
#        words=[]
#        for j in range(0,len(w)):
#            words.append(w[j])
#        s=""
#        s = " ".join(str(x) for x in words)
#        if(dict1.has_key(s)):
#            continue
#        else:
#            dict1[s]=1
#        
#    dictlist=[]
#    for key, value in dict1.iteritems():
#        dictlist.append(key)
#    
#    thefile = open(path+'/list.txt','w')
#    for item in dictlist:
#        thefile.write("%s\n" % item)

############       making a dictionary and storing into list.txt..DO NOT DELETE LIST.TXT###
    
    print "Running part 2"
    dict2={}
    for i in range(0,5671):
        s = d[i]["requester_user_flair"]        
        #print s
        if(dict2.has_key(s)):
            dict2[s]=dict2[s]+1
        else:
            dict2[s]=1
    
    dictlist2=[]
    for key, value in dict2.iteritems():
        if(key!=None):
            dictlist2.append(key.encode("utf-8"))
        else:
            dictlist2.append(key)
            
    
    array_flair_train = np.zeros((5104,len(dictlist2)))
    array_flair_test = np.zeros((567,len(dictlist2)))
    x=0
    y=0
    for i in range(0,5671):
        if(i in list_random):
            s=d[i]["requester_user_flair"]
            if(s in dictlist2):
                array_flair_test[x][dictlist2.index(s)] = array_flair_test[x][dictlist2.index(s)] + 1
                x=x+1
        else:
            s=d[i]["requester_user_flair"]
            if(s in dictlist2):
                array_flair_train[y][dictlist2.index(s)] = array_flair_train[y][dictlist2.index(s)] + 1
                y=y+1
    
    dictlist1 = [line.strip() for line in open(path+"/list.txt", 'r')]
    array_reddits_train = np.zeros((5104,len(dictlist1)))
    array_reddits_test = np.zeros((567,len(dictlist1)))
    
    x=0
    y=0
    for i in range(0,5671):
        if(i in list_random):
            #print "i in test"
            for j in range(0,len(d[i]["requester_subreddits_at_request"])):
                s=d[i]["requester_subreddits_at_request"][j]
                if(s in dictlist1):
                    #print "x=",x
                    array_reddits_test[x][dictlist1.index(s)] = array_reddits_test[x][dictlist1.index(s)] + 1
            x=x+1
        else:
            #print "i in train"
            for j in range(0,len(d[i]["requester_subreddits_at_request"])):
                s=d[i]["requester_subreddits_at_request"][j]
                if(s in dictlist1):
                    #print "y=",y
                    array_reddits_train[y][dictlist1.index(s)] = array_reddits_train[y][dictlist1.index(s)] + 1
            y=y+1
    
    array_rest_train = np.zeros((5104,20))
    array_rest_test = np.zeros((567,20))
    
    x=0
    y=0
    for i in range (0,5671):
        if(i in list_random):
            array_rest_test[x][0] = d[i]["post_was_edited"]
            array_rest_test[x][1] = d[i]["requester_account_age_in_days_at_request"]
            array_rest_test[x][2] = d[i]["requester_account_age_in_days_at_retrieval"]
            array_rest_test[x][3] = d[i]["requester_days_since_first_post_on_raop_at_request"]
            array_rest_test[x][4] = d[i]["requester_days_since_first_post_on_raop_at_retrieval"]
            array_rest_test[x][5] = d[i]["requester_number_of_comments_at_request"]
            array_rest_test[x][6] = d[i]["requester_number_of_comments_at_retrieval"]
            array_rest_test[x][7] = d[i]["requester_number_of_comments_in_raop_at_request"]
            array_rest_test[x][8] = d[i]["requester_number_of_comments_in_raop_at_retrieval"]
            array_rest_test[x][9] = d[i]["requester_number_of_posts_at_request"]
            array_rest_test[x][10] = d[i]["requester_number_of_posts_at_retrieval"]
            array_rest_test[x][11] = d[i]["requester_number_of_posts_on_raop_at_request"]
            array_rest_test[x][12] = d[i]["requester_number_of_posts_on_raop_at_retrieval"]
            array_rest_test[x][13] = d[i]["requester_number_of_subreddits_at_request"]
            array_rest_test[x][14] = d[i]["number_of_downvotes_of_request_at_retrieval"]
            array_rest_test[x][15] = d[i]["number_of_upvotes_of_request_at_retrieval"]
            array_rest_test[x][16] = d[i]["requester_upvotes_minus_downvotes_at_request"]
            array_rest_test[x][17] = d[i]["requester_upvotes_minus_downvotes_at_retrieval"]
            array_rest_test[x][18] = d[i]["requester_upvotes_plus_downvotes_at_request"]
            array_rest_test[x][19] = d[i]["requester_upvotes_plus_downvotes_at_retrieval"]
            x=x+1
        else:
            array_rest_train[y][0] = d[i]["post_was_edited"]
            array_rest_train[y][1] = d[i]["requester_account_age_in_days_at_request"]
            array_rest_train[y][2] = d[i]["requester_account_age_in_days_at_retrieval"]
            array_rest_train[y][3] = d[i]["requester_days_since_first_post_on_raop_at_request"]
            array_rest_train[y][4] = d[i]["requester_days_since_first_post_on_raop_at_retrieval"]
            array_rest_train[y][5] = d[i]["requester_number_of_comments_at_request"]
            array_rest_train[y][6] = d[i]["requester_number_of_comments_at_retrieval"]
            array_rest_train[y][7] = d[i]["requester_number_of_comments_in_raop_at_request"]
            array_rest_train[y][8] = d[i]["requester_number_of_comments_in_raop_at_retrieval"]
            array_rest_train[y][9] = d[i]["requester_number_of_posts_at_request"]
            array_rest_train[y][10] = d[i]["requester_number_of_posts_at_retrieval"]
            array_rest_train[y][11] = d[i]["requester_number_of_posts_on_raop_at_request"]
            array_rest_train[y][12] = d[i]["requester_number_of_posts_on_raop_at_retrieval"]
            array_rest_train[y][13] = d[i]["requester_number_of_subreddits_at_request"]
            array_rest_train[y][14] = d[i]["number_of_downvotes_of_request_at_retrieval"]
            array_rest_train[y][15] = d[i]["number_of_upvotes_of_request_at_retrieval"]
            array_rest_train[y][16] = d[i]["requester_upvotes_minus_downvotes_at_request"]
            array_rest_train[y][17] = d[i]["requester_upvotes_minus_downvotes_at_retrieval"]
            array_rest_train[y][18] = d[i]["requester_upvotes_plus_downvotes_at_request"]
            array_rest_train[y][19] = d[i]["requester_upvotes_plus_downvotes_at_retrieval"]
            y=y+1

    array_final_train = np.concatenate([array_flair_train,array_reddits_train,array_rest_train],axis=1)
    array_final_test = np.concatenate([array_flair_test,array_reddits_test,array_rest_test],axis=1)
    
    clf = svm.LinearSVC()
    clf.fit(array_final_train,y_train)
    answer1 = clf.predict(array_final_test)
    p1 = precision_score(y_test, answer1)
    r1 = recall_score(y_test, answer1)
    f1 = f1_score(y_test, answer1)
    a1 = accuracy_score(y_test, answer1)
    auc1 = roc_auc_score(y_test, answer1)
    mat = confusion_matrix(y_test, answer1)
    print mat
    sp = float(mat[0][0])/float(mat[0][0]+mat[0][1])
    print "accuracy, precision, recall, f1, specificity, auc for first model: ", a1, p1, r1, f1, sp, auc1 
        
###############################################third model################################################

    print "Running part 3"
    with open(path+'\\resources\\narratives\\desire.txt', 'r') as outfile:
        des = [x.strip('\n') for x in outfile.readlines()]
    with open(path+'\\resources\\narratives\\family.txt', 'r') as outfile:
        fam = [x.strip('\n') for x in outfile.readlines()]
    with open(path+'\\resources\\narratives\\job.txt', 'r') as outfile:
        job = [x.strip('\n') for x in outfile.readlines()]
    with open(path+'\\resources\\narratives\\money.txt', 'r') as outfile:
        mon = [x.strip('\n') for x in outfile.readlines()]
    with open(path+'\\resources\\narratives\\student.txt', 'r') as outfile:
        stud = [x.strip('\n') for x in outfile.readlines()]
    
    
    part3 = np.zeros((len(X_train),5))
    i=0
    
    resultDes = re.compile(r'\b%s\b' % '\\b|\\b'.join(des), flags=re.IGNORECASE)
    resultFam = re.compile(r'\b%s\b' % '\\b|\\b'.join(fam), flags=re.IGNORECASE)
    resultJob = re.compile(r'\b%s\b' % '\\b|\\b'.join(job), flags=re.IGNORECASE)
    resultMon = re.compile(r'\b%s\b' % '\\b|\\b'.join(mon), flags=re.IGNORECASE)
    resultStud = re.compile(r'\b%s\b' % '\\b|\\b'.join(stud), flags=re.IGNORECASE)

    for line in X_train:
        denom = len(line.split(' '))
        count = len(resultDes.findall(line))
        part3[i][0]=float(count) / float(denom)
        count = len(resultFam.findall(line))
        part3[i][1]=float(count) / float(denom)
        count = len(resultJob.findall(line))
        part3[i][2]=float(count) / float(denom)
        count = len(resultMon.findall(line))
        part3[i][3]=float(count) / float(denom)
        count = len(resultStud.findall(line))
        part3[i][4]=float(count) / float(denom)
        i=i+1
    
    part4 = np.zeros((len(X_test),5))
    i=0
    for line in X_test:
        denom = float(len(line.split(' ')))
        count = float(len(resultDes.findall(line)))
        part4[i][0]=count / denom
        count = len(resultFam.findall(line))
        part4[i][1]=count / denom
        count = len(resultJob.findall(line))
        part4[i][2]=count / denom
        count = len(resultMon.findall(line))
        part4[i][3]=count / denom
        count = len(resultStud.findall(line))
        part4[i][4]=count / denom
        i=i+1

    
    clf = svm.LinearSVC(class_weight = 'balanced')
    clf.fit(part3,y_train)
    answer1 = clf.predict(part4)
    p1 = precision_score(y_test, answer1)
    r1 = recall_score(y_test, answer1)
    f1 = f1_score(y_test, answer1)
    a1 = accuracy_score(y_test, answer1)
    auc1 = roc_auc_score(y_test, answer1)
    mat = confusion_matrix(y_test, answer1)
    sp = float(mat[0][0])/float(mat[0][0]+mat[0][1])    
    print "accuracy, precision, recall, f1, specificity, auc for first model: ", a1,p1,r1,f1,sp,auc1


############################################## fourth model ##############################################

    print "Running part 4"
    harmVirtue = []
    harmVice = []
    fairnessVirtue = []
    fairnessVice = []
    ingroupVirtue = []
    ingroupVice = []
    authorityVirtue = []
    authorityVice = []
    purityVirtue = []
    purityVice = []
    moralityGeneral = []
    
    des = []
    with open(path+'\\resources\MoralFoundations.dic', 'r') as outfile:
        for x in outfile.readlines():
            s = x.strip('\n')
            s = s.strip()
            s = s.split('\t') 
            des.append(s)
        
    for i in range(0,len(des)):
        for j in range(0,len(des[i])):
            if(re.search('[a-zA-Z]', des[i][j])):       #if it is the word
                str1 = des[i][j]
            elif(des[i][j]==''):                        #if it is a blank space
                continue
            else:                                       #if it is a number
                a = des[i][j].split()
                for k in range(0,len(a)):
                    if(a[k]=='01'):
                        harmVirtue.append(str1)
                    elif(a[k]=='02'):
                        harmVice.append(str1)
                    elif(a[k]=='03'):
                        fairnessVirtue.append(str1)
                    elif(a[k]=='04'):
                        fairnessVice.append(str1)
                    elif(a[k]=='05'):
                        ingroupVirtue.append(str1)
                    elif(a[k]=='06'):
                        ingroupVice.append(str1)
                    elif(a[k]=='07'):
                        authorityVirtue.append(str1)
                    elif(a[k]=='08'):
                        authorityVice.append(str1)
                    elif(a[k]=='09'):
                        purityVirtue.append(str1)
                    elif(a[k]=='10'):
                        purityVice.append(str1)
                    elif(a[k]=='11'):
                        moralityGeneral.append(str1.strip())

    print (harmVirtue)

####   training data    
    for l in range(0,len(X_train)):
        X_train[l] = X_train[l].replace('\'','')
        X_train[l] = X_train[l].replace('(','')    
        X_train[l] = X_train[l].replace(')','')   
        X_train[l] = X_train[l].replace(',','')   
        X_train[l] = X_train[l].replace('[','')    
        X_train[l] = X_train[l].replace(']','')   
        
    mat4 = np.zeros((len(X_train),11))
    i=0
    for line in X_train:
        for w in harmVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][0]=mat4[i][0]+1
        for w in harmVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][1]=mat4[i][1]+1
        for w in fairnessVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][2]=mat4[i][2]+1
        for w in fairnessVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][3]=mat4[i][3]+1
        for w in ingroupVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][4]=mat4[i][4]+1
        for w in ingroupVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][5]=mat4[i][5]+1
        for w in authorityVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][6]=mat4[i][6]+1
        for w in authorityVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][7]=mat4[i][7]+1
        for w in purityVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][8]=mat4[i][8]+1
        for w in purityVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][9]=mat4[i][9]+1
        for w in moralityGeneral:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4[i][10]=mat4[i][10]+1
        i=i+1
    
    i=0
    for line in X_train:
        denom = float(len(line.split(' ')))
        for j in range(0,len(mat4[i])):
            mat4[i][j] = float(mat4[i][j]) / float(denom)
        i = i + 1
    
    
    ####   testing data    
    for l in range(0,len(X_test)):
        X_test[l] = X_test[l].replace('\'','')
        X_test[l] = X_test[l].replace('(','')    
        X_test[l] = X_test[l].replace(')','')   
        X_test[l] = X_test[l].replace(',','')   
        X_test[l] = X_test[l].replace('[','')    
        X_test[l] = X_test[l].replace(']','')   
        
    mat4_test = np.zeros((len(X_test),11))
    i=0
    for line in X_test:
        for w in harmVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][0]=mat4_test[i][0]+1
        for w in harmVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][1]=mat4_test[i][1]+1
        for w in fairnessVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][2]=mat4_test[i][2]+1
        for w in fairnessVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][3]=mat4_test[i][3]+1
        for w in ingroupVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][4]=mat4_test[i][4]+1
        for w in ingroupVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][5]=mat4_test[i][5]+1
        for w in authorityVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][6]=mat4_test[i][6]+1
        for w in authorityVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][7]=mat4_test[i][7]+1
        for w in purityVirtue:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][8]=mat4_test[i][8]+1
        for w in purityVice:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][9]=mat4_test[i][9]+1
        for w in moralityGeneral:
            if(re.search(w,line, re.X | re.IGNORECASE)):
                mat4_test[i][10]=mat4_test[i][10]+1
        i=i+1
    
    i=0
    for line in X_test:
        denom = float(len(line.split(' ')))
        for j in range(0,len(mat4_test[i])):
            mat4_test[i][j] = float(mat4_test[i][j]) / float(denom)
        i = i + 1
    
    
    clf = svm.LinearSVC(class_weight='balanced')
    clf.fit(mat4,y_train)
    answer1 = clf.predict(mat4_test)
    p1 = precision_score(y_test, answer1)
    r1 = recall_score(y_test, answer1)
    f1 = f1_score(y_test, answer1)
    a1 = accuracy_score(y_test, answer1)
    auc1 = roc_auc_score(y_test, answer1)
    mat = confusion_matrix(y_test, answer1)
    sp = float(mat[0][0])/float(mat[0][0]+mat[0][1])
    print "accuracy, precision, recall, f1, specificity, auc for first model: ", a1, p1, r1, f1, sp,auc1 
    
if __name__ == "__main__":
    assignment()