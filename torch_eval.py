import numpy as np
import os
import math
import sys

LOG_LIM = 1E-300

def createCatDic(catNames):
    catDic = {}
    label = 0;
    for cat in catNames:
        catDic[label]=catNames[label]
        label += 1
    return catDic

def transformtxt2(pb1_list_file, y_file, a1, b1):

    catNames = ['Other','Cause-Effect(e1,e2)','Component-Whole(e1,e2)','Content-Container(e1,e2)','Entity-Destination(e1,e2)','Entity-Origin(e1,e2)',
                'Instrument-Agency(e1,e2)','Member-Collection(e1,e2)','Message-Topic(e1,e2)','Product-Producer(e1,e2)',
                'Cause-Effect(e2,e1)','Component-Whole(e2,e1)','Content-Container(e2,e1)','Entity-Destination(e2,e1)','Entity-Origin(e2,e1)',
                'Instrument-Agency(e2,e1)','Member-Collection(e2,e1)','Message-Topic(e2,e1)','Product-Producer(e2,e1)']
    catDic = createCatDic(catNames)     
    
    fylist = open(y_file).readlines()
    y = []
    for i in xrange(len(fylist)):
	y.append(int(fylist[i]))
    pb1_list = []
    fpb1 = open(pb1_list_file).readlines()
    for i in xrange(len(fpb1)):
	pb1_list.append(int(fpb1[i]))  

    #true_label = []    
    fout = open(a1, 'w')
    i = 1
    for k in range(len(y)):
        label_1 = y[k]-1
        #true_label.append(catDic[label])
        fout.writelines(str(i) + '\t'+catDic[label_1] + '\n')
        i += 1
    fout.close()

    #predict_label = []
    fout = open(b1, 'w')
    j = 1
    for w in range(len(pb1_list)):
	label_2 = pb1_list[w]-1
	    #true_label.append(catDic[label])
	fout.writelines(str(j) + '\t'+catDic[label_2] + '\n')
        j += 1
    fout.close()

    
if __name__=="__main__":
    
    pred = sys.argv[1]
    y = sys.argv[2]

    result = ''#'result_mi_joint_neg0.3' + os.sep
    
    transformtxt2(pred, y, result+'true_ori.txt', result+'pred_ori.txt')
    os.system('perl SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_scorer-v1.2' + os.sep + 'semeval2010_task8_scorer-v1.2.pl' + ' ' + result+'pred_ori.txt' + ' '+ result + 'true_ori.txt'+ ' > '+result+'result_jfyu.txt')
    
    F_score = open(result+'result_jfyu.txt', 'r').readlines()
    F1 = F_score[-1].strip().split('F1 = ')[1]
    print pred+' F1_score:', F1    
    if pred == 'test_predict':
        print '-----------------------------'

    
    


