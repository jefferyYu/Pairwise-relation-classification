import numpy as np
import h5py
import re
import sys
import operator
import argparse
import os

def formatLine(line):
    # This will take in a raw input sentence and return [[element strings], [indicies of elements], [sentence with elements removed]]
    words = line.split(' ')
    
    e1detagged = []
    e2detagged = []
    rebuiltLine = ''
    count = 1
    for word in words:
        # if tagged at all
        if word.find('<e') != -1 or word.find('</e') != -1:
            # e1 or e2
            if word[2] == '1' or word[word.find('>')-1] == '1':
                # remove tags from word for 
                e1detagged = getWord(words,word)
                e1detagged.append(count)
                # replace and tac back on . at end if needed
                word = replaceWord(word)
            else:
                e2detagged = getWord(words,word)
                e2detagged.append(count)
                word = replaceWord(word)
        rebuiltLine += ' ' + word
        count += 1
    rebuiltLine = rebuiltLine[1:len(rebuiltLine)]
    rebuiltLine += '\n'
    return [[e1detagged[0], e2detagged[0]],[e1detagged[1],e2detagged[1]],[e1detagged[2],e2detagged[2]],rebuiltLine]


def getWord(words, word):
    if endTwoWords(word):
        return [replaceWord(word, False), 1]
    else:
        return [replaceWord(word, False), 0]
    

def replaceWord(word, shouldEndSentence = True):
    wordList = word.split('</')
    endSentence = ''
    if len(wordList) == 2 and len(wordList[len(wordList)-1]) != 3:
        end = wordList[len(wordList)-1]
        endSentence += end[end.find('>')+1:len(end)]
    wordList = wordList[0].split('>')
    newWord = wordList[len(wordList)-1]
    if shouldEndSentence:
        newWord += endSentence
    return newWord
    
# if this has a two words ex. <e2>fast cars</e2>
def endTwoWords(word):
    #print word
    return word.find('<e') == -1

def createCatDic(catNames):
    catDic = {}
    label = 0;
    for cat in catNames:
        catDic[catNames[label]]=label
        label += 1
    return catDic

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def line_to_words(line, dataset):
  if dataset == 'SST1' or dataset == 'SST2':
    clean_line = clean_str_sst(line.strip())
  else:
    clean_line = clean_str(line.strip())
  words = clean_line.split(' ')
  words = words[1:]

  return words

def get_vocab(file_list, dep_path_dict, dep_test_path_dict, dataset=''):
  max_sent_len = 0
  word_to_idx = {}
  pos_to_idx = {}
  # Starts at 2 for padding
  idx = 2
  clean_string = True
  for filename in file_list:
    f = open(filename, "r")
    flen = len(f.readlines())
    f = open(filename, "r")
    if flen == 32000:
	sentenceNum = 1
	for line in f:
	    if (line != '' and not line[0].isdigit()):
		continue;
	    line = line[line.find('"')+1:len(line)-3]
	    rev = []
	    rev.append(line.strip())		    
	    if clean_string:
		orig_rev = clean_str(" ".join(rev))
	    else:
		orig_rev = " ".join(rev).lower()	    
      
	    formatOutput = formatLine(orig_rev)
	    line = formatOutput[3]
	    orig_rev = formatOutput[3]   
	    entity = formatOutput[2]
	    words = orig_rev.split()
    
	    max_sent_len = max(max_sent_len, len(words))
	    for word in words:
		if not word in word_to_idx:
		    word_to_idx[word] = idx
		    idx += 1
	    dep_labels = set(dep_path_dict[str(sentenceNum)+'com'].split(', '))
	    for dep in dep_labels:
		if not dep in word_to_idx:
		    word_to_idx[dep] = idx
		    idx += 1
	    sentenceNum += 1
    else:
	sentenceNum = 1
	for line in f:
	    if (line != '' and not line[0].isdigit()):
		continue;
	    line = line[line.find('"')+1:len(line)-3]
	    rev = []
	    rev.append(line.strip())		    
	    if clean_string:
		orig_rev = clean_str(" ".join(rev))
	    else:
		orig_rev = " ".join(rev).lower()	    
      
	    formatOutput = formatLine(orig_rev)
	    line = formatOutput[3]
	    orig_rev = formatOutput[3]   
	    entity = formatOutput[2]
	    words = orig_rev.split()
    
	    max_sent_len = max(max_sent_len, len(words))
	    for word in words:
		if not word in word_to_idx:
		    word_to_idx[word] = idx
		    idx += 1
	    dep_labels = set(dep_test_path_dict[str(8000+sentenceNum)+'com'].split(', '))
	    for dep in dep_labels:
		if not dep in word_to_idx:
		    word_to_idx[dep] = idx
		    idx += 1
	    sentenceNum += 1	

    f.close()
    ps_idx = 2
    for i in range(-max_sent_len, max_sent_len):
	if not i in word_to_idx:
	    pos_to_idx[i] = ps_idx
	    ps_idx += 1

  return max_sent_len, word_to_idx, pos_to_idx

def load_data(dataset, dep_path_dict, dep_test_path_dict, train_name, test_name='', dev_name='', padding=4):
  """
  Load training data (dev/test optional).
  """
  catNames = ['Other','Cause-Effect(e1,e2)','Component-Whole(e1,e2)','Content-Container(e1,e2)','Entity-Destination(e1,e2)','Entity-Origin(e1,e2)',
              'Instrument-Agency(e1,e2)','Member-Collection(e1,e2)','Message-Topic(e1,e2)','Product-Producer(e1,e2)',
              'Cause-Effect(e2,e1)','Component-Whole(e2,e1)','Content-Container(e2,e1)','Entity-Destination(e2,e1)','Entity-Origin(e2,e1)',
              'Instrument-Agency(e2,e1)','Member-Collection(e2,e1)','Message-Topic(e2,e1)','Product-Producer(e2,e1)']  
  catDic = createCatDic(catNames)
  f_names = [train_name]
  if not test_name == '': f_names.append(test_name)
  if not dev_name == '': f_names.append(dev_name)

  max_sent_len, word_to_idx, pos_to_idx = get_vocab(f_names, dep_path_dict, dep_test_path_dict, dataset)

  dev = []
  dev_label = []
  midev_label = []
  dev_e1 = []
  dev_e2 = []  
  dev_dep = []
  dev_dep_e1 = []
  dev_dep_e2 = []
  
  train = []
  train_e1 = []
  train_e2 = []
  train_label = []
  mitrain_label = []
  train_dep = []
  train_dep_e1 = []
  train_dep_e2 = []
  
  
  test = []
  test_e1 = []
  test_e2 = []
  test_label = []
  mitest_label = []
  test_dep = []
  test_dep_e1 = []
  test_dep_e2 = []

  files = []
  data = []
  entity1 = []
  entity2 = []
  data_dep = []
  dep_entity1 = []
  dep_entity2 = []
  data_label = []
  midata_label = []
  clean_string = True

  f_train = open(train_name, 'r')
  files.append(f_train)
  data.append(train)
  entity1.append(train_e1)
  entity2.append(train_e2)
  data_dep.append(train_dep)
  dep_entity1.append(train_dep_e1)
  dep_entity2.append(train_dep_e2)
  data_label.append(train_label)
  midata_label.append(mitrain_label)
  
  if not test_name == '':
    f_test = open(test_name, 'r')
    files.append(f_test)
    data.append(test)
    entity1.append(test_e1)
    entity2.append(test_e2)    
    data_label.append(test_label)
    midata_label.append(mitest_label)
    data_dep.append(test_dep)
    dep_entity1.append(test_dep_e1)
    dep_entity2.append(test_dep_e2)    
  if not dev_name == '':
    f_dev = open(dev_name, 'r')
    files.append(f_dev)
    data.append(dev)
    entity1.append(dev_e1)
    entity2.append(dev_e2)      
    data_label.append(dev_label)
    midata_label.append(midev_label)
    data_dep.append(dev_dep)
    dep_entity1.append(dev_dep_e1)
    dep_entity2.append(dev_dep_e2)    

  for d, e1, e2, depd, depe1, depe2, lbl, milbl, f in zip(data, entity1, entity2, data_dep, dep_entity1, dep_entity2, data_label, midata_label, files):
    alllines = f.readlines()
    if len(alllines) == 32000:
	sentenceNum = 1
	for i in xrange(len(alllines)):
	    line = alllines[i]
	    if (line != '' and not line[0].isdigit()):
		continue;
	    line = line[line.find('"')+1:len(line)-3]
	    rev = []
	    rev.append(line.strip())		    
	    if clean_string:
		orig_rev = clean_str(" ".join(rev))
	    else:
		orig_rev = " ".join(rev).lower()	    
      
	    formatOutput = formatLine(orig_rev)
	    line = formatOutput[3]
	    orig_rev = formatOutput[3]   
	    entity = formatOutput[2]
	    words = orig_rev.split()  
	    #print(words[entity[0]-1])
	    #print(words[entity[1]-1])
	    #print(words)
	    
	    entity1 = []
	    entity2 = []
	    
	    line = alllines[i+1]
	    if (line != '' and not line[0].isdigit()):
		category = line
	    # use dictionary to get sentence label
	    y = int(catDic.get(category.strip())) + 1
	    sent = [word_to_idx[word] for word in words]
	    word_num = 1
	    pos_e1 = []
	    pos_e2 = []
	    for word in words:  
		pos_e1.append(word_num-entity[0])
		entity1.append(pos_to_idx[word_num-entity[0]])
		pos_e2.append(word_num-entity[1])
		entity2.append(pos_to_idx[word_num-entity[1]])            
		word_num += 1	
	    
	    sdp = dep_path_dict[str(sentenceNum)+'com'].split(', ')
	    dep_path = [word_to_idx[word] for word in sdp]
	    dep_pos_entity1 = []
	    dep_pos_entity2 = []
	    dep_num = 0
	    for word in sdp:
		dep_pos_entity1.append(pos_to_idx[dep_num])
		dep_pos_entity2.append(pos_to_idx[dep_num-len(sdp)+1])
		dep_num += 1
	    #print(pos_e1)
	    #print(pos_e2)
	    # end padding
	    if len(sent) < max_sent_len + padding:
		sent.extend([1] * (max_sent_len + padding - len(sent)))
		entity1.extend([1] * (max_sent_len + padding - len(entity1)))
		entity2.extend([1] * (max_sent_len + padding - len(entity2)))
		dep_path.extend([1] * (max_sent_len + padding - len(dep_path)))
		dep_pos_entity1.extend([1] * (max_sent_len + padding - len(dep_pos_entity1)))
		dep_pos_entity2.extend([1] * (max_sent_len + padding - len(dep_pos_entity2)))
	    # start padding
	    #print(sent)
	    #print(entity1)
	    #print(entity2)
	    sent = [1]*padding + sent
	    entity1 = [1]*padding + entity1
	    entity2 = [1]*padding + entity2
	    dep_path =  [1]*padding + dep_path
	    dep_pos_entity1 = [1]*padding + dep_pos_entity1
	    dep_pos_entity2 = [1]*padding + dep_pos_entity2
	    
	    d.append(sent)
	    e1.append(entity1)
	    e2.append(entity2)
	    depd.append(dep_path)
	    depe1.append(dep_pos_entity1)
	    depe2.append(dep_pos_entity2)
	    lbl.append(y)
	    if y ==1:
		milbl.append(y)
	    elif 1<y<=10:
		milbl.append(y+9)
	    else:
		milbl.append(y-9)	
	    sentenceNum += 1
    else:
	sentenceNum = 1
	for i in xrange(len(alllines)):
	    line = alllines[i]
	    if (line != '' and not line[0].isdigit()):
		continue;
	    line = line[line.find('"')+1:len(line)-3]
	    rev = []
	    rev.append(line.strip())		    
	    if clean_string:
		orig_rev = clean_str(" ".join(rev))
	    else:
		orig_rev = " ".join(rev).lower()	    
      
	    formatOutput = formatLine(orig_rev)
	    line = formatOutput[3]
	    orig_rev = formatOutput[3]   
	    entity = formatOutput[2]
	    words = orig_rev.split()  
	    #print(words[entity[0]-1])
	    #print(words[entity[1]-1])
	    #print(words)
	    
	    entity1 = []
	    entity2 = []
	    
	    line = alllines[i+1]
	    if (line != '' and not line[0].isdigit()):
		category = line
		# use dictionary to get sentence label
	    y = int(catDic.get(category.strip())) + 1
	    sent = [word_to_idx[word] for word in words]
	    word_num = 1
	    pos_e1 = []
	    pos_e2 = []
	    for word in words:  
		pos_e1.append(word_num-entity[0])
		entity1.append(pos_to_idx[word_num-entity[0]])
		pos_e2.append(word_num-entity[1])
		entity2.append(pos_to_idx[word_num-entity[1]])            
		word_num += 1	
	    
	    sdp = dep_test_path_dict[str(8000+sentenceNum)+'com'].split(', ')
	    dep_path = [word_to_idx[word] for word in sdp]
	    dep_pos_entity1 = []
	    dep_pos_entity2 = []
	    dep_num = 0
	    for word in sdp:
		dep_pos_entity1.append(pos_to_idx[dep_num])
		dep_pos_entity2.append(pos_to_idx[dep_num-len(sdp)+1])
		dep_num += 1
	    #print(pos_e1)
	    #print(pos_e2)
	    # end padding
	    if len(sent) < max_sent_len + padding:
		sent.extend([1] * (max_sent_len + padding - len(sent)))
		entity1.extend([1] * (max_sent_len + padding - len(entity1)))
		entity2.extend([1] * (max_sent_len + padding - len(entity2)))
		dep_path.extend([1] * (max_sent_len + padding - len(dep_path)))
		dep_pos_entity1.extend([1] * (max_sent_len + padding - len(dep_pos_entity1)))
		dep_pos_entity2.extend([1] * (max_sent_len + padding - len(dep_pos_entity2)))
	    # start padding
	    #print(sent)
	    #print(entity1)
	    #print(entity2)
	    sent = [1]*padding + sent
	    entity1 = [1]*padding + entity1
	    entity2 = [1]*padding + entity2
	    dep_path =  [1]*padding + dep_path
	    dep_pos_entity1 = [1]*padding + dep_pos_entity1
	    dep_pos_entity2 = [1]*padding + dep_pos_entity2
	    
	    d.append(sent)
	    e1.append(entity1)
	    e2.append(entity2)
	    depd.append(dep_path)
	    depe1.append(dep_pos_entity1)
	    depe2.append(dep_pos_entity2)
	    lbl.append(y)
	    if y ==1:
		milbl.append(y)
	    elif 1<y<=10:
		milbl.append(y+9)
	    else:
		milbl.append(y-9)	
	    sentenceNum += 1

  f_train.close()
  if not test_name == '':
    f_test.close()
  if not dev_name == '':
    f_dev.close()

  return word_to_idx, pos_to_idx, np.array(train, dtype=np.int32), np.array(train_e1, dtype=np.int32), np.array(train_e2, dtype=np.int32), np.array(train_label, dtype=np.int32), np.array(mitrain_label, dtype=np.int32), np.array(test, dtype=np.int32), np.array(test_e1, dtype=np.int32), np.array(test_e2, dtype=np.int32), np.array(test_label, dtype=np.int32), np.array(mitest_label, dtype=np.int32), np.array(dev, dtype=np.int32), np.array(dev_e1, dtype=np.int32), np.array(dev_e2, dtype=np.int32),np.array(dev_label, dtype=np.int32),np.array(midev_label, dtype=np.int32), np.array(train_dep, dtype=np.int32), np.array(train_dep_e1, dtype=np.int32), np.array(train_dep_e2, dtype=np.int32), np.array(test_dep, dtype=np.int32), np.array(test_dep_e1, dtype=np.int32), np.array(test_dep_e2, dtype=np.int32), np.array(dev_dep, dtype=np.int32), np.array(dev_dep_e1, dtype=np.int32), np.array(dev_dep_e2, dtype=np.int32)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\"", "", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r";", " ; ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"--", " ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)      
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

FILE_PATHS = {"SST1": ("data/stsa.fine.phrases.train",
                  "data/stsa.fine.dev",
                  "data/stsa.fine.test"),
              "SST2": ("data/stsa.binary.phrases.train",
                  "data/stsa.binary.dev",
                  "data/stsa.binary.test"),
              "MR": ("data/rt-polarity.all", "", ""),
              "SUBJ": ("data/subj.all", "", ""),
              "CR": ("data/custrev.all", "", ""),
              "MPQA": ("data/mpqa.all", "", ""),
              "TREC": ("data/TREC.train.all", "", "data/TREC.test.all"),
              "Semeval_pedep": ('SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_training' + os.sep + 'TRAIN_FILE.TXT', "", 'SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_testing_keys' + os.sep + 'TEST_FILE_FULL.TXT'),
              }
args = {}

def main():
  global args
  parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  #parser.add_argument('dataset', help="Data set", type=str)
  #parser.add_argument('w2v', help="word2vec file", type=str)
  parser.add_argument('--train', help="custom train data", type=str, default="")
  parser.add_argument('--test', help="custom test data", type=str, default="")
  parser.add_argument('--dev', help="custom dev data", type=str, default="")
  parser.add_argument('--padding', help="padding around each sentence", type=int, default=4)
  parser.add_argument('--custom_name', help="name of custom output hdf5 file", type=str, default="custom")
  args = parser.parse_args()
  dataset = 'Semeval_pedep'
  if dataset == 'custom':
    dataset = args.custom_name

  # Dataset name
  if dataset == 'custom':
    # Train on custom dataset
    train_path, dev_path, test_path = args.train, args.dev, args.test
  else:
    train_path, dev_path, test_path = FILE_PATHS[dataset]

  '''shortest dependency path extraction'''
  dep_path_dict = {}
  fin1 = open('SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_training' + os.sep + 'train_p1.txt', 'r')
  fin2 = open('SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_training' + os.sep + 'train_p2.txt', 'r')
  fin3 = open('SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_training' + os.sep + 'train_p3.txt', 'r')
  fin4 = open('SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_training' + os.sep + 'train_p4.txt', 'r')
  dep1_list = fin1.readlines()
  for i in range(len(dep1_list)):
      if len(dep1_list[i].strip()) == 0:
	  continue;
      else:
	  sent_num = int(dep1_list[i].strip().split('[')[0].split(' ')[0])
	  sent_num += 0
	  dep_path = dep1_list[i].strip().split('[')[1].split(' ]')[0]
	  if dep_path_dict.has_key(sent_num):
	      continue;
	  else:
	      if dep1_list[i+1] == '\n':
		  dep_path_dict[str(sent_num)+'inv'] = dep_path
	      else:
		  dep_path_dict[str(sent_num)+'com'] = dep_path     
  dep2_list = fin2.readlines()
  for i in range(len(dep2_list)):
      if len(dep2_list[i].strip()) == 0:
	  continue;
      else:
	  sent_num = int(dep2_list[i].strip().split('[')[0].split(' ')[0])
	  sent_num += 2000
	  dep_path = dep2_list[i].strip().split('[')[1].split(' ]')[0]
	  if dep_path_dict.has_key(sent_num):
	      continue;
	  else:
	      if dep2_list[i+1] == '\n':
		  dep_path_dict[str(sent_num)+'inv'] = dep_path
	      else:
		  dep_path_dict[str(sent_num)+'com'] = dep_path    
  dep3_list = fin3.readlines()
  for i in range(len(dep3_list)):
      if len(dep3_list[i].strip()) == 0:
	  continue;
      else:
	  sent_num = int(dep3_list[i].strip().split('[')[0].split(' ')[0])
	  sent_num += 4000
	  dep_path = dep3_list[i].strip().split('[')[1].split(' ]')[0]
	  if dep_path_dict.has_key(sent_num):
	      continue;
	  else:
	      if dep3_list[i+1] == '\n':
		  dep_path_dict[str(sent_num)+'inv'] = dep_path
	      else:
		  dep_path_dict[str(sent_num)+'com'] = dep_path
  
  dep4_list = fin4.readlines()
  for i in range(len(dep4_list)):
      if len(dep4_list[i].strip()) == 0:
	  continue;
      else:
	  sent_num = int(dep4_list[i].strip().split('[')[0].split(' ')[0])
	  sent_num += 6000
	  dep_path = dep4_list[i].strip().split('[')[1].split(' ]')[0]
	  if dep_path_dict.has_key(sent_num):
	      continue;
	  else:
	      if dep4_list[i+1] == '\n':
		  dep_path_dict[str(sent_num)+'inv'] = dep_path
	      else:
		  dep_path_dict[str(sent_num)+'com'] = dep_path    
		  
  dep_test_path_dict = {}
  fin_test = open('SemEval2010_task8_all_data' + os.sep + 'SemEval2010_task8_testing_keys' + os.sep + 'test_all.txt', 'r')
  dep_test_list = fin_test.readlines()
  for i in range(len(dep_test_list)):
      if len(dep_test_list[i].strip()) == 0:
	  continue;
      else:
	  sent_num = int(dep_test_list[i].strip().split('[')[0].split(' ')[0])
	  sent_num += 8000
	  dep_path = dep_test_list[i].strip().split('[')[1].split(' ]')[0]
	  if dep_test_path_dict.has_key(sent_num):
	      continue;
	  else:
	      if dep_test_list[i+1] == '\n':
		  dep_test_path_dict[str(sent_num)+'inv'] = dep_path
	      else:
		  dep_test_path_dict[str(sent_num)+'com'] = dep_path       
  # Load data
  word_to_idx, pos_to_idx, train, train_e1, train_e2, train_label, mitrain_label, test, test_e1, test_e2, test_label, mitest_label, dev, dev_e1, dev_e2, dev_label, midev_label, train_dep, train_dep_e1, train_dep_e2, test_dep, test_dep_e1, test_dep_e2, dev_dep, dev_dep_e1, dev_dep_e2 = load_data(dataset, dep_path_dict, dep_test_path_dict, train_path, test_name=test_path, dev_name=dev_path, padding=args.padding)

  # Write word mapping to text file.
  with open(dataset + '_word_mapping.txt', 'w+') as embeddings_f:
    embeddings_f.write("*PADDING* 1\n")
    for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
      embeddings_f.write("%s %d\n" % (word, idx))
    
  with open(dataset + '_pos_mapping.txt', 'w+') as embeddings_f:
    embeddings_f.write("*PADDING* 1\n")
    for pos, idx in sorted(pos_to_idx.items(), key=operator.itemgetter(1)):
      embeddings_f.write("%s %d\n" % (pos, idx))

  # Load word2vec
  w2v = load_bin_vec('/home/jfyu/torch/1.bin', word_to_idx)
  V = len(word_to_idx) + 1
  PV = len(pos_to_idx) + 1
  print 'Vocab size:', V
  print 'Postion:', PV

  # Not all words in word_to_idx are in w2v.
  # Word embeddings initialized to random Unif(-0.25, 0.25)
  embed = np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0])))
  embed[0] = 0
  for word, vec in w2v.items():
    embed[word_to_idx[word] - 1] = vec
  
  pembed = np.random.uniform(-0.25, 0.25, (PV, 50))
  pembed[0] = 0

  # Shuffle train
  print 'train size:', train.shape
  N = train.shape[0]
  perm = np.random.permutation(N)
  train = train[perm]
  train_e1 = train_e1[perm]
  train_e2 = train_e2[perm]
  train_label = train_label[perm]
  mitrain_label = mitrain_label[perm]
  train_dep = train_dep[perm]
  train_dep_e1 = train_dep_e1[perm]
  train_dep_e2 = train_dep_e2[perm]
  
  filename = dataset + '.hdf5'
  with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(embed)
    f["p2v"] = np.array(pembed)
    f['train'] = train
    f['train_e1'] = train_e1
    f['train_e2'] = train_e2
    f['train_dep'] = train_dep
    f['train_dep_e1'] = train_dep_e1
    f['train_dep_e2'] = train_dep_e2    
    f['train_label'] = train_label
    f['mitrain_label'] = mitrain_label
    f['test'] = test
    f['test_e1'] = test_e1
    f['test_e2'] = test_e2
    f['test_dep'] = test_dep
    f['test_dep_e1'] = test_dep_e1
    f['test_dep_e2'] = test_dep_e2
    f['test_label'] = test_label
    f['mitest_label'] = mitest_label
    f['dev'] = dev
    f['dev_e1'] = dev_e1
    f['dev_e2'] = dev_e2   
    f['dev_dep'] = dev_dep
    f['dev_dep_e1'] = dev_dep_e1
    f['dev_dep_e2'] = dev_dep_e2
    f['dev_label'] = dev_label
    f['midev_label'] = midev_label

if __name__ == '__main__':
  main()
