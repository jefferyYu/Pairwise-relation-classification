Author

Jianfei YU

jfyu.2014@phdis.smu.edu.sg

Mar 12, 2017

Data and Code for:

Pairwise relation classification with mirror instances and a combined convolutional neural network
COLING 2016
http://aclweb.org/anthology/C/C16/C16-1223.pdf

I. Data

1. In this releasing code, we just use the Semeval-2010 dataset to show our proposed Comb+MI and Comb+RMI models.
2. We attach the original dataset in the folder "SemEval2010_task8_all_data".
3. We also attach our extracted shortest dependency path(SDP) between two entities in the folder "SemEval2010_task8_all_data".
   The SDPs of training instances are under the folder "SemEval2010_task8_training", named "train_p1.txt", "train_p2.txt", "train_p3.txt" and "train_p4.txt". Each contains 2000 training instances.
   The SDPs of test instances are under the folder "SemEval2010_task8_testing_keys", named "test_all.txt".
4. The ACE data is not included because of licensing issues.

II. Code

Part1: Pre-process code: 

Run on Python 2.7, and the pre-process code requires Python package hdf5 (the h5py module)

Step 1. You can directly run the following codes:
python preprocess_mipe+dep.py
Note that before you run, you need to download word2vec vectors from here: https://code.google.com/archive/p/word2vec/  , and then set w2v_path in line 626.


Part2: Model Code:

Run on Torch7, and the training model requires the lua package: hdf5
To run the Comb+MI and Comb+RMI, you can just run:
sh run.sh

Part3: Results:

By running the codes, you should get the following result (the "main_mipecomb.lua" file refers to the Comb+MI model, while "main_mipecombneg.lua" file refers to the Comb+RMI model):
	       Comb+MI	Comb+RMI
F1_score    84.08	  84.86

which is slightly different from the results we report in Table 5 in our paper. 
The reason is that in our previous experiments, we use a random seed for both Comb+MI and Comb+RMI. 
But now for fair comparison, we set the seed in both models to the same value 0.
Also, in this released code, I reduce 80% negative mirror instances while in our paper we reduce 50%. 


For convenience, to show our running process, we also attach the "miresult.txt" and "rmiresult.txt" in the folder "runing_example".


Acknowledgements

Most of the code are based on the code by Harvard NLP group: https://github.com/harvardnlp/sent-conv-torch.
Using this code means you have read and accepted the copyrights set by the dataset providers.

License:

Singapore Management University
