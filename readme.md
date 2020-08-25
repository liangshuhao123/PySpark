##COMP5349 Assignment 2 - Text Corpus Analysis
by Shuhao Liang and Shulei Chen

The folder contains 4 files (voc_exp.py, voc_exp.ipynb, sen_vec_exp.py, sen_vec_exp.ipynb)
Each application are stored as two format (*.py, *.ipynb)

This assignment implements a spark application to analyze words in MultiNLI which contains 2 parts. One is vocabulary exploration and the other is sentence vector exploration. We implemented it by Python and pyspark and use AWS EMR and Jupyter notebook to run and test the results.


##Vocabulary Exploration

* Part A (First three questions)
    *  the number of common words between matched and mismatched sets 
    *  the number of words unique to the matched sets
    *  the number of words unique to the mismatched sets
* Part B
    *  The percentages of words appearing in five genres, in four genres, in three genres, in two genres and in one genre
    *  The same percentages after removing a given list of stop words

##Vocabulary Exploration

* Part A 
    *  Confusion Matrix for TFIDF
* Part B
    *  Confusion Matrix for Pre-trained Sentence Encoder


##Environment Configuration
### Create EMR Cluster
#### - In Step1:  
  Release : emr-5.29.0;  
  Choose Hadoop2.8.5, spark2.4.4, Livy0.6.0, Tensorflow1.14.0  
  Enter configuration:  [{"classification":"spark","properties":{"maximizeResourceAllocation":"true"}}]
  
#### - In Step2: 
Use m4.xlarge with at least 1 master node and 1 core node 

* For performance evaluation, we use
    * Use m4.large for 1 master node and 5 core nodes
    * Use m4.xlarge for 1 master node and 2 core nodes

#### - In Step3:  
  Change cluster name to a readable name  
  Bootstrapping Action:  
    sudo python3 -m nltk.downloader -d /usr/local/share/nltk_data all  
    sudo pip-3.6 install --quiet tensorflow-hub  
    sudo pip-3.6 install --quiet matplotlib  

#### - In Step4:  
  Choose your own key value pair  
  
## Create Jupyter Notebook  
* Choose the cluster that we create before and choose a location to store your notebook log 

##File Storage
In our assignment, we use our own S3 storage path. Hence, if you would like to run it on your own environment, please change the path to your own path.