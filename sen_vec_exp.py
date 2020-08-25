#######################################################################
## Vocabulary Exploration - Part A                                   ##
#######################################################################
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import HashingTF,IDF,Tokenizer
from pyspark.sql import SQLContext
import tensorflow as tf
import tensorflow_hub as hub
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import udf

def deleteFirstRow(record):
    try:
        index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, \
        sentence1_parse, sentence2_parse, sentence1, sentence2, label1, label2, label3, \
        label4, label5, gold_label = record.split("\t")
        if(index == "index"):
        	return False
        else:
        	return True
    except:
        try:
            index, promptID, pairID, genre, sentence1_binary_parse, \
    		sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1, \
    		sentence2 = record.split("\t")
            if(index == "index"):
                return False
            else:
                return True
        except:
            try:
                index, promptID, pairID, genre, sentence1_binary_parse, \
                sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1, \
                sentence2, label1, gold_label = record.split("\t")
                if(index == "index"):
                    return False
                else:
                    return True
            except:
                return ()
                
def extractGenreAndSentencesForFlatmap(record):
    try:
        index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, \
        sentence1_parse, sentence2_parse, sentence1, sentence2, label1, \
        gold_label = record.split("\t")
        genre_sentences = [(genre, sentence1), (genre, sentence2)]
        return (genre_sentences)
    except:
        return()
    
def emb(record):
    url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(url)
    data = [row for row in record]
    sentence_list = [row[1] for row in data]
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(sentence_list))
    return [(data[i][0], embeddings[i]) for i in range(len(data))]

def toPrint(result):
    sum_list = [0,0,0,0,0]
    genres = []
    for item in result:
        predict, genre, count = item
        sum_list[predict] += count
        if genre not in genres:
            genres.append(genre)

    cluster = {}
    for item in result:
        predict, genre, count = item
        index = genres.index(genre)
        if predict not in cluster.keys():
            cluster[predict] = [0]*5
        cluster[predict][index] = count/sum_list[predict]*100
    print('{:15s}'.format(""), end = "")
    for g in genres:
        print('{:15s}'.format(g), end = "")
    print()
    i=0
    while i<len(cluster):
        pos = cluster[i].index(max(cluster[i]))
        print('{:15s}'.format(genres[pos]), end = "")
        for p in cluster[i]:
            print('{:15s}'.format(str(round(p,2)) + "%"), end = "")
        i+=1
        print()
        
def toList(record):
    genre, features = record
    return(genre, features.tolist())

input_file_train = 's3://comp5349-slia7223/e-4AT2I9YSEAYUVL0BCTAB2KP5Y/train.tsv'

spark_conf = SparkConf().setAppName("Comp 5349 Assignment 2 Sentence Vector Exploaration")
sc=SparkContext.getOrCreate(spark_conf) 
sqlContext = SQLContext(sc)

text_train = sc.textFile(input_file_train)

pure_text_train = text_train.filter(deleteFirstRow)
genre_and_sentences_after_flatmap = pure_text_train.flatMap(extractGenreAndSentencesForFlatmap)
genre_and_sentences_after_flatmap.persist()

# TFIDF
tfidf_dataFrame = genre_and_sentences_after_flatmap.toDF(["genre","sentence"])
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tfidf_words_data = tokenizer.transform(tfidf_dataFrame)

hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=512)
tfidf_featurized_data = hashing_tf.transform(tfidf_words_data)

idf_model = IDF(inputCol="rawFeatures", outputCol="features").fit(tfidf_featurized_data)
tfidf_rescaled_data = idf_model.transform(tfidf_featurized_data)
tfidf_genre_features = tfidf_rescaled_data.select("genre", "features")

# Confusion matrix for TFIDF
tfidf_kmeansmodel = KMeans().setK(5).setFeaturesCol('features').setPredictionCol('prediction').fit(tfidf_genre_features)
tfidf_predictions = tfidf_kmeansmodel.transform(tfidf_genre_features).select("prediction", "genre")
tfidf_res = tfidf_predictions.groupBy(['prediction', 'genre']).count().collect()
print("Confusion matrix for TFIDF:")
toPrint(tfidf_res)
print()

#######################################################################
## Vocabulary Exploration - Part B                                   ##
#######################################################################

# pretrained
pretrained_genre_features = genre_and_sentences_after_flatmap.mapPartitions(emb)
pretrained_dataFrame = pretrained_genre_features.map(toList).toDF(["genre","features"])

new_schema = ArrayType(DoubleType(), containsNull=False)
udf_foo = udf(lambda x:x, new_schema)
pretrained_dataFrame = pretrained_dataFrame.withColumn("features",udf_foo("features"))

# Confusion matrix for pretrained
pretrained_kmeansmodel = KMeans().setK(5).setFeaturesCol('features').setPredictionCol('prediction').fit(pretrained_dataFrame)
pretrained_predictions = pretrained_kmeansmodel.transform(pretrained_dataFrame).select("prediction", "genre")
pretrained_res = pretrained_predictions.groupBy(['prediction', 'genre']).count().collect()
print("Confusion matrix for pretrained:")
toPrint(pretrained_res)

genre_and_sentences_after_flatmap.unpersist()
