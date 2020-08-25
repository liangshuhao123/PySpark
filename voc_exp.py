#######################################################################
## Vocabulary Exploration - Part A                                   ##
#######################################################################
from pyspark import SparkConf, SparkContext
import nltk

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
        
def extractSentences(record):
    try:
        index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, \
        sentence1_parse, sentence2_parse, sentence1, sentence2, label1, label2, label3, \
        label4, label5, gold_label = record.split("\t")
        return (sentence1, sentence2)
    except:
    	try:
    		index, promptID, pairID, genre, sentence1_binary_parse, \
    		sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1, \
    		sentence2 = record.split("\t")
    		return (sentence1, sentence2)
    	except:
    		return ()
    		
def extractWords(record):
    try:
        sentence1, sentence2 = record
        return(word.lower() for word in nltk.word_tokenize(sentence1 + ' ' + sentence2))
    except:
        return()

input_file_dev_mat = 's3://comp5349-slia7223/e-4AT2I9YSEAYUVL0BCTAB2KP5Y/dev_matched.tsv'
input_file_test_mat = 's3://comp5349-slia7223/e-4AT2I9YSEAYUVL0BCTAB2KP5Y/test_matched.tsv'
input_file_dev_mismat = 's3://comp5349-slia7223/e-4AT2I9YSEAYUVL0BCTAB2KP5Y/dev_mismatched.tsv'
input_file_test_mismat = 's3://comp5349-slia7223/e-4AT2I9YSEAYUVL0BCTAB2KP5Y/test_mismatched.tsv'

spark_conf = SparkConf().setAppName("Comp 5349 Assignment 2 Vocabulary Exploration")
sc=SparkContext.getOrCreate(spark_conf) 
text_file_dev_mat = sc.textFile(input_file_dev_mat)
text_file_test_mat = sc.textFile(input_file_test_mat)
text_file_dev_mismat = sc.textFile(input_file_dev_mismat)
text_file_test_mismat = sc.textFile(input_file_test_mismat)

pure_text_file_dev_mat = text_file_dev_mat.filter(deleteFirstRow)
pure_text_file_test_mat = text_file_test_mat.filter(deleteFirstRow)
pure_text_file_dev_mismat = text_file_dev_mismat.filter(deleteFirstRow)
pure_text_file_test_mismat = text_file_test_mismat.filter(deleteFirstRow)

sentences_dev_mat = pure_text_file_dev_mat.map(extractSentences)
sentences_test_mat = pure_text_file_test_mat.map(extractSentences)
sentences_dev_mismat = pure_text_file_dev_mismat.map(extractSentences)
sentences_test_mismat = pure_text_file_test_mismat.map(extractSentences)

words_dev_mat = sentences_dev_mat.flatMap(extractWords)
words_test_mat = sentences_test_mat.flatMap(extractWords)
words_dev_mismat = sentences_dev_mismat.flatMap(extractWords)
words_test_mismat = sentences_test_mismat.flatMap(extractWords)

words_mat = words_dev_mat.union(words_test_mat)
words_mismat = words_dev_mismat.union(words_test_mismat)

distinct_words_mat = words_mat.distinct()
distinct_words_mismat = words_mismat.distinct()
distinct_words_mat.persist()
distinct_words_mismat.persist()

num_distinct_words_mat = len(distinct_words_mat.collect())
num_distinct_words_mismat = len(distinct_words_mismat.collect())

common_words = distinct_words_mat.intersection(distinct_words_mismat)
# Answer for Question 1
num_common_words = len(common_words.collect())
print("number of common words: " + str(num_common_words))

# Answer for Question 2
num_unique_words_mat = num_distinct_words_mat - num_common_words
print("number of unique words for matched data: " + str(num_unique_words_mat))

# Answer for Question 3
num_unique_words_mismat = num_distinct_words_mismat - num_common_words
print("number of unique words for mismatched data: " + str(num_unique_words_mismat))

distinct_words_mat.unpersist()
distinct_words_mismat.unpersist()

#######################################################################
## Vocabulary Exploration - Part B                                   ##
#######################################################################
def extractGenreAndSentences(record):
    try:
        index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, \
        sentence1_parse, sentence2_parse, sentence1, sentence2, label1, \
        gold_label = record.split("\t")
        return (genre, sentence1, sentence2)
    except:
        return()

def extractWordsAndGenre(record):
    try:
        genre, sentence1, sentence2 = record
        return((word.lower(), genre) for word in nltk.word_tokenize(sentence1 + ' ' + sentence2))
    except:
        return()

def extractWordAndValueForCount(record):
    try:
        word, genre = record
        return(word, 1)
    except:
        return()
    
def extractCount(record):
    try:
        word, count = record
        return(count, 1)
    except:
        return()
    
def removeStopWords(record):
    try:
        word, genre = record
        if(word in stop_words.value):
            return False
        else:
            return True
    except:
        return()
    
input_file_train = 's3://comp5349-slia7223/e-4AT2I9YSEAYUVL0BCTAB2KP5Y/train.tsv'

text_train = sc.textFile(input_file_train)

pure_text_train = text_train.filter(deleteFirstRow)
genre_and_sentences_text_train = pure_text_train.map(extractGenreAndSentences)
words_and_genre_text_train = genre_and_sentences_text_train.flatMap(extractWordsAndGenre)
distinct_words_and_genre_text_train = words_and_genre_text_train.distinct()
words_and_value_for_count = distinct_words_and_genre_text_train.map(extractWordAndValueForCount)
words_and_count = words_and_value_for_count.reduceByKey(lambda x, y: x + y)
words_and_count.persist()

num_distinct_words = words_and_count.count()
pure_count = words_and_count.map(extractCount)
num_five_genre_words = pure_count.countByKey()[5]
num_four_genre_words = pure_count.countByKey()[4]
num_three_genre_words = pure_count.countByKey()[3]
num_two_genre_words = pure_count.countByKey()[2]
num_one_genre_words = pure_count.countByKey()[1]
# Answer for Question 1
percentages_of_words_appearing_in_five_genres = num_five_genre_words/num_distinct_words
percentages_of_words_appearing_in_four_genres = num_four_genre_words/num_distinct_words
percentages_of_words_appearing_in_three_genres = num_three_genre_words/num_distinct_words
percentages_of_words_appearing_in_two_genres = num_two_genre_words/num_distinct_words
percentages_of_words_appearing_in_one_genres = num_one_genre_words/num_distinct_words

print("percentages of words_appearing in one genres: " + str(percentages_of_words_appearing_in_one_genres))
print("percentages of words_appearing in two genres: " + str(percentages_of_words_appearing_in_two_genres))
print("percentages of words_appearing in three genres: " + str(percentages_of_words_appearing_in_three_genres))
print("percentages of words_appearing in four genres: " + str(percentages_of_words_appearing_in_four_genres))
print("percentages of words_appearing in five genres: " + str(percentages_of_words_appearing_in_five_genres))

# For Question 2
stop_words = sc.broadcast(['!!', '?!', '??', '!?', '`', '``', "''", '-lrb-', '-rrb-', \
                          '-lsb-', '-rsb-', '', '', '.', ':', ';', '"', "'", '?', '<', '>', \
                          '{', '}', '[', ']', '+', '-', '(', ')', '&', '%', '$', '@', '!', \
                          '^', '#', '*', '..', '...', "'ll", "'s", "'m", 'a', 'about', 'above', \
                          'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', \
                          "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', \
                          'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', \
                          'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", \
                          'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', \
                          'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", \
                          'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", \
                          'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', \
                          "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", \
                          'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', \
                          "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', \
                          'once', 'only', 'or', 'other', 'ought', 'our', 'ours ', 'ourselves', \
                          'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", \
                          "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', \
                          "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', \
                          'there', "there's", 'these', 'they', "they'd", "they'll", "they're", \
                          "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', \
                          'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", \
                          "we've", 'were', "weren't", 'what', "what's", 'when', "when's", \
                          'where', "where's", 'which', 'while', 'who', "who's", 'whom', \
                          'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', \
                          "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', \
                          'yourselves', '###', 'return', 'arent', 'cant', 'couldnt', 'didnt', \
                          'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'hes', 'heres', 'hows', \
                          'im', 'isnt', 'its', 'lets', 'mustnt', 'shant', 'shes', 'shouldnt', \
                          'thats', 'theres', 'theyll', 'theyre', 'theyve', 'wasnt', 'were', \
                          'werent', 'whats', 'whens', 'wheres', 'whos', 'whys', 'wont', \
                          'wouldnt', 'youd', 'youll', 'youre', 'youve'])

words_without_stopwords_and_count = words_and_count.filter(removeStopWords)
words_without_stopwords_and_count.persist()
num_distinct_words_without_stopwords = words_without_stopwords_and_count.count()
pure_count_without_stopwords = words_without_stopwords_and_count.map(extractCount)
num_five_genre_words_without_stopwords = pure_count_without_stopwords.countByKey()[5]
num_four_genre_words_without_stopwords = pure_count_without_stopwords.countByKey()[4]
num_three_genre_words_without_stopwords = pure_count_without_stopwords.countByKey()[3]
num_two_genre_words_without_stopwords = pure_count_without_stopwords.countByKey()[2]
num_one_genre_words_without_stopwords = pure_count_without_stopwords.countByKey()[1]
# Answer for Question 2
percentages_of_words_appearing_in_five_genres_without_stopwords = num_five_genre_words_without_stopwords/num_distinct_words_without_stopwords
percentages_of_words_appearing_in_four_genres_without_stopwords = num_four_genre_words_without_stopwords/num_distinct_words_without_stopwords
percentages_of_words_appearing_in_three_genres_without_stopwords = num_three_genre_words_without_stopwords/num_distinct_words_without_stopwords
percentages_of_words_appearing_in_two_genres_without_stopwords = num_two_genre_words_without_stopwords/num_distinct_words_without_stopwords
percentages_of_words_appearing_in_one_genres_without_stopwords = num_one_genre_words_without_stopwords/num_distinct_words_without_stopwords

print("percentages of words appearing in one genres without stopwords: " + str(percentages_of_words_appearing_in_one_genres_without_stopwords))
print("percentages of words appearing in two genres without stopwords: " + str(percentages_of_words_appearing_in_two_genres_without_stopwords))
print("percentages of words appearing in three genres without stopwords: " + str(percentages_of_words_appearing_in_three_genres_without_stopwords))
print("percentages of words appearing in four genres without stopwords: " + str(percentages_of_words_appearing_in_four_genres_without_stopwords))
print("percentages of words appearing in five genres without stopwords: " + str(percentages_of_words_appearing_in_five_genres_without_stopwords))

words_and_count.unpersist()
words_without_stopwords_and_count.unpersist()
