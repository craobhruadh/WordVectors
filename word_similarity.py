from gensim.models import KeyedVectors
import sys


def LoadModel():
	filename = 'glove_50d_wordvec.txt'
	gensim_model = KeyedVectors.load_word2vec_format(filename, binary=False)
	return gensim_model


def FindSimilarity(input_words):
	#input_words = ['larp', 'game', 'apple', 'pie']
	number_of_results = 10

	result = gensim_model.most_similar(positive=input_words, 
			topn=number_of_results)

	for word in result:
		print(word[0])
		
	#	words_to_avoid = ['snake', 'playstation']

	#	result = gensim_model.most_similar(positive=input_words, 
	#                                   negative=words_to_avoid,
	#                                   topn=number_of_results)
	


if __name__ == "__main__":
	gensim_model = LoadModel()



	input_words = sys.argv
	input_words = input_words[1:]

	FindSimilarity(input_words)


