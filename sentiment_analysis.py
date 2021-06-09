from nltk.corpus import movie_reviews, stopwords
from nltk.probability import FreqDist
import string
import random

class PreprocessText:
    """Handle the text preparation before classification"""
    def __init__(self):
        self.reviews = []
        self.word_features = []

    def _extract_word_features(self):
        """Extract 3000 of the most common words from the movie reviews"""
        all_words = []
        for w in movie_reviews.words():
            all_words.append(w.lower())
        all_words = FreqDist(self._remove_useless_words(all_words))
        self.word_features = list(all_words.keys())[:3000]

    def _find_word_features(self, data: list) -> dict:
        """Returns only words contained inside the most used words"""
        words = set(data)
        features = {}
        for w in self.word_features:
            features[w] = (w in words)
        return features

    def _remove_useless_words(self, unfiltered_words: list) -> list:
        """Get rid of non useful words by removing them from the tokenized list imported
            also performs case normalisation"""
        useless_words = stopwords.words("english") + list(string.punctuation)
        filtered_words = [word.lower() for word in unfiltered_words if word not in useless_words]
        return filtered_words
        
    def _import_movie_reviews(self):
        """Importing the data into a list, each index consist of a tuple :
            - first index of this tuple is the tokenized words of a review
            - second index is his label (pos/neg)"""
        for category in movie_reviews.categories():
            for fileid in movie_reviews.fileids(category):
                tokenized_words = movie_reviews.words(fileid)                                           # Get a movie reviews by it's id, tokenized
                filtered_words = self._remove_useless_words(tokenized_words)                            # Remove useless words such as stopwords and punctuation
                labeled_reviews = (list(filtered_words), category)                                      # Create a tuple with filtered words and their corresponding category
                self.reviews.append(labeled_reviews)                                                    # Append to the full list of reviews
    
    def execute(self) -> tuple:
        """Entrypoint function returns a tuple with training and testing set ready for
            classification"""
        self._import_movie_reviews()
        random.shuffle(self.reviews)

        self._extract_word_features()
        featuresets = [(self._find_word_features(review), category) for (review, category) in self.reviews]

        features_nbr = len(featuresets) // 3
        training_set = featuresets[features_nbr:]
        testing_set = featuresets[:features_nbr]
        return (training_set, testing_set)

pt = PreprocessText()

training_set, testing_set = pt.execute()