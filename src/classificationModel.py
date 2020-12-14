import numpy as np
import csv
from sklearn.svm import SVC

class faculty_directory_classification:

    def __init__(self):
        self.train_file = './trainingData/trainingDataSet.csv'
        self.vocabulary = []
        self.trainedModel = None

    def read_file(self, file_path):
        rows = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)
        return rows

    def tokenize_url(self, url):
        remove_words = ['https://', 'http://']
        hint_words = ['faculty', 'Faculty', 'people', 'People', 'staff', 'Staff', 'members', 'Members']
        for remove_word in remove_words:
            url = url.replace(remove_word, '')

        # Remove empty string
        tokens = [token for token in url.split('/') if token]

        for token in tokens:
            for hint_word in hint_words:
                if hint_word in token and (len(token) != len(hint_word)):
                    tokens.append(hint_word)

        return tokens

    def train(self):
        # Read rows from training csv file
        rows = self.read_file(self.train_file)

        # Exact urls and labels
        urls = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        
        # Build feature matrix for SVM
        feature_matrix = self.build_feature_matrix(urls)

        # Train the model using support vector machine
        SVM = SVC(kernel='linear')
        self.trainedModel = SVM.fit(feature_matrix , labels)

    def predict(self, url):
        if (self.trainedModel is None):
            self.train()
   
        feature_matrix = np.zeros([1, len(self.vocabulary)], dtype=np.int)
        url_token = self.tokenize_url(url)
        feature_matrix[0] = self.build_url_word_matrix(url_token)
        return self.trainedModel.predict(feature_matrix)[0]

    def build_feature_matrix(self, urls):
        # Tokenize urls
        url_tokens = [self.tokenize_url(url) for url in urls]

        # Build vocabulary list
        for tokens in url_tokens:
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary.append(token)
        
        # Build feature matrix
        feature_matrix = np.zeros([len(url_tokens), len(self.vocabulary)], dtype=np.int)
        for index, url_token in enumerate(url_tokens):
            feature_matrix[index] = self.build_url_word_matrix(url_token)
     
        return feature_matrix

    def build_url_word_matrix(self, url_token):
        matrix = np.zeros(len(self.vocabulary), dtype=np.int)
        for word in url_token:
            if word in self.vocabulary:
                index = self.vocabulary.index(word)
                matrix[index] += 1

        return matrix

if __name__ == "__main__":
    model = faculty_directory_classification()
    model.train()

    test_file = './trainingData/testDataSet.csv'
    rows = model.read_file(test_file)
    urls = [row[0] for row in rows]
    labels = [row[1] for row in rows]

    test_labels = []
    for url in urls:
        prediction = model.predict(url)
        if (prediction == '0'):
            print('0 label is: ' + url)
        test_labels.append(prediction)
    
    # Calculate precision
    positive = 0
    truePositive = 0
    for index, label in enumerate(test_labels):
        trueLabel = labels[index]
        if (label == '1'):
            positive += 1
            if(label == trueLabel):
                truePositive += 1
        
    precision = truePositive / positive
    
    # Calculate recall
    labelPositive = 0
    truePositive = 0
    for index, label in enumerate(test_labels):
        trueLabel = labels[index]
    
        if (trueLabel == '1'):
            labelPositive += 1
        if (label == '1' and label == trueLabel):
            truePositive += 1
        
    recall = truePositive / labelPositive


