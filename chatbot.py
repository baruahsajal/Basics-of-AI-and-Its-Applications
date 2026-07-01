import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class AIChatbot:
    def __init__(self, intents_file='intents.json'):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.load(open(intents_file))
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']
        self.model = None
        self._prepare_data()
        
    def _prepare_data(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
                    
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
    def _create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = []
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
                
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
            
        random.shuffle(training)
        training = np.array(training, dtype=object)
        return list(training[:, 0]), list(training[:, 1])
    
    def build_model(self):
        train_x, train_y = self._create_training_data()
        
        self.model = Sequential([
            Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(train_y[0]), activation='softmax')
        ])
        
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        
        self.model.save('chatbot_model.h5')
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))
        print("Model trained and saved successfully!")
