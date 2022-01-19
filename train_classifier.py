# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')

def load_data(database_filepath):
    """Parameters:
    database_filename: string. Filename of SQLite database containing the cleaned message data.
       
    Returns:
    X: Dataframe containing messages used as the predictive column.
    Y: Dataframe containing the categories we are trying to predict.
    category_names: List of strings containing category names.
    """
     # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)

    # Create X and Y datasets
    X = df['message'] #messages used as the predictive column
    Y = df.drop(['id', 'message', 'genre'], axis = 1) #keep only the categories we are trying to predict
    
    # Create list containing category names
    category_names = Y.columns.values.tolist()
    
    return X, Y, category_names
    
    
def tokenize(text):
    """Normalize, tokenize, lemmatize and stem text string
    Parameters:
    text: String containing message for processing.
       
    Returns:
    stemmed: List of strings containing normalized and stemmed word tokens.
    """

    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Split text into words using NLTK
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    
    return stemmed

def build_model():
    """Build a k-nearest neighbor pipeline
    
    Parameters: None
       
    Returns:
    cv: Gridsearchcv object that transforms the data, creates the 
    model object and finds the optimal parameters.
    """
    
    knn_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    """parameters = {'clf__estimator__n_neighbors': [5, 10],
                  'clf__estimator__leaf_size':[30, 60]}"""
    
    parameters = {'clf__estimator__n_neighbors': [5, 10]}
    
    cv = GridSearchCV(knn_pipeline, param_grid = parameters, n_jobs=4, verbose=2)
    
    return cv

def display_accuracy(Y_test, Y_pred):
    """Parameters:
    Y_test: Dataframe containing test messages.
    Y_pred: Predicted categories from test data.
       
    Returns: None
    """
    y_pred_array = np.array(Y_pred)
    y_test_array = np.array(Y_test)
    labels = np.unique(Y_pred)
    confusion_mat = confusion_matrix(y_test_array.argmax(axis=1), y_pred_array.argmax(axis=1), labels=labels)
    accuracy = (Y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:\n", accuracy)
    

def evaluate_model(model, X_test, Y_test, category_names):
    """Parameters:
    model: Fitted model object.
    X_test: Dataframe containing test messages.
    Y_test: Dataframe containing test categories.
    category_names: List of strings containing category names.
    
    Returns: None
    """
    # Predict on test data
    Y_pred = model.predict(X_test)
    
    # display accuracy
    display_accuracy(Y_test, Y_pred)
    
    #display precision, recall and f-score
    print("\n")
    print(classification_report(Y_test, Y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """Create pickle file
    
    Parameters:
    model: Fitted model object.
    model_filepath: Filepath where fitted model is saved.
    
    Returns:
    None
    """
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()