import pandas as pd
from featurizer import get_email_bodies, evaluate_on_test_set
from sklearn.feature_extraction.text import TfidfVectorizer as vec
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data, classes = get_email_bodies()
    df = pd.DataFrame()
    
    df['body'] = data
    df['class'] = classes
    train, test = train_test_split(df, test_size=0.25)
    
    tf = vec(input='content', analyzer='word', min_df=0, max_df = 90, stop_words='english', sublinear_tf=False, decode_error='ignore',
                 max_features=20000)
    
    input_to_model = tf.fit_transform(train['body'])
    
    # classifier = MultinomialNB()
    classifier = LogisticRegression()
    # classifier = RandomForestClassifier()
    # classifier = DecisionTreeClassifier()

    targets = train['class'].values
    classifier.fit(input_to_model, targets)
    print(classifier.score(input_to_model, targets))

    print(evaluate_on_test_set(test, classifier, tf))