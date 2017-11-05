import pandas as pd
import numpy as np
from featurizer import get_email_bodies, get_email_bodies_test, evaluate_on_test_set
from scipy.stats import uniform as sp_rand
from sklearn.feature_extraction.text import TfidfVectorizer as vec
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    train_data, train_classes = get_email_bodies()

    test_data = get_email_bodies_test()

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    train_df['body'] = train_data
    train_df['class'] = train_classes

    test_df['body'] = test_data
    print(test_data[0:9])
    # train, test = train_test_split(df, test_size=0.25)
    train = train_df
    test = test_df

    tf = vec(input='content', analyzer='word', min_df=0, max_df = 90, stop_words='english', sublinear_tf=False, decode_error='ignore',
                 ngram_range=(1,5),max_features=50000)
    
    input_to_model = tf.fit_transform(train['body'])

    param_grid = {'C': [10 ** -i for i in range(-5, 5)]}
    clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)

    targets = train['class'].values

    clf.fit(input_to_model.toarray(), targets)
    print(clf.score(input_to_model.toarray(), targets))
    print(evaluate_on_test_set(test, clf, tf))