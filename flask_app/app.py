import gpt_2_simple as gpt2
from flask import Flask, render_template, url_for, request
import pickle
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    '''
    
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)
    '''

    if request.method == 'POST':

        prefix = 'neural'  # None is default
        prefix = request.form['message']
        '''
        text = gpt2.generate(sess,
                             length=40,
                             temperature=0.7,
                             prefix=prefix,
                             nsamples=1,
                             batch_size=1,
                             return_as_list=True
                             )

        t = text[0].title()
        t = t.replace('<|Startoftext|>', '').replace(
            '\n', '')  # remove extraneous stuff
        t = t[:t.index('<|Endoftext|>')]  # only get one title
        '''
        t = prefix
        return render_template('result.html', prediction=t)


'''
@app.route('/predict', methods=['POST'])
def predict():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html', prediction = my_prediction)
'''


if __name__ == '__main__':
    app.run(debug=True)
