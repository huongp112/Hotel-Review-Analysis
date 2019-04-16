# Batch imports of text processing libraries
import scipy as sp
import nltk
import string
global string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
'''CountVectorizer, and TfidfVectorizer are what we will use 
to convert text into a matrix of token counts
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import pandas as pd # Import pandas library
# Import the clean csv file, after prepring in the Jupyter Notebook
path = r'C:\Users\Huong Pham\Documents\Graduate School\Winter 2019\Independent Study\\'
data = 'new_hotels.csv'
df= pd.read_csv(path+data)
print (df.shape)

# create a new DataFrame that only contains the rounded numbers of Reviewer Scores
df1 = df[(df.Reviewer_Score==3) | (df.Reviewer_Score==5) | (df.Reviewer_Score==7) | (df.Reviewer_Score==9)]
print (df1.shape)
print df1.head()

# importing the default list of stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['a','after','according','accordingly','act','abst','adj','actually','across','added','already','along','also','almost','ah','afterwards','although','always','among','amongst','anyway','anyways','anyhow','approximately','apparently','anybody','anyone','anywhere','anymore','arise','around','aside','ask','asking','away','become','became','becomes','becoming','back','be','begin','begins','beginning','beforehand','believe','beside','besides','behind','beyond','biol','brief','briefly','ca','came',"can't",'cannot','cause','causes','certain','certainly','come','comes','contain','containing','contains','therefore','able','date','different','doing','done','downwards','due','ed','edu','effect','eg','eight','eighty','either','else','elsewhere','end','ending','enough','especially','et','et-al','etc','even','ever','every','everybody','everyone','everywhere','everywhere','ex','except','far','fifth','first','five','fix','followed','following','follows','for','former','formerly','forth','found','four','furthermore','gave','get','gets','gets','getting','give','given','gives','giving','go','goes','gone','got','gotten','had','happens','hardly','hed','hence','hereafter','hereby','herein','heres','hereupon','hes','hi','hid','hither','how','howbeit','however','hundred','id','ie','if',"i'll",'im','immediately','immediate','inc','indeed','index','information','instead','into','invention','inward','itd',"it'll",'itself',"i've",'keep','keeps','kept','kg','km','cm','know','known','knows','largely','last','lately','later','latter','latterly','least','less','lest','let','lets','like','liked','likely','line','little','look','looks','looking','ltd','made','mainly','make','makes','many','may','maybe','mean','means','meantime','meanwhile','merely','mg','might','million','ml','moreover','mostly','mr','mrs','much','mug','must','na','name','namely','nay','nd','near','nearly','necessarily','necessary','need','needs','neither','never','nevertheless','next','nine','ninety','no','nobody','none','nonetheless','normally','nos','not','noted','nothing','nowhere','obtain','obtained','obviously','often','oh','ok','old','omitted','one','ones','onto','ord','others','otherwise','outght','out','outside','overall','owing','page','pages','part','particularly','particular','past','per','perhaps','placed','please','plus','possible','possibly','potentially','pp','predominantly','present','previously','primarily','promptly','provides','put','que','quickly','quite','qv','r','ran','rather','rd','re','readily','really','recent','recently','ref','refs','regarding','regardless','regards','related','relatively','research','respectively','resulted','resulting','results','right','run','said','saw','say','saying','says','sec','section','see','seeing','seem','seemed','seeming','seems','seen','self','selves','sent','seven','several','shall','shed','shes','show','showed','shown','showns','shows','similar','similarly','since','six','slightly','somebody','somehow','someone','somethan','something','sometime','sometimes','somewhat','somewhere','soon','sorry','specifically','specified','specify','specifiying','still','stop','sub','suggest','sup','sure','take','taken','taking','tell','tends','th','thats',"that've",'thence','thereafter','thereby','thered','therein',"there'll",'thereof','therere','theres','thereto','thereupon',"there've",'theyd',"they'll",'theyre',"they've",'think','thou','though','thoughh','thousand','through','throug','throughout','thru','thus','till','tip','together','toward','towards','tried','try','tries','truly','try','trying','ts','twice','two','un','unlike','unless','unfortunately','unlikely','until','until','unto','upon','ups','us','use','used','useful','usefulness','uses','using','usually','value','various','via','viz','vol','vols','vs','want','wants','wasnt','way','wed',"we'll",'went','werent',"we've",'whatever',"what'll",'whats','whence','whenever','whereafter','whereas','whereby','wherein','wheres','whereupon','wherever','whether','whim','whither','whod','whoever','whole',"who'll",'whos','whose','widely','willing','wish','within','without','wont','words','would','www','yes','yet','youd','zero'])

#Remove the extended list of stop words
df1.Review = df1.Review.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#Remove punctionation 
df1['Review'] = df1['Review'].str.replace('[^\w\s]','')
#df1.Review = df1.Review.apply(lambda x: x.translate(None, string.punctuation))

#Remove numbers
df1['Review'] = df1['Review'].apply(lambda x: x.translate(None, string.digits))

#Convert to lower case
df1['Review'] = df1['Review'].apply(lambda x: x.lower())

#Replace special characters by ''
df1['Review'] = df1['Review'].str.replace("[^\w\s]",'')

#Save dataframe df1 to a CSV file after cleaning up the text. Will use this CSV for Sentimental Analysis later
path_d = r'C:\Users\Huong Pham\Documents\Graduate School\Winter 2019\4 classes\\'
df1.to_csv(os.path.join(path_d,'clean_data.csv'))

# Experiment with special text mining techniques
# TOKENIZATION
# define X and y
X = df1.Review
y = df1.Reviewer_Score

# split the new DataFrame into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#let's look at the vector options we can set
vect = CountVectorizer() #instantiate CountVectorizer
print vect

# use CountVectorizer to create document-term matrices from X_train and X_test
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

print X_train_dtm.shape

# first 50 features
print vect.get_feature_names()[:50]

# last 50 features
print vect.get_feature_names()[-50:]

# create a set of unigrams, bigrams, and trigrams
vect1 = CountVectorizer(ngram_range=(1,1))
vect2 = CountVectorizer(ngram_range=(1,2))
vect3 = CountVectorizer(ngram_range=(1,3))

X_train_dtm1 = vect1.fit_transform(X_train)
X_train_dtm2 = vect2.fit_transform(X_train)
X_train_dtm3 = vect3.fit_transform(X_train)

# Checking the shape of ech set of grams
X_train_dtm1.shape
X_train_dtm2.shape
X_train_dtm3.shape

# PREDICTING THE REVIEW SCORES 3 & 10
# use default options for CountVectorizer
vect = CountVectorizer()

# define a function that accepts a vectorizer and calculates precision
from sklearn import svm
from sklearn import metrics
def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print 'Features: ', X_train_dtm.shape[1]
    X_test_dtm = vect.transform(X_test)
    clf = svm.SVC(gamma=0.001, C=1)
    clf.fit(X_train_dtm, y_train)
    y_pred_class = clf.predict(X_test_dtm)
    print 'Accuracy: ', metrics.accuracy_score(y_test, y_pred_class)

# include unigrams, default for CountVectorizer()
vect1 = CountVectorizer(ngram_range=(0, 1))
tokenize_test(vect1)

# include bigrams
vect2 = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect2)

# include trigrams
vect3 = CountVectorizer(ngram_range=(1, 10))
tokenize_test(vect3)

# include 1-grams and 2-grams, and limit the number of features
vect4 = CountVectorizer(ngram_range=(1, 2), max_features=100000)
tokenize_test(vect4)

# include 1-grams and 2-grams, and limit the number of features
vect5 = CountVectorizer(ngram_range=(1, 2), max_features=30000)
tokenize_test(vect5)

vect6 = CountVectorizer(stop_words=stopwords)
vect6 = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect6)

#STEMMING & LEMMATIZATION
'''
Stemming
What: Reduce a word to its base/stem/root form
Why: Often makes sense to treat related words the same way
'''

from nltk.stem import PorterStemmer
st = PorterStemmer()
df1['Review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

# Use st=PorterStemmer the feature extraction function
vect7 = CountVectorizer(st)
tokenize_test(vect7)

'''
Lemmatization
What: Derive the canonical form ('lemma') of a word
Why: Can be better than stemming
'''

# define a function that accepts text and returns a list of lemmas
from textblob import TextBlob, Word
def split_into_lemmas(text):
    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]

# use split_into_lemmas as the feature extraction function 
vect4 = CountVectorizer(analyzer=split_into_lemmas)
tokenize_test(vect4)
