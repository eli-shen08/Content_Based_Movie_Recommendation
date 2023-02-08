from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from flask import Flask,render_template,request
import pandas as pd
import numpy as np


movie_list = list(np.load('movie_list.npy',allow_pickle=True))
df = np.load('df.npy',allow_pickle=True)
similarity = np.load('similarity.npy',allow_pickle=True)

df1 = pd.DataFrame(df,columns=['title','index'])



app = Flask(__name__)


# vectorizer = TfidfVectorizer()

# df = pd.read_csv('movies.csv')

# selected_feature = ['genres', 'keywords', 'tagline', 'cast', 'director']
# for feature in selected_feature:
#     df[feature] = df[feature].fillna('')


# combined_features = df['genres'] +' '+ df['keywords'] +' '+ df['tagline'] +' '+ df['cast'] +' '+ df['director']

# feature_vectors = vectorizer.fit_transform(combined_features)
# similarity = cosine_similarity(feature_vectors)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    
    pred_list = []

    file = request.form['movie_name']
    # title_list = df['title'].tolist()
    find_close_match = difflib.get_close_matches(file,movie_list)
    close_match = find_close_match[0]
    index_of_movie = df1[df1['title']==close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_similarity_score = sorted(similarity_score, key=lambda x:x[1], reverse=True)
    i=1
    for movie in sorted_similarity_score[1:]:
        index = movie[0]
        title = df1[df1['index']==index]['title'].values[0]
        if i <=10:
            pred_list.append(title)
            i+=1


    # final = df['budget'][0]
    return render_template('predict.html', final=pred_list)




if __name__ == "__main__":
    app.run()
