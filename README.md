# HW0_Political_News_Analysis

# Intruduction 
My name is Evan Chiang.  
I currently a finance gradute student in NTU.   
I am a newbee to this code world, but I would try hard to face the challenges.  

## Installation
Download the repository
```bash
$ git clone https://github.com/A2Zntu/HW0_Political_News_Analysis.git
```

# Flow Chart
The draw.io flow chart is below.  
[Evan_Draw.io](https://www.draw.io/?lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1#G16-idBqn3LSorsOPPr3Or34L24_v0FYCX)    
The green boxes represent original functions and procedure.  
The orange boxes represent new functions or procedure.  

# Debug


# New Function 
* Word2vec model -- analysis the vector of words 

  + Using specific words to find similar words
     ```python
    def most_similar(w2v_model, words, topn=10):
        similar_df = pd.DataFrame()
        for word in words:
            try:
                similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
                similar_df = pd.concat([similar_df, similar_words], axis=1)
            except:
                print(word, "not found in Word2Vec model!")
        return similar_df
    ```
    ```python
    most_similar(w2v_model, ['蔡英文', '韓國瑜', '姚文智', '柯文哲', '高嘉瑜', '九二共識'], 10)
    ```
    Output
    ![image](https://github.com/A2Zntu/HW0_Political_News_Analysis/blob/master/politicians/words_similar.JPG)
    
  + Coordinate with `ExtraTreesClassifier` to classify words label
   ```python
       etree_w2v_tfidf = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    etree_w2v = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
   ```
   ```python 
   #training data
    X = np.array([['蔡英文', '柯文哲', '韓國瑜', '賴清德'],
                    ['九二共識', '一中各表', '兩岸一家親'],
                                        ['下午', '上午'], 
                        ['台北', '台中', '高雄', '桃園']])
    y = np.array(['政客', '兩岸議題', '時間', '地名'])
    # test data
    test_x = [['鄧小平'], ['高嘉瑜'], ['中午'], ['宜蘭'], ['王世堅'], ['傍晚'], ['統戰'], ['新竹']]
    ```
    Output
    ![image](https://github.com/A2Zntu/HW0_Political_News_Analysis/blob/master/politicians/ETC.JPG)

* add `idf`  in function `merge_one_day_news_dict(inverse = True)`.  
  + change the original `df_occur` to `df_idf`, in case we like to know the tf-idf of each word. 


