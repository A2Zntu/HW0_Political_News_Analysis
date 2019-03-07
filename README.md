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
    ![image](https://github.com/A2Zntu/HW0_Political_News_Analysis/blob/master/politicians/words_similar.JPG)
    
  + Train `TfidfVectorizer` model to classify the new word

* add `idf`  in function `merge_one_day_news_dict(inverse = True)`.  
  + change the original `df_occur` to `df_idf`


