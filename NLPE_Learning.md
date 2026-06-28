# Part B: Researcher's Playground
 In part B I applied advanced Natural Language Processing techniques I had not learned before, which is beyond what I learnt in class (Tf-DIF and Dense models), to improve accuracy and better understand the sentence context.

## Techniques chosen: Pretrained GloVe word embeddings and bidirectional LSTM
 I chose to explore on GloVe word embeddings and bidirectional LSTM. I learnt that GloVe helps the model understand the meaning of words while BiLSTM helps the model understand the sentence context and word order.

## Technique One: GloVe Word Embeddings

### 1. The Concept:
 For part B I explored on GloVe word embeddings, an nlp technque used to convert words into dense vectors.( dense vector is a list of numbers used to represent the meaning a word in ai and machine learning.) Unlike TF-DIF that focuses on word importance, GloVe can understand the relationships between words with similar meanings. 
 
 I used this so that the model would better understand emotions in text.

### 2. The Implementation:
 I implemented GloVe embeddings by loading a pretrained word vector from the `glove.6b.100d.txt` file and using them in the embedding layer.

 Code Snippet:

```
EMBEDDING_DIM = 100

embeddings_index = {}

with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
```
### 3. The learning:
 GloVe works by learning the relationships between words from a very large amount of text data. I learnt that words with similar meanings such as “happy”, “joyful” and “excited” would have similar vector representations because they often appear in similar contexts.

 Before techniques like GloVe existed, many NLP systems used methods such as one-hot encoding. In one-hot encoding, every word is treated as completely separate and unrelated. For example, even though “happy” and “joyful” have very similar meanings, the computer would treat them as entirely different words with no relationship at all.

 I learnt that one-hot encoding is similar to using a giant checklist, where each word is simply marked as present or absent. While this works, it becomes inefficient because the vectors are extremely large and sparse, and the model cannot understand semantic meaning between words.

 GloVe embeddings improved this by converting words into dense vectors, which are smaller and more meaningful numerical representations. Instead of memorizing words separately, the model can understand relationships and similarities between words. This makes the embedding process faster, more efficient and more accurate for NLP tasks.

 Through this experiment, I also learnt that GloVe embeddings help the model understand emotions in text more effectively because emotionally similar words are placed closer together in vector space. Compared to TFIDF, which mainly focuses on word importance and frequency, GloVe focuses more on semantic meaning and relationships between words.

 In my experiment, my baseline model achieved around 0.81 accuracy. After implementing GloVe embeddings, the validation accuracy improved to around 0.90. This showed that understanding semantic relationships between words significantly improved the model’s ability to classify emotions correctly.

## Technique Two: Bidirectional LSTM(BiLSTM)

### 1. The concept:
 For my second technique introduced in my model I explored on bidirectional LSTM which is a sequence based neural network ( processes information in order, one part at a time). Unlike Dense neural networks, BiLSTM can understand sentence order and the context. I used this because I think that emotions in text depends heavily on the sentence structure.

### 2. Implementation:
 I implemented BiLSTM using TensorFlow/Keras
 
 Code Snippet:
```
model.add(Bidirectional(LSTM(64)))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
```

### The learning: 
 BiLSTM works by reading sentences in both forward and backward directions. Which helps the model understand context better. For example if my sentence was i am happy versus i am not happy the word not changes the sentence completely.

 I also learnt that stopwords like 'not', 'never', 'very' is very important for NLP sequence because they would affect the sentence meaning. I also introduced dropout to reduce overfitting by randomly disabling neurons during training. In my experiment i combined both GloVe and BiLSTM together which helped the model improved by a 0.08 in accuracy. This shows that BiLSTM improved my models understanding of the sentence context and emotions.

## Technique Three: Early stopping & Dropout

### 1. The concept:
 For my third technique which is Early stopping and Dropout, both are techniques used to prevent overfitting during training . It helps to automatically stop training when validation performance stops improving.(refers to how well the model performs using the validation data during training.) 

### 2. Implementation:
 I implemented Early stopping using a keras callback which is a tool that runs automatically during model training to help monitor or control the training process.

 Code Snippet for Early stopping:
```
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
```
 Code Snippet for Dropout:
 ```
    from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_LENGTH,
            trainable=False
        )
    )

    model.add(Bidirectional(LSTM(64, return_sequences=False)))

    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(len(emotion_order), activation='softmax'))

    model.summary()
 ```

### 3. The learning:
 Early stopping works by monitoring validation loss during training. And I learnt that training too many epochs would cause the model to memorise the training data instead of what I want which is learning general patterns of the data sets.

 Although it sounds similar to dropout where they both are used to reduce overfitting, early stopping is used to monitor the validation performance and would automatically stop training when the model stops improving, which would prevent it from learning too much extra things like random or unimportant patterns from the training data. And dropout is a tool which would turn off some of the neurons during training so the model does not rely too much on specific features. Which would help the model generalise better.

 Specific features can refer to certain words or phrases like 'furious' 'i hate' or 'i am so happy'

# Overall Learning

 Through this Researcher’s Playground, I gained a much deeper understanding of how advanced Natural Language Processing techniques can improve text classification models beyond the basic TF-IDF and Dense neural network approaches that I learnt in class.

 One of the most important things I learnt was that understanding language is not only about identifying important words, but also understanding the meaning, relationships and context between words inside a sentence.

 From GloVe word embeddings, I learnt that words can be represented as dense vectors that capture semantic meaning. Unlike TF-IDF which mainly measures word importance and frequency, GloVe embeddings allow the model to understand relationships between words with similar meanings such as “happy”, “joyful” and “excited”. This improved the model’s ability to detect emotions more accurately because the model could better understand the meaning behind the text instead of treating every word separately.

 From Bidirectional LSTM (BiLSTM), I learnt the importance of sentence order and context in NLP tasks. Unlike Dense neural networks which process text without sequence understanding, BiLSTM reads sentences in both forward and backward directions, allowing the model to better understand the meaning of a sentence. For example, phrases like “I am happy” and “I am not happy” contain similar words but completely different meanings because of the word “not”. This showed me that stopwords such as “not”, “never” and “very” are actually very important in sequence-based NLP models because they strongly affect sentence meaning and emotion.

 I also learnt the importance of reducing overfitting during training. Through Early Stopping and Dropout, I understood that training a model for too long can cause it to memorize the training data instead of learning general patterns. Early Stopping helped prevent unnecessary training by monitoring validation loss and stopping the training process once performance stopped improving. Dropout helped reduce overfitting by randomly disabling neurons during training so that the model would not rely too heavily on specific words or features.

 Overall, combining GloVe embeddings with BiLSTM produced the best performance because the model could both understand the meaning of words and the context of sentences at the same time. My validation accuracy improved showing that these advanced NLP techniques significantly improved the model’s understanding of emotions in text.

 This Researcher’s Playground helped me better understand how modern NLP models process human language and showed me that context, sequence and semantic meaning are all very important for accurate emotion classification.
