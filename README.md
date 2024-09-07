# Natural-language-processing-NLP-Zero-to-Hero
In This notebook i was practicing to get the core intuition  of NLP concepts from Tokenizers to Transfomer Architecture
#From the findamental of natural language processing ,my journey of learning nlp ,The series include

# Tokenization
**1) TOKENIZATION**
Tokenization is the process of breaking down a corpus into tokens. The procedure might look like segmenting a piece of text into sentences and then further segmenting these sentences into individual words, numbers and punctuation, which would be tokens.
# Pre-Processing
The pre-processing steps includes case folding i.e converting every token to uniform lower case or upper case

stop word removal

Lemmatization

Stemming

Part of speech tagging

Named-Entity Recognition
# Bag of words and similarity
After tokenization and pre-processing, we are left with variable length sequences of text, but the problem is machine learning algorithms require fixed length vectors of numbers.

The simplest approach to overcome this is by using a bag-of-words, which simply counts how many times each word appears in a document. It's called a bag because the order of the words is ignored - we only care about whether a word appeared or not.

For now, we'll focus on the binary version of a bag-of-words. This just indicates whether a word appeared or not, ignoring word order and word frequency. Each row in a binary bag-of-words matrix corresponds to a single document in the corpus. Each column corresponds to a token in the vocabulary. Note that the order of the tokens isn't important but it does need to be fixed beforehand when building the vocabulary.

To construct the matrix, we place a 1 in entry (i,j) if and only if the j-th token appears in the i-th document and a 0 otherwise.

For a general bag-of-words, the (i,j) entry would instead be the frequency of the j-th token in the i-th document (but we will see there are better ways to encode frequency later).

Similiarity;Vector space search We have gone from thinking of documents as a sequence of words to points in a multi-dimensional vector space. Importantly, the dimension of this space if fixed, i.e. each vector has the same length.

here are many metrics we could use to measure how 'close' two points are. For example, we could consider using the Euclidean distance, Manhattan distance or even Hamming distance. However, if documents in the same corpus have very different lengths, or the vocabulary is extremely large, these metrics become less reliable.

Instead, in the NLP domain it is much more common to use Cosine Similarity. This measures the cosine of the angle between any two points (more precisely their vectors starting from the origin). The closer the score 1, the smaller the angle between the vectors and the more similar the documents are.

n-gram One way to get around the problem of losing word order information is to use n-grams. This is when we group chunks of n tokens together to behave as if they were a single token
# TF-IDF and Document search
TF-IDF stands for Term Frequency - Inverse Document Frequency and is made up of two components. The first is the term frequency.

Whilst some people define the term frequency to be the relative frequency, it is more common to use the raw frequency of the token/term in document .

However, some documents may be much longer than others and so will naturally have higher frequencies across the board. For this reason, it is standard practice to apply the log-transform to reduce this bias.

The second part of TF-IDF is the inverse document frequency. This is the part that will emphasise the more important words
# Naive Bayes and text classification

# LDA and word Embeddings
So far we've seen how to vectorize documents of text either through Bag-of-Words or TF-IDF. While these approaches work well for simple NLP tasks like classification, they have the drawback that they don't capture any relationships between words.

In this notebook, we will see how to vectorize individual words via static embeddings in order to capture word meaning. For example, this will let us model that "brother" and "sister" are more similar in meaning than "tree" and "car". Note that in a later notebook we will also cover contextual embeddings where the embedding can change depending on the context.

The simplest way to vectorize a set of words, is to use one-hot encoding. This maps each word into a vector with length equal to the size of the vocabulary. The vector is completely filled with 0's except for a single entry, which has a 1 correspoding to the index of the word in the vocabulary.

This is a pretty terrible way to vectorize words, not only because it is very memory inefficient but also because there is no relationship between words.

In particular, if the voculary contains 10,000 words, then each vector has length equal to 10,000. Furthermore, any two distinct vectors will always have a dot product equal to 0, corresponding to no similarity.

A better way then would be to represent words as shorter and denser vectors that capture some meaning between words. And this is what an embedding aim to do.

An embedding is simply a representation of an object (e.g. a word, movie, graph, etc) as a vector of real numbers. It embeds an object into a high-dimensional vector space.

Word2vec is a way of learning word embeddings by training shallow neural networks on either the continuous bag-of-words or the skip-gram task. We will discuss both of these shortly. The dimension of the resulting dense vectors is usually between 50-1000 and a common value is 300.

The amazing thing to come out of Word2vec is not only that similar words are close together, but that we can perform addition and subtraction on these word vectors. For example, “king” - “man” + “woman” = "queen". That is, if we take the vectors for king, man, woman and add/subtract them in this way, we will end up with a vector close to the one corresponding to queen. This means that these vectors can capture very precisely abstract concepts (like gender and royalty) without any input from us.

Note that the values in each vector don't necessarily correspond to anything like fantasy, strategy, etc like in the video game example from before. The algorithm learns the best representation even if the numbers don't correspond to anything tangible. The most important thing though is that similar words are close together.

Word2vec is a collection of two algorithms that each learn word vectors indirectly through a word prediction task. One option is Continuous Bag-of-Words (CBOW) and the other is skip-gram.

Training accurate word embeddings from scratch takes a lot of computational resources. If we want to use word vectors in our models then we have a few options depending on the requirements of our task.

Use pre-trained word vectors as model inputs and keep them constant during training. Use pre-trained word vectors as model inputs and allow the model to tune them during training. Train word embeddings from scratch at the same time as training the model. Option 1 will be the fastest and will usually lead to very good results. Option 2 can produce marginally better results if you are willing to take longer to train your model. Option 3 can be useful if you have lots of training data and want to train a model for a very specific task that pretrained embeddings don't perform well on. We will explore these different options with code in the next section.

Word vector

We're going to be using pre-trained word vectors via the Gensim library, which we came across last time. These particular vectors are known as the Google News vectors as they were trained on a 3 billion word Google News corpus in 2015. In total, there are 3 million, 300-dimension vectors.

GoogleNews-vectors-negative300 - https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

Spotify App Reviews -https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022
# Machine translation Attention

# Transfomers and RNN_LMs
In this notebook, we are going to go through what a transformer is, build one from scratch and then download a pre-trained model from hugging face and fine-tune it for a text classification problem.

We've already discussed how the inputs are first passed through an embeddding layer and combined with a positional encoding before being going through the first encoder block.

The encoder block is made up of a multi-head attention layer followed by a two layered feed forward neural network applied pointwise. In the paper, the dimensions of the layers were 512 for the input layer, 2048 for the hidden layer and 512 for the output layer. Furthermore, the encoder was made up of 6 encoder blocks stacked on top of each other.

Quickly after the original paper came out, a whole host of different transformers were released. These can be categorized into encoder only (e.g. BERT), decoder only (e.g. GPT) and encoder-decoder (e.g. T5). Let's briefly discuss the main models and see how they are different to each other.

BERT stands for Bidirectional Encoder Representations from Transformers and has up to 340 million parameters. It was trained by a team at Google using masked language modelling, i.e. guess the missing word in a sentence.

DistilBERT is a distilled version of BERT that achives 97% performance of BERT while being 60% faster. It can be used for text classification and sequence labelling tasks.

RoBERTa is similar to BERT but was trained for much longer and with better design choices. It significantly exceeded the performance of BERT on some tasks.

XLM is a cross-lingual language model variant of BERT that smashed several benchmarks on multilingual and translation tasks.

ALBERT is an optimized version of BERT that used fewer parameters thus making it possible to train larger models with fewer parameters.

GPT stands for Generative Pre-trained Transformer and is a family of decoder only transformers developed by OpenAI and used for text generation tasks. These were trained on next word prediction and are able to produce coherent text thanks to its enormous model size (GPT-3 has 175 billion parameters!).

T5 is a high-performing encoder-decoder transformer based on the original paper and is used for sequence-to-sequence tasks.

Building a transformer from scratch

To reinforce our understanding of transformers, lets write the code to implement one from scratch. We wont't be able to train it on our own but we will later see how was can use transfer learning to fine tuning a pre-trained transformer for our applications.

Sentiment Analysis Company Reviews -



