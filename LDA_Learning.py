from warcio.archiveiterator import ArchiveIterator
from langdetect import detect
import re
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
import subprocess
import gzip
import shutil
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Download the latest text file from common crawl
url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-50/segments/1700679518883.99/wet/CC-MAIN-20231211210408-20231212000408-00899.warc.wet.gz"
filename_gz = url.split('/')[-1]
filename = filename_gz.replace('.gz', '')

# Download the file
if not os.path.isfile(filename):
    if not os.path.isfile(filename_gz):
        subprocess.run(['wget', url])
    
    # Uncompress the compressed file
    with gzip.open(filename_gz, 'rb') as f_in:
        with open(filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the compressed file
    os.remove(filename_gz)
else:
    print(f"The file {filename} already exists.")

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Regular expression pattern to filter out non-English text
#english_pattern = re.compile(r'^[a-zA-Z0-9\s,.!?\'-]*$')
english_pattern = re.compile(r'^[a-zA-Z\s,.!?\'-]*$')


# Set to keep track of unique content
unique_content = set()
#unique_content=[]

# List to store preprocessed documents
preprocessed_documents = []
# Read the WARC file
with open('CC-MAIN-20231211210408-20231212000408-00899.warc.wet', 'rb') as stream:
    for record in ArchiveIterator(stream):
        if record.rec_type == 'conversion':
            content_bytes = record.content_stream().read()
            try:
                # Decode the bytes content to a string using UTF-8 encoding
                content_str = content_bytes.decode('utf-8')
                
                # Use the regular expression to filter out non-English text
                if english_pattern.match(content_str):
                    # Detect the language of the content
                    if detect(content_str) == 'en':  # Check if the content is in English
                        # Check if the content is unique
                        if content_str not in unique_content:
                            # Add the unique content to the set
                            unique_content.add(content_str)
                            # Tokenize and preprocess the content
                            tokens = word_tokenize(content_str.lower())
                            preprocessed_documents.append(tokens)
            except Exception as e:
                print(f"Error processing content: {e}")

# Tokenize and clean the documents
stop_words = set(stopwords.words('english'))
# Add custom stop words
custom_stop_words = {'please','sorry', 'pages', 'new', 'sign', 'error', 'smartcaptcha','password', 'enabled', 'deleted', 'account', 'page', 'login', 'rights', 'reserved', 'browser', 'site', 'web', 'react', 'domain','copyright', 'javascript', 'enable', 'JavaScript'}
stop_words.update(custom_stop_words)

preprocessed_documents = []
for content_str in unique_content:
    tokens = word_tokenize(content_str.lower())
    # Filter out stop words and non-alphanumeric words
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    if filtered_tokens: # Check if the filtered tokens are not empty
        preprocessed_documents.append(filtered_tokens)

# Bigram
bigram = Phrases(preprocessed_documents, min_count=5, threshold=100)
bigram_mod = Phraser(bigram)
preprocessed_documents_with_bigrams = [bigram_mod[doc] for doc in preprocessed_documents]
# Trigram
trigram = Phrases(bigram_mod[preprocessed_documents], threshold=100)
trigram_mod = Phraser(trigram)
preprocessed_documents_with_trigrams = [trigram_mod[bigram_mod[doc]] for doc in preprocessed_documents]


# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(preprocessed_documents)

# Filter out extremes to limit the number of features
dictionary.filter_extremes(no_below=1, no_above=0.9)
#Convert the dictionary to a bag of words corpus
corpus = [dictionary.doc2bow(text) for text in preprocessed_documents]

#dictionary of trigrams
dictionary_with_phrases = corpora.Dictionary(preprocessed_documents_with_bigrams)
corpus_with_phrases = [dictionary_with_phrases.doc2bow(text) for text in preprocessed_documents_with_bigrams]


# Apply LDA
lda_model = LdaModel(corpus, num_topics=50, id2word=dictionary, passes=50, alpha='auto', eta='auto')

lda_model_with_phrases = LdaModel(corpus_with_phrases, num_topics=15, id2word=dictionary_with_phrases, passes=20, alpha='auto', eta='auto')


# Function to extract the topic labels
def get_topic_labels(lda_model, num_terms=3):
    topic_labels = {}
    for i in range(lda_model.num_topics):
        # Extract the terms and their probabilities
        terms = lda_model.show_topic(i, num_terms)
        # Combine the terms to create a label
        label = ' '.join([term for term, prob in terms])
        topic_labels[i] = label
    return topic_labels

# Function to remove 2-character words from topics
def remove_two_char_words(topics):
    cleaned_topics = {}
    for topic_num, words in topics.items():
        # Remove 2-character words using regular expression
        cleaned_words = ' '.join(word for word in words.split() if len(word) > 2)
        cleaned_topics[topic_num] = cleaned_words
    return cleaned_topics

# Function to remove verbs from topics using NLTK
def remove_verbs(topics):
    cleaned_topics = {}
    for topic_num, words in topics.items():
        # Tokenize the words in the topic
        tokens = word_tokenize(words)
        # Perform part-of-speech tagging
        tagged = nltk.pos_tag(tokens)
        # Remove words that are tagged as verbs (VB*)
        non_verbs = [word for word, tag in tagged if not tag.startswith('VB')]
        cleaned_topics[topic_num] = ' '.join(non_verbs)
    return cleaned_topics



# Get topic labels
auto_topic_labels = get_topic_labels(lda_model)
cleaned_topics = remove_two_char_words(auto_topic_labels)
cleaned_topics = remove_verbs(cleaned_topics)

# Function to create a word cloud
def plot_word_cloud(topic_words):
    all_words = ' '.join([' '.join(words.split()) for words in topic_words.values()])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Topics Word Cloud')
    plt.show()

# Plot the word cloud
plot_word_cloud(cleaned_topics)