import collections, csv, fileinput, nltk, os, random, re, requests, spacy, statistics, sys, textacy, torch, torchtext
import Levenshtein as lev
import pandas as pd
import tensorflow as tf
import numpy as np
from collections import Counter
from nltk.corpus import wordnet
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM, TFAutoModelForMaskedLM

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 9000000
glove = torchtext.vocab.GloVe(name="6B", dim=50)
stopwords = nltk.corpus.stopwords.words('english')

# P1 Dataset
# model_names_lst = ['0000-gamma', '1995-gemini', '1999-multi-mahjong', '2000-nasax38', '2001-beyond', '2001-ctcnetwork', '2001-hats', '2001-libra', '2001-spacefractions', '2002-evlaback', '2002-sceapi', '2003-pnnl', '2003-tachonet', '2004-gridbgc', '2004-jse', '2004-philips', '2004-watcom', '2004-watcomgui', '2005-claruslow', '2005-nenios']

# P2 Dataset
model_names_lst = ['0000-cctns', '0000-inventory', '1998-themas', '1999-dii', '1999-tcs', '2001-elsfork', '2001-esa', '2001-npac', '2001-telescope', '2002-evlacorr', '2003-agentmom', '2003-qheadache', '2004-colorcast', '2004-e-procurement', '2004-ijis', '2004-rlcs', '2004-sprat', '2005-clarushigh', '2005-grid3D', '2005-microcare']

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

cutoff_similarity_threshold = 0.85
cutoff_common_words_threshold = 250
num_pred_words_lst = [15]
wikidominer_depth_lst = [0]
masked_words = []

file = open("common_words_list.txt", "r")
common_words_lst = file.read()
common_words_to_remove = common_words_lst[0:cutoff_common_words_threshold]
for i in common_words_to_remove:
    stopwords.append(i)

# Remove additional stopwords from text
other_stopwords = ('-', '।', '|', '!', '?', ',', '.', '...', ':', ';', '@', '$', '%', '^', '&', '*', '/', '(', ')', '<', '>', '[', ']', '{', '}', '~', '`', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', "#", "+", "=", '\t', '\n', '\n\n', "s", 'the', 'as', 'are', 'own', 'in', 'ii', 'iii', 'iv', 'v', 'a', 'ed','first', 'second', 'third', 'fourth', 'fifth')
for i in other_stopwords:
    stopwords.append(i)

# Helper function for feature 3 and 4 - find position of ranking in bucket
def find(bucket, target):
    for rank, words in bucket.items():
        if target in words:
            return rank
    return 10

# 0. POS tag for actual word
def feature0(token):
    doc = nlp(token)
    return doc[0].pos_

# 1. POS tag for predicted word
def feature1(masked_sentence, sentence, mask_index, predicted_token):
    words_lst = masked_sentence.split()
    words_lst[mask_index] = predicted_token
    sentence = " ".join(words_lst)
    doc = nlp(sentence)
    for token in doc:
        if token.text == predicted_token:
                return token.pos_
    return "X"

# 2. POS tag for predicted word matching actual word
def feature2(feature0, feature1):
    return feature0 == feature1

# 3. Number of times words are predicted (normalized)
def feature3(predicted_tokens):
    count = 0
    bucket_num = 0
    bucket = {}
    bucket.setdefault(0, [])
    # Get frequency of all unique words in prediction list
    unique_words = collections.Counter(predicted_tokens)
    unique_words_keys = []
    for word in unique_words.most_common():
        unique_words_keys.append(word[0])
    # Sort unique words into ranked frequency buckets
    for word in unique_words_keys:
      bucket[bucket_num].append(word)
      count += unique_words[word]
      if (count / len(predicted_tokens)) >= 0.10:
          count = 0
          bucket_num += 1
          bucket.setdefault(bucket_num, [])
    return bucket

# 4. Number of times predicted word appears in corpus
def feature4(predicted_tokens, corpus):
    count = 0
    bucket_num = 0
    bucket = {}
    bucket.setdefault(0, [])
    # Get frequency of all unique words in corpus and prediction list
    unique_corpus_words = collections.Counter(corpus)
    unique_corpus_words_keys = []
    for word in unique_corpus_words.most_common():
        unique_corpus_words_keys.append(word[0])

    unique_predicted_words = collections.Counter(predicted_tokens)
    unique_predicted_words_keys = []
    for word in unique_predicted_words.most_common():
        unique_predicted_words_keys.append(word[0])

    unshared_words_lst = [x for x in unique_predicted_words_keys if x not in unique_corpus_words_keys]
    shared_words_lst = [x for x in unique_corpus_words_keys if x in unique_predicted_words_keys]
    denominator = 0
    for word in shared_words_lst:
        denominator += unique_corpus_words[word]

    # Count and sort frequency of predicted word appearing in corpus
    for word in shared_words_lst:
      bucket[bucket_num].append(word)
      count += unique_corpus_words[word]
      if (count / denominator) >= 0.10:
          count = 0
          bucket_num += 1
          bucket.setdefault(bucket_num, [])
    # Insert all words not appearing in corpus to the last bucket
    bucket_num += 1
    bucket.setdefault(bucket_num, [])
    for word in unshared_words_lst:
        bucket[bucket_num].append(word)
    return bucket

# 5. Length of predicted token; 6. Length of actual word
def feature5_6(token):
    return len(token)

# 7. Ratio of token length to masked word's length
def feature7(feature5, feature6):
    return float(feature5/feature6)

# 8. Lev. Distance between predicted word and actual word
def feature8(predicted_token, actual_token):
    return lev.ratio(actual_token, predicted_token)

# 9. Semantic similarity between predicted word and actual word
def feature9(predicted_token, actual_token):
    return cosine_sim(actual_token.casefold(), predicted_token.casefold())

# 10. TFIDF average rank using corpus --> Location for where the word was predicted
def feature10(tfidf, predicted_token):
    if predicted_token in tfidf.keys():
        return (tfidf.loc['Mean'][predicted_token])
    return 0

# 11. TFIDF highest rank using corpus --> Location for where the word was predicted
def feature11(tfidf, predicted_token):
    if predicted_token in tfidf.keys():
        return (tfidf.loc['Max'][predicted_token])
    return 0

# Class value for each prediction
def output_class(predicted_token, feat1, second_half_words_without_duplicates):
    if feat1 == 'NOUN' or feat1 == 'VERB':
        for word in updated_second_half_words_without_duplicates:
            cosine_distance = cosine_sim(word.casefold(), predicted_token.casefold())
            if cosine_distance >= cutoff_similarity_threshold:
                return predicted_token, True
    return predicted_token, False

def feature_matrix(features_lst, prediction_class_lst, actual_word, predicted_word, predicted_word_lemmatized, token_weight, mask_index, masked_sentence, sentence, tfidf, frequency_of_predicted_words_bucket, corpus_words_bucket, updated_second_half_words_without_duplicates):
    '''
    Features List: Calculated Per Predicted Word
    0. POS tag for actual word
    1. POS tag for prediction word
    2. Is POS tag for predicted word same as for actual word?
    3. Number of times predicted (normalized) - global
    4. Number of times appearing in corpus (normalized)
    5. Predicted token length
    6. Length of the actual word
    7. Ratio of token length to masked word's length
    8. Lev. Distance between predicted word and actual word
    9. Semantic similarity between predicted word and actual word
    10. TFIDF average rank using corpus --> Location where word is predicted
    10. TFIDF max rank using corpus --> Location where word is predicted
    12. Probability value of predicted token
    '''
    feat0 = feature0 (actual_word)
    feat1 = feature1(masked_sentence, sentence, mask_index, predicted_word)
    feat2 = feature2(feat0, feat1)
    feat3 = find(frequency_of_predicted_words_bucket, predicted_word_lemmatized)
    feat4 = find(corpus_words_bucket, predicted_word_lemmatized)
    feat5 = feature5_6(predicted_word_lemmatized)
    feat6 = feature5_6(actual_word)
    feat7 = feature7(feat5, feat6)
    feat8 = feature8(predicted_word_lemmatized, actual_word)
    feat9 = feature9(predicted_word_lemmatized, actual_word)
    feat10 = feature10(tfidf, predicted_word_lemmatized)
    feat11 = feature10(tfidf, predicted_word_lemmatized)
    feat12 = float(token_weight)
    class_value, matched_word = output_class(predicted_word_lemmatized, feat1, updated_second_half_words_without_duplicates)
    prediction_class_lst.append([matched_word, class_value])
    features_lst.append([feat0, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11, feat12, class_value])

# Create global buckets for features 3 and 4 then append the appearance of predicted token to features_lst
def create_buckets(predicted_words, corpus):
    count = 0
    frequency_of_predicted_words_bucket = feature3(predicted_words)
    corpus_words_bucket = feature4(predicted_words, corpus)
    return frequency_of_predicted_words_bucket, corpus_words_bucket

# GloVe vectors used to calculate cosine similarity between two words
def cosine_sim(original_word, comparison_word):
    word1 = glove[original_word]
    word2 = glove[comparison_word]
    tensor_value = torch.cosine_similarity(word1.unsqueeze(0), word2.unsqueeze(0))
    return tensor_value.item()

# Extract individual sentences from text
def extract_sentences_from_file(text):
    file_sentences = []
    for paragraph in text:
        sentences = [sentence for sentence in paragraph.split('.') if sentence != '' and len(sentence.split()) > 3]
        sentences = list(map(str.strip,sentences))
        file_sentences.extend(sentences)
    return file_sentences

# Extract individual words from each sentence
def extract_words(sentence):
    doc = nlp(sentence)
    # create list of tokens from sentence
    words_lst = [token.text for token in doc]
    word_pos_tags = [token.pos_ for token in doc]
    return words_lst, word_pos_tags

# Extract and lemmatize individual words from each sentence
def extract_lemmatize_words(sentence):
    doc = nlp(sentence)
    # create list of tokens from sentence
    words_lst = [token.text for token in doc]
    token_lst = [token for token in doc]
    # check if tokens are NOUNS/VERBS and lemmatize if true
    for token in token_lst:
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            words_lst[token_lst.index(token)] = token.lemma_
    return words_lst

def lst_to_pdf(lst, model_name):
    # Create new input document to feed  WikiDoMiner
    file_name = model_name+".txt"
    file = open(file_name, "w")
    for i in lst:
        file.writelines(i)
        file.writelines("\n")
    # Output text file as BERT fine-tuning text
    return file_name

# Match predicted words to second half of text
def match_against_2nd_half(terms, updated_second_half_words_without_duplicates, all_first_half_words_lemmatized):
    predictions_not_in_first_half = 0
    all_matched_terms = []
    novel_matched_terms = []
    duplicate_free_matched_terms = []

    for item in terms:
        # Ensure the predicted word is lemmatized
        doc = nlp(item)
        if doc[0].pos_ == 'NOUN' or doc[0].pos_ == 'VERB':
            item = doc[0].lemma_
            if item not in all_first_half_words_lemmatized:
                predictions_not_in_first_half += 1
                # Search for predicted words that are semantically close to words that appear in second half
                for word in updated_second_half_words_without_duplicates:
                    distance = cosine_sim(item.casefold(), word.casefold())
                    # Write out words that exceed the cosine sim. requirement and whose lemmatized form does not appear in 1st half of text
                    if distance >= cutoff_similarity_threshold:
                        all_matched_terms.append(item)
                        # Assign only one prediction to each word in 2nd half
                        if word not in novel_matched_terms:
                            novel_matched_terms.append(word)
                        # Filter out duplicate predictions
                        if item not in duplicate_free_matched_terms:
                            duplicate_free_matched_terms.append(item)
                            break
    return predictions_not_in_first_half, len(all_matched_terms), len(novel_matched_terms), len(duplicate_free_matched_terms)

# accuracy = Number of matched terms / Number of terms in Predictions not in First Half
def compute_accuracy(matched_words, predictions_not_in_first_half):
    if matched_words == 0:
        return 0
    return ((matched_words / predictions_not_in_first_half) * 100)

# coverage = Number of matched terms / Number of all terms in second half of text
def compute_coverage(matched_words, updated_second_half_words_without_duplicates):
    if matched_words == 0:
        return 0
    return ((matched_words / len(updated_second_half_words_without_duplicates)) * 100)

home_path = os.getcwd()
for model_name in model_names_lst:
    output_table = PrettyTable(["Predictions/Mask", "Total Num. of predictions", "Num. of non-duplicate predictions", "Num. of predictions not in 1st half", "Num. of total matched terms (with duplicates)", "Num. of total matched terms (without duplicates)", "Num. of novel matched terms", "Num. of non-duplicate terms in 2nd half", "Accuracy", "Coverage"])

    dataset_url = "https://raw.githubusercontent.com/dluitel/refsq23/Cleaned_Datasets/Test_P2/"+model_name+".txt"
    data = requests.get(dataset_url)
    text = data.text.split('\n')
    text_sentences = extract_sentences_from_file(text)

    for num_pred_words in num_pred_words_lst:
        random.shuffle(text_sentences)
        first_half_text = text_sentences[:len(text_sentences)//2]
        second_half_text = text_sentences[len(text_sentences)//2:]

        # Extract and lemmatize words from first half of text
        all_first_half_words_lemmatized = []
        for sentence in first_half_text:
            all_first_half_words_lemmatized.extend(extract_lemmatize_words(sentence))

        # Extract and lemmatize words from second half of text
        all_second_half_words_lemmatized = []
        for sentence in second_half_text:
            all_second_half_words_lemmatized.extend(extract_lemmatize_words(sentence))

        wikidominer_title = lst_to_pdf(first_half_text, model_name)
        corpus_folder = model_name+"_"+str(num_pred_words)+"_Predictions/Mask_Corpus"
        merged_corpus = 'all_merged_files'
        bert_training_corpus = corpus_folder+"/"+merged_corpus
        os.system("python WikiDoMiner.py --doc {a} --output-dir {b} --wiki-depth {c}".format(a=wikidominer_title, b=corpus_folder, c=0))

        # Merge individual WikiDoMiner files into single corpus
        os.chdir(corpus_folder)
        corpus_files_name = os.listdir()
        os.system("cat * > {}".format(merged_corpus))

        # Read in merged WikiDoMiner corpus file and lemmatize words for bucket creation
        with open(merged_corpus) as corpus_file:
            corpus = corpus_file.read()
        corpus_text = corpus.split('\n')
        corpus_sentences = extract_sentences_from_file(corpus_text)
        lemmatized_corpus = []
        for sentence in corpus_sentences:
            lemmatized_corpus.extend(extract_lemmatize_words(sentence))

        # Remove stopwords from corpus
        updated_corpus = [x.lower() for x in lemmatized_corpus if x not in stopwords]

        # Process documents in corpus for TFIDF and buckets
        for file in corpus_files_name:
            # Read in document and clean text by lemmatizing and removing stopwords
            with open(file) as corpus_file:
                file_text = corpus_file.read()
                lemmatized_text = extract_lemmatize_words(file_text)
                updated_text = [x.lower() for x in lemmatized_text if x not in stopwords]
                cleaned_text = " ".join(updated_text)
            os.remove(file)
            # Write out cleaned text to same filename
            with open(file, 'w') as w:
                w.write(cleaned_text)

        # TFIDF with sklearn
        tfidf_vectorizer = TfidfVectorizer(input='filename')
        tfidf_total = tfidf_vectorizer.fit_transform(corpus_files_name)
        tfidf_df = pd.DataFrame(tfidf_total.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_df.replace(0, np.nan, inplace=True)
        tfidf_df = tfidf_df.mean(axis=0, skipna=True).rename('Mean').pipe(tfidf_df.append)
        tfidf_df = tfidf_df.max(axis=0, skipna=True).rename('Max').pipe(tfidf_df.append)

        os.chdir(home_path)

        # Create file containing BERT output values at each interval of cosine sim. and common word removal
        features_lst = []
        prediction_class_lst = []
        predicted_words = []
        df_rows_lst = []

        for sentence in first_half_text:
            # Create list of tokens and corresponding POS tag from sentence
            sentence_words = [x.strip(' ') for x in sentence.split()]
            words_lst, word_pos_tags = extract_words(" ".join(sentence_words))
            word_mask_index = 0

            # Mask each word in each sentence in first half of text
            for word in words_lst:
                words_lst[word_mask_index] = "[MASK]"
                masked_sentence = " ".join(words_lst)

                # Tokenize input
                text = "[CLS] %s [SEP]"%masked_sentence
                tokenized_text = tokenizer.tokenize(text)
                masked_index = tokenized_text.index("[MASK]")

                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])

                # Predict all tokens for vanilla model
                with torch.no_grad():
                    outputs = model(tokens_tensor)
                    prediction_logits = outputs[0]

                # Probability distribution of predicted tokens using softmax function
                probs = tf.nn.softmax(prediction_logits[0, masked_index])
                top_k_weights, top_k_indices = tf.math.top_k(probs, num_pred_words, sorted=True)

                # Append predicted word to list
                for i, pred_idx in enumerate(top_k_indices):
                    predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
                    token_weight = top_k_weights[i]
                    if predicted_token not in stopwords and word not in stopwords and (word_pos_tags[word_mask_index] == "NOUN" or word_pos_tags[word_mask_index] == "VERB"):
                        doc = nlp(predicted_token)
                        new_row = {'Actual_Word': word, 'Predicted_Word': predicted_token, 'Predicted_Word_Lemmatized': doc[0].lemma_, 'Token_Weight': float(token_weight), 'Mask_Index': word_mask_index, 'Masked_Sentence': masked_sentence, 'Sentence': words_lst}
                        df_rows_lst.append(new_row)
                        predicted_words.append(doc[0].lemma_)
                words_lst[word_mask_index] = word
                word_mask_index += 1

        # Remove stopwords and duplicate words from lists
        second_half_words_lowercase_without_duplicates = [x.lower() for x in list(set(all_second_half_words_lemmatized))]
        updated_second_half_words_without_duplicates = list(set(second_half_words_lowercase_without_duplicates) - set(stopwords))
        updated_novel_second_half_words = list(set(updated_second_half_words_without_duplicates) - set(all_first_half_words_lemmatized))

        updated_predicted_words_without_duplicates = [x.lower() for x in list(set(predicted_words))]
        updated_predicted_words_with_duplicates = [x.lower() for x in predicted_words]

        # Write out to file the predicted words not in first half of text but appear semantically close to words in second half list of words
        predictions_not_in_first_half, all_matched_terms, novel_matched_terms, duplicate_free_matched_terms = match_against_2nd_half(updated_predicted_words_without_duplicates, updated_novel_second_half_words, all_first_half_words_lemmatized)

        # accuracy = Number of matched terms / Number of predicted terms not in 1st half
        accuracy = compute_accuracy(duplicate_free_matched_terms, predictions_not_in_first_half)

        # coverage = Number of matched terms / Number of all terms in second half of text
        coverage = compute_coverage(novel_matched_terms, updated_novel_second_half_words)

        # Create global buckets for feature 3 and 4
        feat3_bucket, feat4_bucket = create_buckets(updated_predicted_words_with_duplicates, updated_corpus)

        # Create dataframe and remove duplicate tokens for feature matrix
        df_with_duplicate_predcitions = pd.DataFrame(df_rows_lst)
        df_without_duplicates = df_with_duplicate_predcitions.drop_duplicates(subset=['Predicted_Word'])

        for i, row in df_without_duplicates.iterrows():
            feature_matrix(features_lst, prediction_class_lst, row['Actual_Word'], row['Predicted_Word'], row['Predicted_Word_Lemmatized'], row['Token_Weight'], row['Mask_Index'], row['Masked_Sentence'], row['Sentence'], tfidf_df, feat3_bucket, feat4_bucket, updated_second_half_words_without_duplicates)

        print("Model Name: ", model_name)
        print("\nNumber of predicted words per mask: ", num_pred_words)
        print("Total number of predictions: ", len(predicted_words))
        print("Number of non-duplicate predictions: ", len(updated_predicted_words_without_duplicates))
        print("Number of predictions not in 1st half: ", predictions_not_in_first_half)
        print("Number of total matched terms (with duplicates): ", all_matched_terms)
        print("Number of total matched terms (without duplicates): ", duplicate_free_matched_terms)
        print("Number of novel matched terms: ", novel_matched_terms)
        print("Number of non-duplicate terms in second half of text: ", len(updated_novel_second_half_words))
        print("Accuracy: ", accuracy," | Coverage: ", coverage)
        print("----------------------------------------------------------------------------------------")

        output_table.add_row([num_pred_words, len(predicted_words), len(updated_predicted_words_without_duplicates), predictions_not_in_first_half, all_matched_terms, duplicate_free_matched_terms, novel_matched_terms, len(updated_novel_second_half_words), accuracy, coverage])

        features_header = []
        relation = "@RELATION "+model_name
        features_header.append([relation])
        features_header.append(["@ATTRIBUTE actualWordPOS {NOUN, VERB, ADJ, ADV, AUX, PRON, PROPN, PUNCT, X, SCONJ, ADP, CCONJ, INTJ, NUM, DET, SYM, SPACE, PART}"])
        features_header.append(["@ATTRIBUTE predictedWordPOS {NOUN, VERB, ADJ, ADV, AUX, PRON, PROPN, PUNCT, X, SCONJ, ADP, CCONJ, INTJ, NUM, DET, SYM, SPACE, PART}"])
        features_header.append(["@ATTRIBUTE posTagsMatch {TRUE, FALSE}"])
        features_header.append(["@ATTRIBUTE numTimesPredicted numeric"])
        features_header.append(["@ATTRIBUTE numTimesInCorpus numeric"])
        features_header.append(["@ATTRIBUTE predictedTokenLength numeric"])
        features_header.append(["@ATTRIBUTE actualTokenLength numeric"])
        features_header.append(["@ATTRIBUTE ratioOfTokenLengths numeric"])
        features_header.append(["@ATTRIBUTE levDistance numeric"])
        features_header.append(["@ATTRIBUTE semanticSim numeric"])
        features_header.append(["@ATTRIBUTE tfidfAverageRank numeric"])
        features_header.append(["@ATTRIBUTE tfidfHighestRank numeric"])
        features_header.append(["@ATTRIBUTE predictedProba numeric"])
        features_header.append(["@ATTRIBUTE class {TRUE, FALSE}"])
        features_header.append(["@DATA"])

        # Write out extracted features to csv file for WEKA
        extracted_features_file_name = model_name+"_"+str(num_pred_words)+"_Predictions/"+model_name+"_ExtractedFeatures.arff"
        features_file = open(extracted_features_file_name, 'w')
        feature_writer = csv.writer(features_file)
        with open(output_table_file_name, 'w') as w:
            for header in features_header:
                w.write(header)
            for features in features_lst:
                w.write(features)

        # Write out extracted features to csv file for WEKA
        prediction_class_file_name = model_name+"_"+str(num_pred_words)+"_Predictions/"+model_name+"_Predictions_Class.csv"
        prediction_class_file = open(prediction_class_file_name, 'w')
        prediction_class_writer = csv.writer(features_file)
        for prediction_class in prediction_class_lst:
            prediction_class_writer.writerow(prediction_class)

        # Write out P1 terms
        p1_file_name = model_name+"_"+str(num_pred_words)+"_Predictions/"+model_name+"_P1_Words.txt"
        p1_file = open(p1_file_name, 'w')
        p1_writer = csv.writer(p1_file)
        for word in all_first_half_words_lemmatized:
            p1_writer.writerow(word)

        # Write out P1 terms
        p2_file_name = model_name+"_"+str(num_pred_words)+"_Predictions/"+model_name+"_P2_Words.txt"
        p2_file = open(p2_file_name, 'w')
        p2_writer = csv.writer(p2_file)
        for word in all_second_half_words_lemmatized:
            p1_writer.writerow(word)

    # Write output_table and yes class prediction values to file
    output_table_file_name =  model_name+"_Output_Table.txt"
    with open(output_table_file_name, 'w') as w:
       w.write(str(output_table))
    print("\nOutput written out to:", output_table_file_name)
