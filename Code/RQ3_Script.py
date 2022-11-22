import  csv, nltk, os.path, requests, spacy, torch, torchtext
from statistics import mean
from prettytable import PrettyTable

with_sampling = False
with_csl = False

cutoff_common_words_threshold = 250
nlp = spacy.load("en_core_web_sm")
glove = torchtext.vocab.GloVe(name="6B", dim=50)
stopwords = nltk.corpus.stopwords.words('english')
file = open("common_words_list.txt", "r")
common_words_lst = file.read()
common_words_to_remove = common_words_lst[0:cutoff_common_words_threshold]
for i in common_words_to_remove:
    stopwords.append(i)

# Remove additional stopwords from text
other_stopwords = ('-', '।', '|', '!', '?', ',', '.', '...', ':', ';', '@', '$', '%', '^', '&', '*', '/', '(', ')', '<', '>', '[', ']', '{', '}', '~', '`', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', "#", "+", "=", '\t', '\n', '\n\n', "s", 'the', 'as', 'are', 'own', 'in', 'ii', 'iii', 'iv', 'v', 'a', 'ed','first', 'second', 'third', 'fourth', 'fifth')
for i in other_stopwords:
    stopwords.append(i)

# GloVe vectors used to calculate cosine similarity between two words
def cosine_sim(original_word, comparison_word):
    first_word = original_word.casefold()
    second_word = comparison_word.casefold()
    word1 = glove[first_word]
    word2 = glove[second_word]
    tensor_value = torch.cosine_similarity(word1.unsqueeze(0), word2.unsqueeze(0))
    return tensor_value.item()

# accuracy = Number of Yes class matched terms / Number of all predicted terms
def compute_accuracy(yes_predictions, all_predictions):
    if yes_predictions == 0 or all_predictions == 0:
        return 0
    return ((yes_predictions / all_predictions) * 100)


# Coverage = Number of P2 matched terms / Number of all terms in P2
def compute_coverage(coverage_lst, p2_words_lst):
    if len(coverage_lst) == 0 or len(p2_words_lst) == 0:
        return 0
    return ((len(coverage_lst) / len(p2_words_lst)) * 100)


# Extract individual sentences from text
def extract_sentences_from_file(text):
    file_sentences = []
    for paragraph in text:
        sentences = [sentence for sentence in paragraph.split('.') if sentence != '' and len(sentence.split()) > 3]
        sentences = list(map(str.strip,sentences))
        file_sentences.extend(sentences)
    return file_sentences


# Extract and lemmatize individual words from each sentence
def extract_lemmatize_words(sentence):
    doc = nlp(sentence)
    # create list of tokens from sentence
    words_lst = []
    token_lst = [token for token in doc]
    # check if tokens are NOUNS/VERBS and lemmatize if true
    for token in token_lst:
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            words_lst.append(token.lemma_)
    return words_lst


# Acquire dataset
model_names_lst = ['0000-cctns', '0000-inventory', '1998-themas', '1999-dii', '1999-tcs', '2001-elsfork', '2001-esa', '2001-npac', '2001-telescope', '2002-evlacorr', '2003-agentmom', '2003-qheadache', '2004-colorcast', '2004-e-procurement', '2004-ijis', '2004-rlcs', '2004-sprat', '2005-clarushigh', '2005-grid3D', '2005-microcare']

for model_name in model_names_lst:
    output_table = PrettyTable(["Iteration", "accuracy", "Coverage"])
    accuracy_lst = []
    coverage_lst = []

    for i in range(5):
        if with_sampling:
            if with_csl:
                model = model_name+"_With_Sampling_With_CSL"
                predicted_results_dataset_url = "https://raw.githubusercontent.com/dluitel/refsq23/Results/RQ3/"+model_name+"/With%20Sampling%20With%20CSL/Predicted%20-%20CSV"+model_name+"_Results_Iteration"+str(i)+".csv"
            else:
                model = model_name+"_With_Sampling_No_CSL"
                predicted_results_dataset_url = "https://raw.githubusercontent.com/dluitel/refsq23/Results/RQ3/"+model_name+"/With%20Sampling%20No%20CSL/Predicted%20-%20CSV"+model_name+"_Results_Iteration"+str(i)+".csv"
        else:
            model = model_name+"_No_Sampling"
            predicted_results_dataset_url = "https://raw.githubusercontent.com/dluitel/refsq23/Results/RQ3/"+model_name+"_RF/No%20Sampling/Predicted%20-%20CSV/"+model_name+"_Results_Iteration"+str(i)+".csv"

        predicted_results_dataset = requests.get(predicted_results_dataset_url)
        predicted_results_file = predicted_results_dataset.text.split('\n')
        predictions_pred_class = [line.strip('\r') for line in predicted_results_file]

        p1 = model_name+"_15_Predictions/"+model_name+"_P1_Words.txt"
        p1_file = open(p2_text, "r")
        p1_words = p1_file.read()
        p1_words_lst = [x.strip(' ') for x in p1_words.split()]

        predictions_without_duplicates = list(set(predictions_pred_class))
        predictions = [item[:item.find(',')] for item in predictions_without_duplicates]
        predictions_without_stopwords = list(set(predictions) - set(stopwords))
        novel_predictions = list(set(predictions_without_stopwords) - set(p1_words_lst))

        novel_prediction_class = []
        for word in predictions_without_duplicates:
            for item in novel_predictions:
                if item == word[:word.find(',')]:
                    novel_prediction_class.append(word)

        all_yes_words = [word[:word.find(',')] for word in novel_prediction_class if 'TRUE' in word]

        novel_yes_words = []
        coverage_values = []
        for word in all_yes_words:
                doc = nlp(word)
                if doc[0].pos_ == 'NOUN' or doc[0].pos_ == 'VERB':
                    for item in p2_words_lst:
                        cosine = cosine_sim(item.casefold(), word.casefold())
                        if cosine >= 0.85:
                            novel_yes_words.append(word)
                            if item not in coverage_values:
                                coverage_values.append(item)

        accuracy = compute_accuracy(len(novel_yes_words), len(all_yes_words))
        coverage = compute_coverage(coverage_values, p2_words_lst)

        accuracy_lst.append(accuracy)
        coverage_lst.append(coverage)
        output_table.add_row([i, accuracy, coverage])

    avg_accuracy = round(mean(accuracy_lst), 2)
    avg_coverage = round(mean(coverage_lst), 2)

    output_table.add_row(['------', '------', '------', '------'])
    output_table.add_row(['Average', avg_accuracy, avg_coverage])

    rq3_output_file_name =  "RQ3_Output_"+model_name+".txt"
    with open(rq3_output_file_name, 'w') as w:
       w.write(str(output_table))
    print("\n", model_name, " | accuracy: ", avg_accuracy, " | Coverage: ", avg_coverage)
    print("Output written out to:", rq3_output_file_name)
