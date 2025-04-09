import os
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle

# Initialize stemmer and stopwords
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def parse_notes(path) -> dict:
    print('Parsing NOTEEVENTS.csv ...')
    notes_path = os.path.join(path, 'NOTEEVENTS_subset2.csv')
    notes = pd.read_csv(
        notes_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'TEXT'],
        dtype={'SUBJECT_ID': 'Int64', 'HADM_ID': 'Int64', 'TEXT': str}
    ).dropna(subset=['HADM_ID', 'TEXT'])

    visit_notes = notes.groupby(['SUBJECT_ID', 'HADM_ID'])['TEXT'].apply(lambda x: ' '.join(x)).to_dict()
    print(f'Processed {len(visit_notes)} visits')
    return visit_notes

def extract_word(text: str) -> list:
    """Extract words from text, apply stemming, and remove stopwords, ignoring [CLS] token."""
    text = re.sub(r'[^A-Za-z_]', ' ', text.strip().lower())
    words = word_tokenize(text)
    return [ps.stem(word) for word in words if word not in stopwords_set and word != '[cls]']

def encode_note_train(visit_notes: dict, vocab_size=None) -> (dict, dict):
    print('Encoding notes ...')
    dictionary = {}
    patient_note_encoded = {}
    
    # Create vocabulary
    vocab_index = 0
    for (_, _), text in visit_notes.items():
        words = extract_word(text)
        for word in words:
            if word not in dictionary:
                dictionary[word] = vocab_index
                vocab_index += 1
    
    vocab_size = vocab_size or vocab_index
    
    patient_visits = {}
    for (pid, vid), text in visit_notes.items():
        words = extract_word(text)
        multi_hot_vector = np.zeros(vocab_size, dtype=int)
        for word in words:
            if word in dictionary:
                multi_hot_vector[dictionary[word]] = 1
        if pid not in patient_visits:
            patient_visits[pid] = []
        patient_visits[pid].append(multi_hot_vector)
    
    # Convert lists of visit vectors to 2D arrays per patient
    for pid in patient_visits:
        patient_visits[pid] = np.vstack(patient_visits[pid])
    
    print(f'Encoded {len(patient_visits)} patients')
    
    # Save patient visit notes to pkl file
    with open('patient_notes_multihot.pkl', 'wb') as f:
        pickle.dump(patient_visits, f)
    
    # Save dictionary separately to pkl file
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
    
    return patient_visits, dictionary

def main():
    path = "/Users/pratikranjan/Desktop/vecocare_v2.0/subset_data/subset_2"
    visit_notes = parse_notes(path)
    encoded_notes, vocab = encode_note_train(visit_notes)
    print("Processing complete. Encoded notes and vocabulary saved.")

if __name__ == "__main__":
    main()
