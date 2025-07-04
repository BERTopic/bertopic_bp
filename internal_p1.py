import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import time
import nltk
import torch
import warnings
import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from scipy.stats import norm
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Stopwords and lemmatizer setup
nltk.download('stopwords')
stop_words = stopwords.words('portuguese')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def clean_and_prepare(df: pd.DataFrame, text_col: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Renomeia coluna, padroniza texto, remove duplicatas, stopwords, tokeniza e lematiza.
    Retorna três listas paralelas:
      - lemmatized: textos prontos para o modelo
      - originals: textos originais (limpos, sem lower)
      - valid_idx: índices das linhas válidas no DataFrame original
    """
    df = df.rename(columns={text_col: 'pipeline_text'})
    df = df.astype(str).fillna("")
    df['pipeline_text'] = df['pipeline_text'].str.strip()

    # Filtra vazios e duplicados
    mask_valid = df['pipeline_text'] != ""
    df = df[mask_valid].drop_duplicates(subset=['pipeline_text'])
    valid_idx = df.index.tolist()

    # Guarda os originais
    originals = df['pipeline_text'].tolist()

    # Preprocessamento: lower, stopwords e tokenização
    processed = df['pipeline_text'].str.lower().apply(
        lambda x: " ".join(
            word for word in simple_preprocess(x) if word not in stop_words
        )
    ).tolist()

    # Lematização
    lemmatized = [
        " ".join(lemmatizer.lemmatize(tok) for tok in doc.split())
        for doc in processed
    ]

    return lemmatized, originals, valid_idx


def main():
    # Load and prepare training data
    train_df = pd.read_csv("training.csv", encoding='utf-8')
    docs_train, originals_train, _ = clean_and_prepare(train_df, "proposal_text")
    print(f"*** Loaded {len(docs_train)} documents for training")

    # Configurations
    num_iterations = 10
    desired_topic_numbers = [10, 30, 50, 70, 90, 110, 130, 'auto']

    # Embedding models
    models = {
        'LaBSE': SentenceTransformer('sentence-transformers/LaBSE'),
        'MiniLM': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),
        'MPNet': SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'),
        'BERTimbau-base': SentenceTransformer('neuralmind/bert-base-portuguese-cased'),
        'BERTimbau-large': SentenceTransformer('neuralmind/bert-large-portuguese-cased'),
        'Legal-BERT': SentenceTransformer('ulysses-camara/legal-bert-pt-br')
    }

    # Results DataFrame
    results = pd.DataFrame(columns=[
        'Model','Num_Topics','num_topics_found',
        'NC','ND','WC','WD','Weighted Score','valid_iters'
    ])

    for model_name, model in models.items():
        print(f"\n=== Processing model: {model_name} ===")
        embeddings = model.encode(docs_train, show_progress_bar=True)

        for num_topics_target in desired_topic_numbers:
            print(f"\n-- Target topics: {num_topics_target} --")
            NC_list, ND_list, WC_list, WD_list, WS_list, ntf_list = [], [], [], [], [], []

            for it in range(num_iterations):
                print(f" Iteration {it+1}/{num_iterations}")
                try:
                    # Initialize BERTopic
                    if num_topics_target == 'auto':
                        topic_model = BERTopic(
                            embedding_model=model,
                            n_gram_range=(1,1),
                            language='portuguese',
                            verbose=False
                        )
                    else:
                        topic_model = BERTopic(
                            embedding_model=model,
                            n_gram_range=(1,1),
                            language='portuguese',
                            nr_topics=num_topics_target,
                            verbose=False
                        )

                    # Fit & transform
                    topics, _ = topic_model.fit_transform(docs_train, embeddings)
                    ntf_list.append(len(set(t for t in topics if t != -1)))

                    # Tokenize for coherence
                    cleaned = topic_model._preprocess_text(docs_train)
                    analyzer = topic_model.vectorizer_model.build_analyzer()
                    texts_tokens = [analyzer(doc) for doc in cleaned]
                    dictionary = Dictionary(texts_tokens)

                    # Extract and filter topics safely
                    raw_topics = topic_model.get_topics()
                    raw_topics.pop(-1, None)

                    # Extract only words and filter out topics with <2 tokens
                    keyword_list = [
                        [word for word, _ in kws]
                        for kws in raw_topics.values()
                    ]
                    keyword_list = [topic for topic in keyword_list if len(topic) > 1]

                    if not keyword_list:
                        print(f"  ⚠️ No valid topics (≥2 tokens) for coherence, skipping")
                        NC_list.append(np.nan)
                        ND_list.append(np.nan)
                        WC_list.append(np.nan)
                        WD_list.append(np.nan)
                        WS_list.append(np.nan)
                        continue

                    # Compute coherence and diversity
                    coherence_model = CoherenceModel(
                        topics=keyword_list,
                        texts=texts_tokens,
                        dictionary=dictionary,
                        coherence='c_npmi'
                    )
                    NC = coherence_model.get_coherence()
                    all_kw = [w for kws in keyword_list for w in kws]
                    ND = len(set(all_kw)) / len(all_kw) if all_kw else 0.0

                    # Weighted scores
                    WC = 0.8 * NC
                    WD = 0.2 * ND
                    WS = WC + WD

                    NC_list.append(NC)
                    ND_list.append(ND)
                    WC_list.append(WC)
                    WD_list.append(WD)
                    WS_list.append(WS)

                except Exception as e:
                    print(f"  ⚠️ Iter {it+1} failed: {e}")
                    NC_list.append(np.nan)
                    ND_list.append(np.nan)
                    WC_list.append(np.nan)
                    WD_list.append(np.nan)
                    WS_list.append(np.nan)
                    ntf_list.append(np.nan)

                finally:
                    # Clean up
                    try:
                        del topic_model
                        torch.cuda.empty_cache()
                    except:
                        pass

            # Aggregate results
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                avg_ntf = np.nanmean(ntf_list)
                avg_NC = np.nanmean(NC_list)
                avg_ND = np.nanmean(ND_list)
                avg_WC = np.nanmean(WC_list)
                avg_WD = np.nanmean(WD_list)
                avg_WS = np.nanmean(WS_list)

            valid_iters = int(np.sum(~np.isnan(NC_list)))
            print(f"-> Valid iterations: {valid_iters}/{num_iterations}")

            results = pd.concat([results, pd.DataFrame([{  
                'Model': model_name,
                'Num_Topics': num_topics_target,
                'num_topics_found': avg_ntf,
                'NC': avg_NC,
                'ND': avg_ND,
                'WC': avg_WC,
                'WD': avg_WD,
                'Weighted Score': avg_WS,
                'valid_iters': valid_iters
            }])], ignore_index=True)

        # Cleanup per model
        del embeddings
        torch.cuda.empty_cache()

    # Save and display
    results.to_csv('resultados_interno_p1.csv', sep=';', decimal=',', index=False, encoding='utf-8')
    print("\n--- Final Results ---")
    print(results.sort_values('Weighted Score', ascending=False))


if __name__ == '__main__':
    main()
