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

    # Hiperparâmetros para Fine-Tuning
    nr_topics_list  = [50, 70, 90, 110, 130, "auto"]
    min_topic_sizes = [3, 5, 10, 15, 20, 25]
    ngram_ranges    = [(1, 1)]
    num_iterations  = 10

    # Carregando o melhor modelo
    best_model = SentenceTransformer("neuralmind/bert-large-portuguese-cased")

    print("Pré-calculando embeddings com BERTimbau large...")
    embeddings = best_model.encode(docs_train, show_progress_bar=True)

    # DataFrame para armazenar resultados do Fine-Tuning
    results_finetune = pd.DataFrame(columns=[
        "n_gram_range", "nr_topics", "min_topic_size",
        "num_topics_found", "NC", "ND", "WC", "WD",
        "Weighted Score", "valid_iters"
    ])

    for ngram in ngram_ranges:
        for nr_topics in nr_topics_list:
            for min_size in min_topic_sizes:

                NC_list        = []
                ND_list        = []
                WC_list        = []
                WD_list        = []
                WS_list        = []
                ntf_list       = []

                print(f"\nConfiguração: ngram={ngram}, nr_topics={nr_topics}, min_topic_size={min_size}")

                for it in range(num_iterations):
                    print(f" Iteração {it+1}/{num_iterations}")
                    try:
                        # Instancia BERTopic com os hiperparâmetros
                        params = {
                            'embedding_model': best_model,
                            'n_gram_range': ngram,
                            'min_topic_size': min_size,
                            'language': 'portuguese',
                            'verbose': False
                        }
                        if nr_topics != "auto":
                            params['nr_topics'] = nr_topics

                        topic_model = BERTopic(**params)
                        topics, _   = topic_model.fit_transform(docs_train, embeddings)

                        # Contagem de tópicos válidos (exclui -1)
                        valid_labels = [t for t in topics if t != -1]
                        ntf_list.append(len(set(valid_labels)))

                        # Reconstrói o dicionário usando o mesmo analyzer do BERTopic
                        cleaned_docs  = topic_model._preprocess_text(docs_train)
                        analyzer      = topic_model.vectorizer_model.build_analyzer()
                        texts_tokens  = [analyzer(doc) for doc in cleaned_docs]
                        dictionary    = Dictionary(texts_tokens)

                        # Extrai tópicos e filtra vazios/outliers
                        tp = topic_model.get_topics()
                        tp.pop(-1, None)
                        keyword_list = [[w for w, _ in kws] for kws in tp.values()]
                        keyword_list = [kw for kw in keyword_list if kw]
                        if not keyword_list:
                            raise ValueError("Nenhum tópico válido para coerência")

                        # Calcula Coerência c_npmi
                        coherence_model = CoherenceModel(
                            topics=keyword_list,
                            texts=texts_tokens,
                            dictionary=dictionary,
                            coherence='c_npmi'
                        )
                        NC = coherence_model.get_coherence()

                        # Calcula Diversidade
                        all_kw = [w for kws in keyword_list for w in kws]
                        ND = len(set(all_kw)) / len(all_kw) if all_kw else 0.0

                        # Métricas ponderadas
                        WC = 0.8 * NC
                        WD = 0.2 * ND
                        WS = WC + WD

                        # Acumula valores
                        NC_list.append(NC)
                        ND_list.append(ND)
                        WC_list.append(WC)
                        WD_list.append(WD)
                        WS_list.append(WS)

                    except Exception as e:
                        print(f"  ⚠️ Falha: {e}")
                        # Imputa nan para manter sempre num_iterations pontos
                        NC_list.append(np.nan)
                        ND_list.append(np.nan)
                        WC_list.append(np.nan)
                        WD_list.append(np.nan)
                        WS_list.append(np.nan)
                        ntf_list.append(np.nan)

                    finally:
                        # Limpeza de memória
                        try:
                            del topic_model
                            torch.cuda.empty_cache()
                        except:
                            pass

                # Cálculo de médias com nanmean e supressão de warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_topics   = np.nanmean(ntf_list)
                    avg_NC       = np.nanmean(NC_list)
                    avg_ND       = np.nanmean(ND_list)
                    avg_WC       = np.nanmean(WC_list)
                    avg_WD       = np.nanmean(WD_list)
                    avg_WS       = np.nanmean(WS_list)

                valid_iters = int(sum(~np.isnan(NC_list)))
                print(f"-> Iterações válidas: {valid_iters}/{num_iterations}")

                # Armazena no DataFrame
                results_finetune = pd.concat([
                    results_finetune,
                    pd.DataFrame([{
                        'n_gram_range': ngram,
                        'nr_topics': nr_topics,
                        'min_topic_size': min_size,
                        'num_topics_found': avg_topics,
                        'NC': avg_NC,
                        'ND': avg_ND,
                        'WC': avg_WC,
                        'WD': avg_WD,
                        'Weighted Score': avg_WS,
                        'valid_iters': valid_iters
                    }])
                ], ignore_index=True)

    # Salva resultados e exibe ordenado por pontuação
    results_finetune.to_csv(
        'resultados_interno_p2.csv',
        sep=';', decimal=',', index=False, encoding='utf-8'
    )
    print("\n--- Fine-Tuning Concluído ---")
    print(results_finetune.sort_values('Weighted Score', ascending=False))

if __name__ == '__main__':
    main()