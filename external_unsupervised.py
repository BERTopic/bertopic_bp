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
from typing import List, Tuple
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

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
    
    # Calcula embeddings dos documentos de treino
    print("Calculando embeddings dos documentos de treino...")
    embedding_model = SentenceTransformer("neuralmind/bert-large-portuguese-cased")
    embeddings = embedding_model.encode(docs_train, show_progress_bar=True)

    # --- Treina o modelo BERTopic ---
    topic_model_0 = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=10,
        nr_topics=70,
        n_gram_range=(1, 1),
        language="portuguese",
        verbose=True
    )

    topics_train_0, probs_train_0 = topic_model_0.fit_transform(docs_train, embeddings)

    # Saves topic information
    topic_info_df_0 = topic_model_0.get_topic_info()
    topic_info_df_0.to_csv("topic_info_0.csv", index=False, encoding='utf-8')
    
    # Contagem de tópicos válidos (exclui -1)
    valid_labels_0 = [t for t in topics_train_0 if t != -1]
    print(len(set(valid_labels_0)))

    # Reconstrói o dicionário usando o mesmo analyzer do BERTopic
    cleaned_docs_0  = topic_model_0._preprocess_text(docs_train)
    analyzer_0      = topic_model_0.vectorizer_model.build_analyzer()
    texts_tokens_0  = [analyzer_0(doc) for doc in cleaned_docs_0]
    dictionary_0    = Dictionary(texts_tokens_0)

    # Extrai tópicos e filtra vazios/outliers
    tp_0 = topic_model_0.get_topics()
    tp_0.pop(-1, None)
    keyword_list_0 = [[w for w, _ in kws_0] for kws_0 in tp_0.values()]
    keyword_list_0 = [kw for kw in keyword_list_0 if kw]
    if not keyword_list_0:
        raise ValueError("Nenhum tópico válido para coerência")

    # Calcula Coerência c_npmi
    coherence_model_0 = CoherenceModel(
        topics=keyword_list_0,
        texts=texts_tokens_0,
        dictionary=dictionary_0,
        coherence='c_npmi'
        )
    NC_0 = coherence_model_0.get_coherence()

    # Calcula Diversidade
    all_kw_0 = [w for kws_0 in keyword_list_0 for w in kws_0]
    ND_0 = len(set(all_kw_0)) / len(all_kw_0) if all_kw_0 else 0.0

    # Métricas ponderadas
    WC_0 = 0.8 * NC_0
    WD_0 = 0.2 * ND_0
    WS_0 = WC_0 + WD_0

    results_0 = pd.DataFrame([{
        'NC_0': NC_0,
        'ND_0': ND_0,
        'WC_0': WC_0,
        'WD_0': WD_0,
        'Weighted Score_0': WS_0,
    }])
    print(results_0)

    # --- Carrega e prepara dados de teste ---
    test_df_0 = pd.read_csv("test.csv", encoding='utf-8')
    docs_test_0, originals_test_0, valid_idx_0 = clean_and_prepare(test_df_0, "proposal_text")
    print("*******len(docs_test_0)")
    print(len(docs_test_0))

    # Faz inferência nos dados de teste
    print("Fazendo inferência nos dados de teste...")
    topics_test_0, probs_test_0 = topic_model_0.transform(docs_test_0)

    # Monta DataFrame de resultados alinhando índices originais
    result_df_0 = test_df_0.loc[valid_idx_0].copy()
    result_df_0['original_text'] = originals_test_0
    # Adiciona as colunas VCGE_N1 e VCGE_N2 diretamente do test_df
    result_df_0['VCGE_N1'] = test_df_0.loc[valid_idx_0, 'VCGE_N1'].values
    result_df_0['VCGE_N2'] = test_df_0.loc[valid_idx_0, 'VCGE_N2'].values
    result_df_0['topic']       = topics_test_0
    result_df_0['probability'] = [p.max() if p is not None else None for p in probs_test_0]

    # Seleciona apenas colunas relevantes e salva
    output_df_0 = result_df_0[['original_text', 'VCGE_N1', 'VCGE_N2', 'topic', 'probability']]
    output_df_0.to_csv("resultados_externo_0.csv", index=False, encoding='utf-8')

    print("Resultados salvos em: resultados_com_topico_0.csv")
    print("Informações de tópicos salvas em: topic_info_0.csv")
    print("\nProcesso concluído.")

    # Filtrar outliers e valores nulos em y_true
    mask_0 = (result_df_0['topic'] != -1) & (result_df_0['VCGE_N1'].notna())
    y_true_filt_0 = result_df_0.loc[mask_0, 'VCGE_N1']
    y_pred_filt_0 = result_df_0.loc[mask_0, 'topic']

    # Calcular métricas
    ari_0 = adjusted_rand_score(y_true_filt_0, y_pred_filt_0)
    nmi_0 = normalized_mutual_info_score(y_true_filt_0, y_pred_filt_0, average_method='arithmetic')
    vmetr_0 = v_measure_score(y_true_filt_0, y_pred_filt_0)

    print(f"Adjusted Rand Index (ARI): {ari_0:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi_0:.4f}")
    print(f"V-Measure: {vmetr_0:.4f}")

if __name__ == '__main__':
    main()