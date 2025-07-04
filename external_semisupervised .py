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
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

# Stopwords and lemmatizer setup
nltk.download('stopwords')
stop_words = stopwords.words('portuguese')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

VCGE_TAXONOMY = {
    "Abastecimento": [],
    "Administração": [
        "Cadastro",
        "Compras governamentais",
        "Fiscalização do Estado",
        "Normas e Fiscalização",
        "Operações de dívida pública",
        "Orçamento",
        "Patrimônio",
        "Recursos humanos",
        "Serviços Públicos",
        "Outros em Administração"
    ],
    "Agropecuária, Pesca e Extrativismo": [
        "Defesa e vigilância sanitária",
        "Produção agropecuária",
        "Outros em Agropecuária"
    ],
    "Comércio e Serviços": [
        "Comercio externo",
        "Defesa do Consumidor",
        "Turismo",
        "Outros em Comércio e serviços"
    ],
    "Comunicações": [
        "Comunicações Postais",
        "Telecomunicações",
        "Outros em Comunicações"
    ],
    "Cultura": [
        "Difusão Cultural",
        "Patrimônio Cultural",
        "Outros em Cultura"
    ],
    "Defesa Nacional": [
        "Defesa Civil",
        "Defesa Militar",
        "Outros em Defesa Nacional"
    ],
    "Economia e Finanças": [
        "Defesa da concorrência",
        "Politica econômica",
        "Sistema Financeiro",
        "Outros em Economia e Finanças"
    ],
    "Educação": [
        "Educação básica",
        "Educação profissionalizante",
        "Educação superior",
        "Outros em Educação"
    ],
    "Energia": [
        "Combustíveis",
        "Energia elétrica",
        "Outros em Energia"
    ],
    "Esporte e Lazer": [
        "Esporte comunitário",
        "Esporte profissional",
        "Lazer",
        "Outros em Esporte e Lazer"
    ],
    "Habitação": [
        "Habitação Rural",
        "Habitação Urbana",
        "Outros em Habitação"
    ],
    "Indústria": [
        "Mineração",
        "Produção Industrial",
        "Propriedade Industrial",
        "Outros em Industria"
    ],
    "Infraestrutura e Fomento": [],
    "Meio ambiente": [
        "Água",
        "Biodiversidade",
        "Clima",
        "Preservação e Conservação Ambiental",
        "Outros em Meio Ambiente"
    ],
    "Pesquisa e Desenvolvimento": [
        "Difusão",
        "Outros em Pesquisa e Desenvolvimento"
    ],
    "Planejamento e Gestão": [],
    "Previdência Social": [
        "Combate a pobreza",
        "Previdência Básica",
        "Previdência Complementar",
        "Benefícios Sociais",
        "Outros em Previdência"
    ],
    "Proteção Social": [
        "Assistência à Criança e ao Adolescente",
        "Assistência ao Idoso",
        "Assistência ao Portador de Deficiência",
        "Cidadania",
        "Combate a desigualdade",
        "Outros em Proteção Social"
    ],
    "Relações Internacionais": [
        "Cooperação Internacional",
        "Relações Diplomáticas",
        "Outros em Relações Internacionais"
    ],
    "Saneamento": [
        "Saneamento Básico Rural",
        "Saneamento Básico Urbano",
        "Outros em Saneamento"
    ],
    "Saúde": [
        "Assistência Hospitalar e Ambulatorial",
        "Atendimento básico",
        "Combate a epidemias",
        "Defesa e vigilância sanitária",
        "Medicamentos e aparelhos",
        "Outros em Saúde"
    ],
    "Segurança e Ordem Pública": [
        "Defesa Civil",
        "Policiamento",
        "Outros em Segurança e Ordem Pública"
    ],
    "Trabalho": [
        "Empregabilidade",
        "Fomento ao Trabalho",
        "Proteção e Benefícios ao Trabalhador",
        "Relações de Trabalho",
        "Outros em Trabalho"
    ],
    "Transportes": [
        "Transporte Aéreo",
        "Transporte Ferroviário",
        "Transporte Hidroviário",
        "Transporte Rodoviário",
        "Outros em Transporte"
    ],
    "Urbanismo": [
        "Infraestrutura Urbana",
        "Serviços Urbanos",
        "Outros em Urbanismo"
    ]
}

# 1. seed_words: reforçar um conjunto representativo de termos de cada categoria
#    Aqui pegamos as próprias chaves de nível 1 e alguns termos de nível 2 como exemplo.
seed_words = []
for nivel1, subcats in VCGE_TAXONOMY.items():
    grupo = [ nivel1.lower() ]
    # pega no máximo duas subcategorias (ou menos, se não houver)
    for sub in subcats[:2]:
        grupo.append(sub.lower())
    seed_words.append(grupo)
seed_words

# 2. Construção automática de vcge_seed_topics
#    – Cada tópico semente é o nome da categoria + todas as suas subcategorias.
vcge_seed_topics = []
for nivel1, subcats in VCGE_TAXONOMY.items():
    termos = [ nivel1.lower() ] + [ sub.lower() for sub in subcats ]
    vcge_seed_topics.append(termos)
vcge_seed_topics

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

    ctfidf_model = ClassTfidfTransformer(
        seed_words=seed_words,
        seed_multiplier=2,
    )

    # --- Trains the BERTopic model ---
    topic_model = BERTopic(
        embedding_model=embedding_model,
        seed_topic_list=vcge_seed_topics,
        ctfidf_model=ctfidf_model,
        min_topic_size=10,
        nr_topics=70,
        n_gram_range=(1, 1),
        language="portuguese",
        verbose=True
    )

    topics_train, probs_train = topic_model.fit_transform(docs_train, embeddings)

    # Saves topic information
    topic_info_df = topic_model.get_topic_info()
    topic_info_df.to_csv("topic_info.csv", index=False, encoding='utf-8')    
    
    # Contagem de tópicos válidos (exclui -1)
    valid_labels = [t for t in topics_train if t != -1]
    print(len(set(valid_labels)))

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

    results = pd.DataFrame([{
        'NC': NC,
        'ND': ND,
        'WC': WC,
        'WD': WD,
        'Weighted Score': WS,
    }])
    print(results)

    # --- Carrega e prepara dados de teste ---
    test_df = pd.read_csv("test.csv", encoding='utf-8')
    docs_test, originals_test, valid_idx = clean_and_prepare(test_df, "proposal_text")
    print("*******len(docs_test)")
    print(len(docs_test))

    # Faz inferência nos dados de teste
    print("Fazendo inferência nos dados de teste...")
    topics_test, probs_test = topic_model.transform(docs_test)

    # Monta DataFrame de resultados alinhando índices originais
    result_df = test_df.loc[valid_idx].copy()
    result_df['original_text'] = originals_test
    # Adiciona as colunas VCGE_N1 e VCGE_N2 diretamente do test_df
    result_df['VCGE_N1'] = test_df.loc[valid_idx, 'VCGE_N1'].values
    result_df['VCGE_N2'] = test_df.loc[valid_idx, 'VCGE_N2'].values
    result_df['topic']       = topics_test
    result_df['probability'] = [p.max() if p is not None else None for p in probs_test]

    # Seleciona apenas colunas relevantes e salva
    output_df = result_df[['original_text', 'VCGE_N1', 'VCGE_N2', 'topic', 'probability']]
    output_df.to_csv("resultados_externo.csv", index=False, encoding='utf-8')

    print("Resultados salvos em: resultados_com_topico.csv")
    print("Informações de tópicos salvas em: topic_info.csv")
    print("\nProcesso concluído.")

    # Filtrar outliers e valores nulos em y_true
    mask = (result_df['topic'] != -1) & (result_df['VCGE_N1'].notna())
    y_true_filt = result_df.loc[mask, 'VCGE_N1']
    y_pred_filt = result_df.loc[mask, 'topic']

    # Calcular métricas
    ari = adjusted_rand_score(y_true_filt, y_pred_filt)
    nmi = normalized_mutual_info_score(y_true_filt, y_pred_filt, average_method='arithmetic')
    vmetr = v_measure_score(y_true_filt, y_pred_filt)

    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"V-Measure: {vmetr:.4f}")

if __name__ == '__main__':
    main()