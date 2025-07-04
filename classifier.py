# main.py
import ollama
import pandas as pd
from tqdm import tqdm
import re
import time

VCGE_TAXONOMY = {
    # Nível 1 (21 categorias principais)
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

def clean_text(text):
    # Limpeza adaptada para propostas governamentais
    text = re.sub(r'\n+', ' ', text)  # Remove quebras de linha múltiplas
    text = re.sub(r'\s{2,}', ' ', text)  # Espaços extras
    text = re.sub(r'\b(Item|Parágrafo|Artigo)\b.*?:?', '', text, flags=re.IGNORECASE)
    return text.strip()[:2000]  # Context window do Gemma 12B

def validate_category(response, allowed_options):
    if response in allowed_options:
        return response
    # Tenta encontrar correspondência aproximada
    matches = [opt for opt in allowed_options if response.lower() in opt.lower()]
    return matches[0] if matches else None

def classify_with_logging(text, options, level):
    prompt = f"""
    CLASSIFIQUE este texto usando APENAS UM destes termos oficiais do VCGE: 
    {", ".join(options)}.
    
    TEXTO: {text[:1500]}...
    
    Responda APENAS com o termo exato. Nada além do termo.
    """
    
    start_time = time.time()
    
    try:
        response = ollama.generate(
            model='gemma3:12b',
            prompt=prompt,
            options={'temperature': 0.2, 'num_ctx': 2048}
        )['response'].strip()
        
        print(f"\n[LLM DEBUG] Nível {level}:")
        print(f"Prompt: {prompt[:200]}...")
        print(f"Resposta bruta: {response}")
        print(f"Tempo: {time.time() - start_time:.2f}s")
        
        validated = validate_category(response, options)
        print(f"Validação: {validated}\n")
        
        return validated
        
    except Exception as e:
        print(f"\n[ERRO] Nível {level}: {str(e)}")
        return None

def classify_proposal(text):
    text = clean_text(text)
    
    # Classificação Nível 1
    nivel1_options = list(VCGE_TAXONOMY.keys())
    nivel1 = classify_with_logging(text, nivel1_options, 1)
    
    if not nivel1:
        return (None, None)
    
    # Classificação Nível 2
    nivel2_options = VCGE_TAXONOMY[nivel1]
    if nivel2_options:
        nivel2 = classify_with_logging(text, nivel2_options, 2)
        return (nivel1, nivel2)
    
    return (nivel1, "Sem subcategoria")

def main():
    print("Iniciando classificação VCGE...")
    print("="*50)
    
    # Carregar dados
    df = pd.read_csv('test.csv', encoding='utf-8-sig')

    print(len(df))
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classificando propostas"):
        result = classify_proposal(row['proposal_text'])
        results.append(result)
        
        # Print resumido
        print(f"\nProposta {idx+1}:")
        print(f"Nível 1: {result[0]}")
        print(f"Nível 2: {result[1]}")
        print("-"*50)
    
    # Salvar resultados
    df[['VCGE_N1', 'VCGE_N2']] = pd.DataFrame(results, columns=['VCGE_N1', 'VCGE_N2'])
    df.to_csv('resultados_classificados.csv', index=False, encoding='utf-8-sig')
    print("\nClassificação concluída! Resultados salvos em 'resultados_classificados.csv'")

if __name__ == "__main__":
    main()