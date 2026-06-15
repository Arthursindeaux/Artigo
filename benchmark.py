import duckdb
import time
import csv
import os

# 1. Configurações do experimento
REPETICOES = 30
ARQUITETURAS = ['parquet', 'delta', 'iceberg']
CONSULTAS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
ARQUIVO_SAIDA = 'resultados/tempos_reais.csv'

os.makedirs('resultados', exist_ok=True)

# 2. Inicializa o DuckDB e carrega as extensões
print("Iniciando motor DuckDB e instalando extensões (isso pode levar alguns segundos na 1ª vez)...")
con = duckdb.connect(database=':memory:')
con.execute("INSTALL iceberg; LOAD iceberg;")
con.execute("INSTALL delta; LOAD delta;")
print("DuckDB pronto!")

def obter_query(arquitetura, consulta):
    """
    Aqui nós definimos o SQL que será rodado.
    Como os caminhos dos arquivos mudam dependendo da arquitetura, 
    ajustamos o 'FROM' dinamicamente.
    """
    # Exemplo de como ficarão as tabelas (precisaremos gerar isso depois):
    caminho_parquet = "'dados/tabela.parquet'"
    caminho_delta = "delta_scan('dados/tabela_delta')"
    caminho_iceberg = "iceberg_scan('dados/tabela_iceberg', 'allow_moved_paths=true')"
    
    tabela_alvo = caminho_parquet # padrão
    
    if arquitetura == 'delta':
        tabela_alvo = caminho_delta
    elif arquitetura == 'iceberg':
        tabela_alvo = caminho_iceberg

    # Aqui entram as suas consultas de verdade do artigo
    if consulta == 'Q1':
        return f"SELECT count(*) FROM {tabela_alvo}"
    elif consulta == 'Q2':
        # Exemplo: return f"SELECT coluna1, AVG(coluna2) FROM {tabela_alvo} GROUP BY coluna1"
        return f"SELECT * FROM {tabela_alvo} LIMIT 1"
    else:
        return f"SELECT 1" # Fallback para as outras queries não configuradas

def rodar_benchmark():
    with open(ARQUIVO_SAIDA, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['arquitetura', 'consulta', 'repeticao', 'tempo_s'])
        
        print("-" * 30)
        
        for arq in ARQUITETURAS:
            for cons in CONSULTAS:
                # Pega o comando SQL correspondente
                query = obter_query(arq, cons)
                print(f"Rodando {cons} no {arq} ({REPETICOES} vezes)...")
                
                for rep in range(1, REPETICOES + 1):
                    inicio = time.time()
                    
                    try:
                        # O DUCKDB EXECUTA A CONSULTA AQUI
                        con.execute(query).fetchall() 
                    except Exception as e:
                        print(f"Erro na {cons} ({arq}): {e}")
                        break # Se der erro, para a repetição para você arrumar
                    
                    fim = time.time()
                    tempo_gasto = fim - inicio
                    
                    writer.writerow([arq, cons, rep, round(tempo_gasto, 4)])
                    
    print("-" * 30)
    print(f"Benchmark finalizado! Resultados salvos em {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    # Importante: Se rodar agora sem os arquivos gerados, vai dar erro de arquivo não encontrado!
    print("Script configurado. Aguardando geração dos dados base.")
    # rodar_benchmark() # Descomente essa linha quando os dados existirem