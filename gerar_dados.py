import pandas as pd
import numpy as np
import os

# Configurações extraídas da Seção 3.1 do seu artigo
np.random.seed(42) # Semente fixa para reprodutibilidade
NUM_NODES = 64
DIAS = 3 # 3 dias geram ~1.6 milhões de linhas. Bom volume para começar.
INTERVALO_SEG = 10
PASTA_SAIDA = 'dados'

print(f"Iniciando geração de telemetria para {NUM_NODES} nós durante {DIAS} dias...")

# 1. Gerar eixo de tempo
timestamps = pd.date_range(
    start='2024-01-01', 
    periods=(DIAS * 24 * 60 * 60) // INTERVALO_SEG, 
    freq=f'{INTERVALO_SEG}s'
)
num_amostras = len(timestamps)

# 2. Definir nós quentes (10% do cluster trabalha mais pesado)
nos_quentes = np.random.choice(range(1, NUM_NODES + 1), size=int(NUM_NODES * 0.1), replace=False)

dados = []

print("Calculando modelos estatísticos por nó (CPU, Memória, Disco, Rede)...")
for node_id in range(1, NUM_NODES + 1):
    is_hot = node_id in nos_quentes
    
    # --- CPU: padrão diurno (senoide) + ruído + picos ---
    horas = timestamps.hour + timestamps.minute / 60.0
    # Senoide com pico à tarde
    cpu_base = 40 + 20 * np.sin(np.pi * (horas - 8) / 12) 
    ruido = np.random.normal(0, 5, num_amostras)
    # Processo de Poisson para picos esporádicos
    picos = np.random.poisson(0.005, num_amostras) * 40 
    
    cpu = cpu_base + ruido + picos
    if is_hot:
        cpu += 25 # Nós quentes têm uso de CPU mais alto
    cpu = np.clip(cpu, 0, 100) # Limita entre 0% e 100%
    
    # --- Memória: Passeio aleatório limitado ---
    # Começa com 32GB (32000 MB) e vai variando
    passos_mem = np.random.normal(0, 50, num_amostras)
    mem_base = 32000 - np.abs(np.cumsum(passos_mem))
    mem = np.clip(mem_base, 1024, 32000) # Nunca cai abaixo de 1GB
    
    # --- Rede e Disco: Rajadas correlacionadas ---
    # 95% do tempo o uso é baixo, 5% do tempo há grandes rajadas
    prob_rajada = np.random.rand(num_amostras)
    rajadas = prob_rajada > 0.95
    
    rede_bytes = np.where(rajadas, np.random.normal(1e8, 1e7, num_amostras), np.random.normal(1e5, 1e4, num_amostras))
    disco_bytes = rede_bytes * 0.8 # Correlação forte entre rede e disco
    disco_fila = np.where(rajadas, np.random.randint(1, 15, num_amostras), 0)
    
    # Montar DataFrame do nó
    df_node = pd.DataFrame({
        'timestamp': timestamps,
        'node_id': f"node_{node_id:03d}",
        'cpu_percent': np.round(cpu, 2),
        'mem_avail_mb': np.round(mem, 2),
        'disk_bytes_sec': np.round(np.clip(disco_bytes, 0, None), 2),
        'disk_queue_len': disco_fila,
        'net_bytes_sec': np.round(np.clip(rede_bytes, 0, None), 2)
    })
    
    dados.append(df_node)

print("Concatenando todos os nós...")
df_final = pd.concat(dados, ignore_index=True)

# 3. Salvar como Parquet Bruto (O nosso baseline do artigo)
os.makedirs(PASTA_SAIDA, exist_ok=True)
caminho_parquet = os.path.join(PASTA_SAIDA, 'telemetria_bruta.parquet')

print(f"Salvando o Data Lake bruto em: {caminho_parquet} ...")
df_final.to_parquet(caminho_parquet, index=False, engine='pyarrow')

print("-" * 40)
print(f"SUCESSO! Geração concluída.")
print(f"Total de registros gerados: {len(df_final):,}")
print(f"Tamanho do arquivo gerado: {os.path.getsize(caminho_parquet) / (1024*1024):.2f} MB")
print("Este é o seu dataset 'Parquet Bruto'.")