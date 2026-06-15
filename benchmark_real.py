import os
import time
import csv
from pyspark.sql import SparkSession

print("Configurando ambiente e inicializando Spark...")

# 1. Configuração do Hadoop/Windows (a mesma que funcionou antes)
hadoop_home = os.path.abspath("hadoop_env")
bin_dir = os.path.join(hadoop_home, "bin")
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["hadoop.home.dir"] = hadoop_home
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

caminho_base = os.path.abspath("dados")
caminho_iceberg = os.path.join(caminho_base, "telemetria_iceberg")

spark = SparkSession.builder \
    .appName("BenchmarkHPC") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.3") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension,org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", caminho_iceberg) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Registrar as tabelas para podermos usar SQL
print("Registrando tabelas...")
spark.read.parquet(os.path.join(caminho_base, "telemetria_bruta.parquet")).createOrReplaceTempView("tabela_parquet")
spark.read.format("delta").load(os.path.join(caminho_base, "telemetria_delta")).createOrReplaceTempView("tabela_delta")
# O Iceberg já está registrado como local.db.telemetria

# 3. Definição das Consultas (Baseado na sua Tabela 2)
consultas_sql = {
    "Q1": "SELECT node_id, AVG(cpu_percent) FROM {tabela} WHERE timestamp >= '2024-01-01 08:00:00' AND timestamp <= '2024-01-01 12:00:00' GROUP BY node_id",
    "Q2": "SELECT timestamp, cpu_percent, mem_avail_mb FROM {tabela} WHERE node_id = 'node_010' AND timestamp >= '2024-01-02 00:00:00' AND timestamp <= '2024-01-02 23:59:59'",
    "Q3": "SELECT node_id, percentile_approx(cpu_percent, 0.95) FROM {tabela} GROUP BY node_id",
    "Q4": "SELECT node_id, MAX(disk_queue_len) as max_q FROM {tabela} GROUP BY node_id ORDER BY max_q DESC LIMIT 5",
    "Q5": "SELECT * FROM {tabela} WHERE cpu_percent > 90 AND net_bytes_sec > 100000000"
}

arquiteturas = {
    "parquet_bruto": "tabela_parquet",
    "delta": "tabela_delta",
    "iceberg": "local.db.telemetria"
}

AQUECIMENTO = 5
REPETICOES = 30
ARQUIVO_SAIDA = 'resultados/tempos_reais.csv'
os.makedirs('resultados', exist_ok=True)

# 4. Loop de Execução
with open(ARQUIVO_SAIDA, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['arquitetura', 'consulta', 'repeticao', 'tempo_s'])
    
    print("-" * 40)
    print("INICIANDO BENCHMARK (Isso vai levar alguns minutos...)")
    print("-" * 40)

    for nome_arq, nome_tabela in arquiteturas.items():
        for id_query, sql_template in consultas_sql.items():
            query_pronta = sql_template.format(tabela=nome_tabela)
            
            print(f"[{nome_arq}] Preparando {id_query}...")
            
            # Fase de Aquecimento (Descartado)
            for _ in range(AQUECIMENTO):
                spark.sql(query_pronta).collect()
            
            print(f"[{nome_arq}] Medindo {id_query} ({REPETICOES} repetições)...")
            # Fase de Medição Oficial
            for rep in range(1, REPETICOES + 1):
                inicio = time.perf_counter()
                
                # O .collect() obriga o Spark a realmente executar a consulta
                spark.sql(query_pronta).collect() 
                
                fim = time.perf_counter()
                tempo_gasto = fim - inicio
                
                writer.writerow([nome_arq, id_query, rep, round(tempo_gasto, 4)])

print("-" * 40)
print(f"BENCHMARK CONCLUÍDO! Resultados salvos em: {ARQUIVO_SAIDA}")
spark.stop()