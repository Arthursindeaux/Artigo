import os
import urllib.request
from pyspark.sql import SparkSession

print("Configurando ambiente para rodar Spark no Windows...")

# ==========================================
# 1. CORREÇÃO DO WINUTILS
# ==========================================
hadoop_home = os.path.abspath("hadoop_env")
bin_dir = os.path.join(hadoop_home, "bin")
os.makedirs(bin_dir, exist_ok=True)

winutils_path = os.path.join(bin_dir, "winutils.exe")
hadoop_path = os.path.join(bin_dir, "hadoop.dll")

url_winutils = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.2.2/bin/winutils.exe"
url_hadoop = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.2.2/bin/hadoop.dll"

try:
    if not os.path.exists(winutils_path): urllib.request.urlretrieve(url_winutils, winutils_path)
    if not os.path.exists(hadoop_path): urllib.request.urlretrieve(url_hadoop, hadoop_path)
except:
    pass

os.environ["HADOOP_HOME"] = hadoop_home
os.environ["hadoop.home.dir"] = hadoop_home
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

# ==========================================
# 2. INICIALIZAÇÃO DO SPARK E CONVERSÃO
# ==========================================
print("Iniciando o Spark com as extensões do Lakehouse (pode levar 1 minuto)...")

caminho_base = os.path.abspath("dados")
caminho_parquet = os.path.join(caminho_base, "telemetria_bruta.parquet")
caminho_delta = os.path.join(caminho_base, "telemetria_delta")
caminho_iceberg = os.path.join(caminho_base, "telemetria_iceberg")

spark = SparkSession.builder \
    .appName("GerarLakehouse") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.3") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension,org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", caminho_iceberg) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("-" * 40)
print("1. Lendo os dados Parquet brutos...")
df = spark.read.parquet(caminho_parquet)

print("2. Convertendo e salvando no formato Delta Lake...")
df.write.format("delta").mode("overwrite").save(caminho_delta)

print("3. Convertendo e salvando no formato Apache Iceberg...")
# Criar o banco de dados virtual do Iceberg primeiro
spark.sql("CREATE DATABASE IF NOT EXISTS local.db")
# Salvar COMO TABELA no catálogo do Iceberg
df.write.format("iceberg").mode("overwrite").saveAsTable("local.db.telemetria")

print("-" * 40)
print("CONVERSÃO CONCLUÍDA COM SUCESSO!")

spark.stop()