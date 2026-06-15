#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analise.py — análise estatística do benchmark (SSCAD 2026)
==========================================================
ESTE é o script que cumpre o requisito da rubrica: IC 95% + TESTE DE HIPÓTESE.
(NÃO confundir com o perfmon_analyzer.py do professor, que só desenha gráficos
 de CPU/memória/disco do Windows — aquilo é monitoramento, não estatística.)

O QUE ELE FAZ, a partir dos tempos brutos de consulta:
  1. Por condição (arquitetura × consulta): n, média, desvio-padrão (σ),
     mediana e INTERVALO DE CONFIANÇA DE 95% (t de Student).
  2. Testa NORMALIDADE de cada grupo (Shapiro–Wilk, α = 0,05).
  3. Faz as COMPARAÇÕES par-a-par por consulta e escolhe o teste sozinho:
        - ambos normais  -> t de Welch  (versão robusta do t de Student)
        - caso contrário  -> Mann–Whitney (o "Wilcoxon" para amostras independentes)
  4. Corrige múltiplas comparações com Holm–Bonferroni.
  5. Calcula tamanho de efeito (Cliff's delta + redução % da mediana).
  6. Gera os gráficos: boxplots e barras com IC 95%.

ENTRADA (CSV), uma linha por execução válida do benchmark:
    arquitetura,consulta,repeticao,tempo_s
    parquet_bruto,Q1,1,2.131
    delta,Q1,1,0.872
    iceberg,Q1,1,0.910
    ...
Arquiteturas esperadas: parquet_bruto, delta, iceberg.

USO:
    # 1) Gerar um CSV de DEMONSTRAÇÃO (dados falsos, só para você ver funcionando):
    python analise.py --demo resultados/tempos.csv

    # 2) Rodar a análise sobre um CSV (de demo OU real):
    python analise.py --entrada resultados/tempos.csv --saida resultados

Quando o experimento real estiver pronto, é só sobrescrever resultados/tempos.csv
pelos tempos medidos de verdade e rodar o passo 2 de novo. Nada mais muda.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")          # não precisa de tela; salva em arquivo
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
ALFA = 0.05
ORDEM_ARQ = ["parquet_bruto", "delta", "iceberg"]
ROTULO = {
    "parquet_bruto": "Parquet bruto",
    "delta":         "Delta Lake",
    "iceberg":       "Apache Iceberg",
}


# ---------------------------------------------------------------------------
# Funções estatísticas
# ---------------------------------------------------------------------------
def ic95(x: np.ndarray):
    """Intervalo de confiança de 95% da média, via distribuição t de Student."""
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, m
    se = x.std(ddof=1) / np.sqrt(n)          # erro-padrão da média
    t = stats.t.ppf(0.975, n - 1)            # valor crítico bicaudal
    return m - t * se, m + t * se


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: tamanho de efeito não paramétrico em [-1, 1].
    |δ|<0,147 desprezível; <0,33 pequeno; <0,474 médio; senão grande."""
    diff = x[:, None] - y[None, :]
    return float(np.sign(diff).sum()) / (x.size * y.size)


def holm(pvalores):
    """Correção de Holm–Bonferroni (step-down). Retorna os p ajustados."""
    p = np.asarray(pvalores, dtype=float)
    m = p.size
    if m == 0:
        return p
    ordem = np.argsort(p)
    ajust = np.empty(m)
    corrente = 0.0
    for k, idx in enumerate(ordem):
        val = (m - k) * p[idx]
        corrente = max(corrente, val)         # impõe monotonicidade
        ajust[idx] = min(1.0, corrente)
    return ajust


def eh_normal(x: np.ndarray) -> bool:
    """Shapiro–Wilk. Precisa de n>=3; com variância nula trata como não-normal."""
    if x.size < 3 or np.allclose(x, x[0]):
        return False
    return stats.shapiro(x).pvalue > ALFA


# ---------------------------------------------------------------------------
# Gerador de dados de DEMONSTRAÇÃO (para testar sem experimento real)
# ---------------------------------------------------------------------------
def gerar_demo(caminho: Path, reps: int = 30, seed: int = 42):
    """Cria um CSV plausível: Parquet bruto mais lento; Delta/Iceberg mais rápidos.
    Uma das consultas é deliberadamente assimétrica para exercitar o Mann–Whitney."""
    rng = np.random.default_rng(seed)
    consultas = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    # média-base de tempo (s) por arquitetura, por consulta
    base = {
        "parquet_bruto": [2.10, 3.40, 1.80, 4.20, 2.60],
        "delta":         [0.88, 1.30, 0.95, 1.55, 1.10],
        "iceberg":       [0.92, 1.42, 0.90, 1.60, 1.05],
    }
    linhas = []
    for qi, q in enumerate(consultas):
        for arq in ORDEM_ARQ:
            mu = base[arq][qi]
            if q == "Q4":   # distribuição com cauda (log-normal) -> não-normal
                amostras = mu * rng.lognormal(mean=0.0, sigma=0.25, size=reps)
            else:           # ruído gaussiano leve -> normal
                amostras = rng.normal(mu, mu * 0.07, size=reps)
            amostras = np.clip(amostras, 0.01, None)
            for r, t in enumerate(amostras, 1):
                linhas.append((arq, q, r, round(float(t), 4)))
    df = pd.DataFrame(linhas, columns=["arquitetura", "consulta", "repeticao", "tempo_s"])
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    print(f"[demo] CSV de demonstração escrito em: {caminho}  "
          f"({len(df)} linhas = {len(consultas)} consultas × {len(ORDEM_ARQ)} arqs × {reps} reps)")


# ---------------------------------------------------------------------------
# Núcleo da análise
# ---------------------------------------------------------------------------
def tabela_descritiva(df: pd.DataFrame) -> pd.DataFrame:
    regs = []
    for (arq, q), g in df.groupby(["arquitetura", "consulta"]):
        x = g["tempo_s"].to_numpy()
        lo, hi = ic95(x)
        regs.append({
            "consulta": q, "arquitetura": arq, "n": x.size,
            "media_s": x.mean(), "dp_s": x.std(ddof=1) if x.size > 1 else 0.0,
            "mediana_s": np.median(x), "ic95_low": lo, "ic95_high": hi,
            "normal": eh_normal(x),
        })
    cat = pd.CategoricalDtype(ORDEM_ARQ, ordered=True)
    out = pd.DataFrame(regs)
    out["arquitetura"] = out["arquitetura"].astype(cat)
    return out.sort_values(["consulta", "arquitetura"]).reset_index(drop=True)


def testes_pareados(df: pd.DataFrame) -> pd.DataFrame:
    regs = []
    for q, gq in df.groupby("consulta"):
        dados = {arq: gq.loc[gq.arquitetura == arq, "tempo_s"].to_numpy()
                 for arq in ORDEM_ARQ if (gq.arquitetura == arq).any()}
        for a, b in combinations([k for k in ORDEM_ARQ if k in dados], 2):
            xa, xb = dados[a], dados[b]
            normais = eh_normal(xa) and eh_normal(xb)
            if normais:
                teste = "t de Welch"
                stat, p = stats.ttest_ind(xa, xb, equal_var=False)
            else:
                teste = "Mann-Whitney"
                stat, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            md_a, md_b = np.median(xa), np.median(xb)
            reducao = (md_a - md_b) / md_a * 100.0 if md_a else 0.0  # b em relação a a
            regs.append({
                "consulta": q, "comparacao": f"{a} vs {b}", "teste": teste,
                "estatistica": float(stat), "p_bruto": float(p),
                "cliffs_delta": cliffs_delta(xa, xb),
                "reducao_mediana_%": reducao,
            })
    res = pd.DataFrame(regs)
    if not res.empty:
        res["p_holm"] = holm(res["p_bruto"].to_numpy())          # corrige toda a família
        res["significativo_5%"] = res["p_holm"] < ALFA
    return res


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def grafico_boxplots(df: pd.DataFrame, saida: Path):
    consultas = sorted(df["consulta"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(consultas) + 2, 5))
    cores = {"parquet_bruto": "#d1495b", "delta": "#3a86ff", "iceberg": "#2a9d8f"}
    largura = 0.22
    for j, arq in enumerate(ORDEM_ARQ):
        posicoes, caixas = [], []
        for i, q in enumerate(consultas):
            x = df[(df.consulta == q) & (df.arquitetura == arq)]["tempo_s"].to_numpy()
            if x.size:
                posicoes.append(i + (j - 1) * largura)
                caixas.append(x)
        bp = ax.boxplot(caixas, positions=posicoes, widths=largura,
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(cores[arq]); patch.set_alpha(0.75)
        for elem in ("medians", "whiskers", "caps"):
            for ln in bp[elem]:
                ln.set_color("#222222")
        ax.plot([], [], color=cores[arq], linewidth=8, alpha=0.75, label=ROTULO[arq])
    ax.set_xticks(range(len(consultas)))
    ax.set_xticklabels(consultas)
    ax.set_ylabel("Tempo de resposta (s)")
    ax.set_xlabel("Consulta")
    ax.set_title("Distribuição do tempo de resposta por consulta e arquitetura")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    print(f"[grafico] {saida}")


def grafico_barras_ic(desc: pd.DataFrame, saida: Path):
    consultas = sorted(desc["consulta"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(consultas) + 2, 5))
    cores = {"parquet_bruto": "#d1495b", "delta": "#3a86ff", "iceberg": "#2a9d8f"}
    largura = 0.25
    x = np.arange(len(consultas))
    for j, arq in enumerate(ORDEM_ARQ):
        medias, err_lo, err_hi = [], [], []
        for q in consultas:
            r = desc[(desc.consulta == q) & (desc.arquitetura == arq)]
            if not r.empty:
                m = r["media_s"].iloc[0]
                medias.append(m)
                err_lo.append(m - r["ic95_low"].iloc[0])
                err_hi.append(r["ic95_high"].iloc[0] - m)
            else:
                medias.append(0); err_lo.append(0); err_hi.append(0)
        ax.bar(x + (j - 1) * largura, medias, largura,
               yerr=[err_lo, err_hi], capsize=4,
               color=cores[arq], alpha=0.85, label=ROTULO[arq],
               error_kw={"ecolor": "#222222", "elinewidth": 1.2})
    ax.set_xticks(x); ax.set_xticklabels(consultas)
    ax.set_ylabel("Tempo médio (s)")
    ax.set_xlabel("Consulta")
    ax.set_title("Tempo médio por consulta com intervalo de confiança de 95%")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    print(f"[grafico] {saida}")


# ---------------------------------------------------------------------------
# Impressão amigável no terminal
# ---------------------------------------------------------------------------
def imprimir_descritiva(desc: pd.DataFrame):
    print("\n" + "=" * 78)
    print("  ESTATÍSTICA DESCRITIVA POR CONDIÇÃO  (média ± dp; mediana; IC 95%)")
    print("=" * 78)
    print(f"  {'Consulta':<9}{'Arquitetura':<16}{'n':>3}  {'média':>8}{'dp':>8}"
          f"{'mediana':>9}   {'IC 95% (s)':>20}  {'Normal?':>8}")
    print("-" * 78)
    for _, r in desc.iterrows():
        print(f"  {r['consulta']:<9}{ROTULO[r['arquitetura']]:<16}{int(r['n']):>3}  "
              f"{r['media_s']:>8.3f}{r['dp_s']:>8.3f}{r['mediana_s']:>9.3f}   "
              f"[{r['ic95_low']:>7.3f}, {r['ic95_high']:>7.3f}]  "
              f"{'sim' if r['normal'] else 'não':>8}")
    print("-" * 78)


def imprimir_testes(res: pd.DataFrame):
    print("\n" + "=" * 78)
    print("  TESTES DE HIPÓTESE PAR-A-PAR  (α = 0,05; p ajustado por Holm)")
    print("=" * 78)
    print(f"  {'Consulta':<9}{'Comparação':<26}{'Teste':<14}{'p (bruto)':>10}"
          f"{'p (Holm)':>10}  {'sig?':>5}")
    print("-" * 78)
    for _, r in res.iterrows():
        print(f"  {r['consulta']:<9}{r['comparacao']:<26}{r['teste']:<14}"
              f"{r['p_bruto']:>10.4g}{r['p_holm']:>10.4g}  "
              f"{'***' if r['significativo_5%'] else '—':>5}")
    print("-" * 78)
    print("  *** = diferença estatisticamente significativa após correção de Holm.")
    print("  Tamanho de efeito (Cliff's delta) e redução % da mediana estão no CSV.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Análise estatística dos tempos do benchmark (IC 95% + teste de hipótese).")
    ap.add_argument("--entrada", help="CSV com os tempos brutos.")
    ap.add_argument("--saida", default="resultados", help="Pasta de saída (padrão: resultados).")
    ap.add_argument("--demo", metavar="CAMINHO_CSV",
                    help="Gera um CSV de demonstração no caminho dado e sai.")
    args = ap.parse_args()

    if args.demo:
        gerar_demo(Path(args.demo))
        print("\nAgora rode a análise:\n"
              f"    python analise.py --entrada {args.demo} --saida resultados\n")
        return

    if not args.entrada:
        ap.error("informe --entrada CAMINHO.csv  (ou use --demo para gerar dados de teste)")

    entrada = Path(args.entrada)
    if not entrada.exists():
        raise SystemExit(f"[ERRO] arquivo não encontrado: {entrada}")

    df = pd.read_csv(entrada)
    faltando = {"arquitetura", "consulta", "repeticao", "tempo_s"} - set(df.columns)
    if faltando:
        raise SystemExit(f"[ERRO] colunas ausentes no CSV: {sorted(faltando)}")
    df["tempo_s"] = pd.to_numeric(df["tempo_s"], errors="coerce")
    df = df.dropna(subset=["tempo_s"])

    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)

    desc = tabela_descritiva(df)
    res = testes_pareados(df)

    imprimir_descritiva(desc)
    if not res.empty:
        imprimir_testes(res)

    desc.to_csv(saida / "resumo_estatistico.csv", index=False)
    if not res.empty:
        res.to_csv(saida / "testes_hipotese.csv", index=False)
    grafico_boxplots(df, saida / "boxplots.png")
    grafico_barras_ic(desc, saida / "barras_ic95.png")

    # aviso sobre o mínimo de 30 repetições exigido pela rubrica
    poucos = desc[desc["n"] < 30]
    if not poucos.empty:
        print("\n[ATENÇÃO] Condições com menos de 30 repetições (a rubrica exige >= 30):")
        for _, r in poucos.iterrows():
            print(f"    - {r['consulta']} / {ROTULO[r['arquitetura']]}: n = {int(r['n'])}")

    print("\nArquivos gerados em:", saida.resolve())
    print("  - resumo_estatistico.csv   (vira a Tabela 3 do artigo)")
    print("  - testes_hipotese.csv      (vira a Tabela 4 do artigo)")
    print("  - boxplots.png             (Figura 1)")
    print("  - barras_ic95.png          (Figura 2)")


if __name__ == "__main__":
    main()