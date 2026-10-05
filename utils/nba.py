import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from requests.exceptions import ReadTimeout, ConnectionError

def gerar_lista_temporadas(ano_inicio=2000, ano_fim=2025):
    """
    Gera a lista de temporadas no formato exigido pela NBA API.
    Ex: 2000 -> '2000-01', 2023 -> '2023-24'
    """
    temporadas = []
    for ano in range(ano_inicio, ano_fim + 1):
        ano_seguinte = str(ano + 1)[-2:]
        temporadas.append(f"{ano}-{ano_seguinte}")
    return temporadas

def coletar_dados_nba(ano_inicio, ano_fim, output_dir="dataset/nba"):
    # Cria a pasta caso não exista
    os.makedirs(output_dir, exist_ok=True)
    temporadas = gerar_lista_temporadas(ano_inicio, ano_fim)
    
    for temporada in temporadas:
        arquivo_saida = os.path.join(output_dir, f"nba_bruto_{temporada}.csv")
        
        # Sistema de retomada: se o arquivo já existir, pula a temporada
        if os.path.exists(arquivo_saida):
            print(f"[PULANDO] Temporada {temporada} já foi baixada anteriormente.")
            continue
            
        print(f">> Baixando temporada {temporada}...")
        
        try:
            # 1. Puxando dados da Temporada Regular
            log_regular = leaguegamelog.LeagueGameLog(
                season=temporada, 
                season_type_all_star='Regular Season',
                timeout=60 # Aumentado para evitar timeout da API
            )
            df_regular = log_regular.get_data_frames()[0]
            df_regular['SEASON_TYPE'] = 'Regular Season'
            
            # Pausa de segurança entre as requisições
            time.sleep(2)
            
            # 2. Puxando dados dos Playoffs
            log_playoffs = leaguegamelog.LeagueGameLog(
                season=temporada, 
                season_type_all_star='Playoffs',
                timeout=60
            )
            df_playoffs = log_playoffs.get_data_frames()[0]
            df_playoffs['SEASON_TYPE'] = 'Playoffs'
            
            # 3. Une Regular Season e Playoffs
            df_temporada = pd.concat([df_regular, df_playoffs], ignore_index=True)
            
            # 4. Salva no diretório
            df_temporada.to_csv(arquivo_saida, index=False)
            print(f"   [SUCESSO] {temporada} salva com {len(df_temporada)} registros.")
            
        except (ReadTimeout, ConnectionError) as e:
            print(f"   [!] Erro de conexão na temporada {temporada}. Servidor demorou a responder.")
            print("   Aguardando 15 segundos antes de tentar a próxima...")
            time.sleep(15)
        except Exception as e:
            print(f"   [!] Falha inesperada ao processar {temporada}: {e}")
        
        # Pausa extra de 3 segundos entre cada temporada para respeitar o limite (rate limit) da NBA
        time.sleep(3)

if __name__ == "__main__":
    print("Iniciando coleta em lote dos dados brutos da NBA...")
    
    # Define o escopo de tempo (de 2000 até a temporada atual que começou em 2025)
    ANO_INICIO = 2000
    ANO_FIM = 2025
    
    coletar_dados_nba(ANO_INICIO, ANO_FIM, output_dir="data/dataset/nba")
    
    print("\n--- Processo de coleta finalizado! ---")