import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder

# Lista de temporadas para extração.
# O formato exigido pela API é 'YYYY-YY', ex: '2023-24'
temporadas = [
    #'2008-09', '2009-10', '2010-11', 
    #'2011-12', '2012-13', '2013-14', '2014-15', 
    #'2015-16',
    #'2016-17', '2017-18',
    #'2018-19', '2019-20',
    #'2020-21', '2021-22', '2022-23', '2023-24'
    '2024-25'] 
dados_tabela = []

for season in temporadas:
    # Busca todos os jogos da liga ('00' = NBA) para a temporada especificada
    # season_type_nullable='Regular Season' isola a fase regular, ignorando Playoffs/Play-in
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, 
        league_id_nullable='00', 
        season_type_nullable='Regular Season'
    )
    
    df_games = gamefinder.get_data_frames()[0]
    
    if df_games.empty:
        print(f"Sem dados para a temporada {season}")
        continue
        
    # A API retorna duas linhas por jogo (uma para cada time). 
    # Para contar a quantidade real de partidas, removemos as duplicatas usando o GAME_ID.
    df_jogos_unicos = df_games.drop_duplicates(subset=['GAME_ID'])
    
    qtd_jogos = len(df_jogos_unicos)
    qtd_equipes = df_games['TEAM_ID'].nunique()
    
    # Aproximação de "Rodadas": verifica o máximo de jogos que um único time disputou
    jogos_por_time = df_games.groupby('TEAM_ID').size()
    max_rodadas = jogos_por_time.max()
    
    # Cálculo de Jogos Faltantes
    # Fórmula teórica: (Total de Equipes * Rodadas disputadas) / 2 times em quadra por jogo
    jogos_esperados = (qtd_equipes * max_rodadas) // 2
    jogos_faltantes = jogos_esperados - qtd_jogos
    
    # Garante que o número não seja negativo
    jogos_faltantes = max(0, jogos_faltantes)
    
    dados_tabela.append({
        'Temporada': season,
        'Jogos': qtd_jogos,
        'Rodadas': max_rodadas,
        'Equipes': qtd_equipes,
        'Jogos Faltantes': jogos_faltantes
    })
    
    # Pausa de 1 segundo para evitar bloqueio por rate limit da API (Timeout)
    time.sleep(1)

# Consolida os resultados em um DataFrame
df_resumo = pd.DataFrame(dados_tabela)
print(df_resumo.to_string(index=False))