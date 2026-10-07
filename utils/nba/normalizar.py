import pandas as pd
import numpy as np
import os
import json

def tratar_dataset_nba_puro(caminho_entrada, caminho_saida, temporada_str, num_jogos_passados_media=8):
    if not os.path.exists(caminho_entrada):
        return None

    print(f">> Processando: {temporada_str}")
    df_bruto = pd.read_csv(caminho_entrada)

    # --- 1. SEPARAR MANDANTES E VISITANTES ---
    df_casa = df_bruto[df_bruto['MATCHUP'].str.contains('vs.')].copy()
    df_visit = df_bruto[df_bruto['MATCHUP'].str.contains('@')].copy()

    df_casa = df_casa.add_suffix('_casa')
    df_visit = df_visit.add_suffix('_visitante')

    df_casa = df_casa.rename(columns={
        'GAME_ID_casa': 'GAME_ID', 
        'GAME_DATE_casa': 'data'
    })
    df_visit = df_visit.rename(columns={'GAME_ID_visitante': 'GAME_ID'})

    # --- 2. JUNTAR CONFRONTO ---
    df_confrontos = pd.merge(df_casa, df_visit, on='GAME_ID')

    df_confrontos['equipe_casa'] = df_confrontos['TEAM_NAME_casa']
    df_confrontos['equipe_visitante'] = df_confrontos['TEAM_NAME_visitante']
    df_confrontos['placar_casa'] = df_confrontos['PTS_casa']
    df_confrontos['placar_visitante'] = df_confrontos['PTS_visitante']
    df_confrontos['data'] = pd.to_datetime(df_confrontos['data'])
    
    # Padroniza a string do ano (ex: 2000-01) para a coluna temporada
    df_confrontos['ano'] = temporada_str

    # --- 3. MAPEAMENTO E CÁLCULO DE ESTATÍSTICAS ---
    cols_estatisticas = [
        'Pts', '3P', '2P', 'LL', 'RT', 'RO', 'RD', 'AS', 'ER', 'IA%', 
        '3PC', '3PT', '3P%', '2PC', '2PT', '2P%', 'LLC', 'LLT', 'LL%', 
        'EN', 'BR', 'B/E', 'TO', 'FC', 'T/FC', 'ET', 'VI', 'EF'
    ]

    for prefixo in ['_casa', '_visitante']:
        # Mapeamento Direto
        df_confrontos[f'Pts{prefixo}'] = df_confrontos[f'PTS{prefixo}']
        df_confrontos[f'3PC{prefixo}'] = df_confrontos[f'FG3M{prefixo}']
        df_confrontos[f'3PT{prefixo}'] = df_confrontos[f'FG3A{prefixo}']
        df_confrontos[f'LLC{prefixo}'] = df_confrontos[f'FTM{prefixo}']
        df_confrontos[f'LLT{prefixo}'] = df_confrontos[f'FTA{prefixo}']
        df_confrontos[f'RO{prefixo}']  = df_confrontos[f'OREB{prefixo}']
        df_confrontos[f'RD{prefixo}']  = df_confrontos[f'DREB{prefixo}']
        df_confrontos[f'RT{prefixo}']  = df_confrontos[f'REB{prefixo}']
        df_confrontos[f'AS{prefixo}']  = df_confrontos[f'AST{prefixo}']
        df_confrontos[f'BR{prefixo}']  = df_confrontos[f'STL{prefixo}']
        df_confrontos[f'TO{prefixo}']  = df_confrontos[f'BLK{prefixo}']
        df_confrontos[f'ER{prefixo}']  = df_confrontos[f'TOV{prefixo}']
        df_confrontos[f'FC{prefixo}']  = df_confrontos[f'PF{prefixo}']

        # Mapeamento Derivado
        df_confrontos[f'3P{prefixo}']  = df_confrontos[f'3PC{prefixo}'] * 3
        df_confrontos[f'LL{prefixo}']  = df_confrontos[f'LLC{prefixo}'] * 1
        df_confrontos[f'2PC{prefixo}'] = df_confrontos[f'FGM{prefixo}'] - df_confrontos[f'FG3M{prefixo}']
        df_confrontos[f'2PT{prefixo}'] = df_confrontos[f'FGA{prefixo}'] - df_confrontos[f'FG3A{prefixo}']
        df_confrontos[f'2P{prefixo}']  = df_confrontos[f'2PC{prefixo}'] * 2

        # Porcentagens
        df_confrontos[f'3P%{prefixo}'] = df_confrontos[f'FG3_PCT{prefixo}'] * 100
        df_confrontos[f'LL%{prefixo}'] = df_confrontos[f'FT_PCT{prefixo}'] * 100
        df_confrontos[f'2P%{prefixo}'] = (df_confrontos[f'2PC{prefixo}'] / df_confrontos[f'2PT{prefixo}']).replace([np.inf, -np.inf], 0).fillna(0) * 100

        # Razões e Índices
        df_confrontos[f'IA%{prefixo}']   = (df_confrontos[f'AS{prefixo}'] / df_confrontos[f'ER{prefixo}']).replace([np.inf, -np.inf], 0).fillna(0)
        df_confrontos[f'B/E{prefixo}']   = (df_confrontos[f'BR{prefixo}'] / df_confrontos[f'ER{prefixo}']).replace([np.inf, -np.inf], 0).fillna(0)
        df_confrontos[f'T/FC{prefixo}']  = (df_confrontos[f'TO{prefixo}'] / df_confrontos[f'FC{prefixo}']).replace([np.inf, -np.inf], 0).fillna(0)

        # Dados ausentes na NBA e Eficiência
        df_confrontos[f'EN{prefixo}'] = 0
        df_confrontos[f'VI{prefixo}'] = 0
        df_confrontos[f'ET{prefixo}'] = df_confrontos[f'ER{prefixo}']
        df_confrontos[f'EF{prefixo}'] = (
            (df_confrontos[f'Pts{prefixo}'] + df_confrontos[f'RT{prefixo}'] + df_confrontos[f'AS{prefixo}'] + 
             df_confrontos[f'BR{prefixo}'] + df_confrontos[f'TO{prefixo}']) - 
            ((df_confrontos[f'FGA{prefixo}'] - df_confrontos[f'FGM{prefixo}']) + 
             (df_confrontos[f'LLT{prefixo}'] - df_confrontos[f'LLC{prefixo}']) + df_confrontos[f'ER{prefixo}'])
        )

    # --- 4. APLICAÇÃO DAS MÉDIAS MÓVEIS ---
    df_confrontos = df_confrontos.sort_values('data').reset_index(drop=True)

    df_stack_casa = df_confrontos[['GAME_ID', 'data', 'ano', 'equipe_casa']].copy()
    df_stack_casa.rename(columns={'equipe_casa': 'equipe'}, inplace=True)
    for c in cols_estatisticas: df_stack_casa[c] = df_confrontos[f'{c}_casa']

    df_stack_visit = df_confrontos[['GAME_ID', 'data', 'ano', 'equipe_visitante']].copy()
    df_stack_visit.rename(columns={'equipe_visitante': 'equipe'}, inplace=True)
    for c in cols_estatisticas: df_stack_visit[c] = df_confrontos[f'{c}_visitante']

    df_historico = pd.concat([df_stack_casa, df_stack_visit]).sort_values('data').reset_index(drop=True)

    def aplicar_media_movel(group):
        return group[cols_estatisticas].shift(1).rolling(window=num_jogos_passados_media, min_periods=1).mean()

    df_medias = df_historico.groupby(['ano', 'equipe'], group_keys=False).apply(aplicar_media_movel)
    df_historico[cols_estatisticas] = df_medias

    df_medias_casa = df_historico[['GAME_ID', 'equipe'] + cols_estatisticas].copy()
    df_medias_casa.columns = ['GAME_ID', 'equipe_casa'] + [f'{c}_casa' for c in cols_estatisticas]

    df_medias_visit = df_historico[['GAME_ID', 'equipe'] + cols_estatisticas].copy()
    df_medias_visit.columns = ['GAME_ID', 'equipe_visitante'] + [f'{c}_visitante' for c in cols_estatisticas]

    cols_to_drop = [f'{c}_casa' for c in cols_estatisticas] + [f'{c}_visitante' for c in cols_estatisticas]
    df_confrontos = df_confrontos.drop(columns=cols_to_drop)
    
    df_confrontos = df_confrontos.merge(df_medias_casa, on=['GAME_ID', 'equipe_casa'], how='left')
    df_confrontos = df_confrontos.merge(df_medias_visit, on=['GAME_ID', 'equipe_visitante'], how='left')

    df_confrontos = df_confrontos.dropna(subset=['Pts_casa', 'Pts_visitante'])

    # --- 5. ORDENAR AS COLUNAS (PADRÃO NBB) ---
    ordem_colunas = ['placar_casa', 'placar_visitante', 'data', 'ano', 'equipe_casa', 'equipe_visitante']
    for prefixo in ['_casa', '_visitante']:
        ordem_colunas.extend([
            f'Pts{prefixo}', f'3P{prefixo}', f'2P{prefixo}', f'LL{prefixo}', f'RT{prefixo}', 
            f'RO{prefixo}', f'RD{prefixo}', f'AS{prefixo}', f'ER{prefixo}', f'IA%{prefixo}', 
            f'3PC{prefixo}', f'3PT{prefixo}', f'3P%{prefixo}', f'2PC{prefixo}', f'2PT{prefixo}', 
            f'2P%{prefixo}', f'LLC{prefixo}', f'LLT{prefixo}', f'LL%{prefixo}', f'EN{prefixo}', 
            f'BR{prefixo}', f'B/E{prefixo}', f'TO{prefixo}', f'FC{prefixo}', f'T/FC{prefixo}', 
            f'ET{prefixo}', f'VI{prefixo}', f'EF{prefixo}'
        ])

    df_final = df_confrontos[ordem_colunas].copy()

    df_final.insert(3, 'round', 1) 
    df_final.insert(4, 'stage', 1)

    # Salva o CSV individual por segurança
    df_final.to_csv(caminho_saida, index=False)
    
    return df_final

def gerar_sql_insert_nba(df, caminho_saida_sql='utils/dataset/nba_dataset_completo.sql'):
    print(f"\n>> Gerando script SQL consolidado em: {caminho_saida_sql}")
    
    equipes = set(df['equipe_casa']).union(set(df['equipe_visitante']))
    anos = set(df['ano'])
    
    with open(caminho_saida_sql, 'w', encoding='utf-8') as f:
        # ==========================================
        # 1. CRIAÇÃO DO BANCO DE DADOS E TABELAS
        # ==========================================
        f.write("-- Criação do Banco de Dados\n")
        f.write("CREATE DATABASE IF NOT EXISTS nba_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\n")
        f.write("USE nba_db;\n\n")

        f.write("-- Estrutura da tabela `equipe`\n")
        f.write("CREATE TABLE IF NOT EXISTS `equipe` (\n")
        f.write("  `equipe` VARCHAR(100) NOT NULL,\n")
        f.write("  PRIMARY KEY (`equipe`)\n")
        f.write(") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n\n")

        f.write("-- Estrutura da tabela `temporada`\n")
        f.write("CREATE TABLE IF NOT EXISTS `temporada` (\n")
        f.write("  `ano` VARCHAR(45) NOT NULL,\n")
        f.write("  PRIMARY KEY (`ano`)\n")
        f.write(") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n\n")

        f.write("-- Estrutura da tabela `jogos`\n")
        f.write("CREATE TABLE IF NOT EXISTS `jogos` (\n")
        f.write("  `id` INT NOT NULL AUTO_INCREMENT,\n")
        f.write("  `placar_casa` INT DEFAULT NULL,\n")
        f.write("  `placar_visitante` INT DEFAULT NULL,\n")
        f.write("  `data` DATETIME DEFAULT NULL,\n")
        f.write("  `round` INT DEFAULT NULL,\n")
        f.write("  `stage` INT DEFAULT NULL,\n")
        f.write("  `ano` VARCHAR(45) NOT NULL,\n")
        f.write("  `equipe_casa` VARCHAR(100) NOT NULL,\n")
        f.write("  `equipe_visitante` VARCHAR(100) NOT NULL,\n")
        f.write("  `estatisticas_casa` LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,\n")
        f.write("  `estatisticas_visitantes` LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,\n")
        f.write("  PRIMARY KEY (`id`),\n")
        f.write("  FOREIGN KEY (`ano`) REFERENCES `temporada` (`ano`),\n")
        f.write("  FOREIGN KEY (`equipe_casa`) REFERENCES `equipe` (`equipe`),\n")
        f.write("  FOREIGN KEY (`equipe_visitante`) REFERENCES `equipe` (`equipe`),\n")
        f.write("  CONSTRAINT `jogos_chk_1` CHECK (json_valid(`estatisticas_casa`)),\n")
        f.write("  CONSTRAINT `jogos_chk_2` CHECK (json_valid(`estatisticas_visitantes`))\n")
        f.write(") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n\n")

        # ==========================================
        # 2. INSERÇÃO DOS DADOS
        # ==========================================
        f.write("-- Povoando a tabela equipe\n")
        f.write("INSERT IGNORE INTO equipe (equipe) VALUES \n")
        valores_equipes = [f"('{eq}')" for eq in equipes]
        f.write(",\n".join(valores_equipes) + ";\n\n")

        f.write("-- Povoando a tabela temporada\n")
        f.write("INSERT IGNORE INTO temporada (ano) VALUES \n")
        valores_anos = [f"('{ano}')" for ano in anos]
        f.write(",\n".join(valores_anos) + ";\n\n")

        f.write("-- Povoando a tabela jogos\n")
        cols_estatisticas = [
            'Pts', '3P', '2P', 'LL', 'RT', 'RO', 'RD', 'AS', 'ER', 'IA%', 
            '3PC', '3PT', '3P%', '2PC', '2PT', '2P%', 'LLC', 'LLT', 'LL%', 
            'EN', 'BR', 'B/E', 'TO', 'FC', 'T/FC', 'ET', 'VI', 'EF'
        ]
        
        chunk_size = 500
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            
            f.write("INSERT INTO jogos (placar_casa, placar_visitante, data, round, stage, ano, equipe_casa, equipe_visitante, estatisticas_casa, estatisticas_visitantes) VALUES \n")
            
            valores_jogos = []
            for _, row in chunk.iterrows():
                estat_casa = {"JO": 1, "Min": 0.0}
                estat_visit = {"JO": 1, "Min": 0.0}
                
                for col in cols_estatisticas:
                    estat_casa[col] = float(round(row[f'{col}_casa'], 2)) if pd.notnull(row[f'{col}_casa']) else 0.0
                    estat_visit[col] = float(round(row[f'{col}_visitante'], 2)) if pd.notnull(row[f'{col}_visitante']) else 0.0

                placar_c = int(row['placar_casa'])
                placar_v = int(row['placar_visitante'])
                data_jogo = row['data'].strftime('%Y-%m-%d %H:%M:%S')
                rnd = int(row['round'])
                stg = int(row['stage'])
                ano = str(row['ano'])
                eq_c = str(row['equipe_casa']).replace("'", "''") 
                eq_v = str(row['equipe_visitante']).replace("'", "''")
                
                json_c = json.dumps(estat_casa)
                json_v = json.dumps(estat_visit)
                
                valores_jogos.append(
                    f"({placar_c}, {placar_v}, '{data_jogo}', {rnd}, {stg}, '{ano}', '{eq_c}', '{eq_v}', '{json_c}', '{json_v}')"
                )
            
            f.write(",\n".join(valores_jogos) + ";\n\n")
            
    print(">> Script SQL criado com sucesso! Tamanho final: aprox.", len(df), "jogos.")

def gerar_lista_temporadas(ano_inicio, ano_fim):
    temporadas = []
    for ano in range(ano_inicio, ano_fim + 1):
        ano_seguinte = str(ano + 1)[-2:]
        temporadas.append(f"{ano}-{ano_seguinte}")
    return temporadas

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("Iniciando o tratamento em lote das temporadas da NBA...\n")
    
    # Gera a lista de temporadas de 2000 até 2025
    temporadas = gerar_lista_temporadas(2000, 2025)
    K = 8
    
    # Lista para empilhar os DataFrames de cada temporada
    todos_os_jogos_nba = [] 
    
    for temporada in temporadas:
        arquivo_entrada = f'data/dataset/nba/nba_bruto_{temporada}.csv'
        arquivo_saida = f'data/dataset/nba/nba_treino_final_{temporada}.csv'
        
        # Garante que a pasta existe
        os.makedirs('data/dataset/nba', exist_ok=True)
        
        # Só processa se o arquivo original foi baixado
        if os.path.exists(arquivo_entrada):
            df_temporada = tratar_dataset_nba_puro(
                caminho_entrada=arquivo_entrada,
                caminho_saida=arquivo_saida,
                temporada_str=temporada,
                num_jogos_passados_media=K
            )
            
            if df_temporada is not None:
                todos_os_jogos_nba.append(df_temporada)
        else:
            print(f"[!] Arquivo bruto não encontrado: {arquivo_entrada}. Pulando...")
            
    # Junta todos os anos e gera o SQL consolidado
    if todos_os_jogos_nba:
        df_completo_nba = pd.concat(todos_os_jogos_nba, ignore_index=True)
        gerar_sql_insert_nba(df_completo_nba, 'utils/dataset/nba_dataset_completo.sql')
        
        print("\n--- Processo finalizado com sucesso! ---")
    else:
        print("\nNenhum dado processado.")