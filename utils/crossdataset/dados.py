import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling  # Importação necessária para o Pool
import json
import time
import copy

load_dotenv()

# Configuração base
db_config_base = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST')
}

# Criar os Pools de ligação (mantém as portas TCP abertas e recicla-as)
config_nbb = db_config_base.copy()
config_nbb['database'] = 'db'
pool_nbb = pooling.MySQLConnectionPool(pool_name="nbb_pool", pool_size=5, **config_nbb)

config_nba = db_config_base.copy()
config_nba['database'] = 'nba_db'
pool_nba = pooling.MySQLConnectionPool(pool_name="nba_pool", pool_size=5, **config_nba)

def conectar(banco_nome):
    """Pede uma ligação emprestada ao pool em vez de criar uma nova porta TCP no Windows"""
    if banco_nome == 'db':
        return pool_nbb.get_connection()
    else:
        return pool_nba.get_connection()

def converter_para_minuto(estatisticas):
    estat_convertida = {}
    cols_volume = [
        'Pts', '3P', '2P', 'LL', 'RT', 'RO', 'RD', 'AS', 'ER', 
        '3PC', '3PT', '2PC', '2PT', 'LLC', 'LLT', 'EN', 
        'BR', 'TO', 'FC', 'ET', 'VI', 'EF'
    ]
    
    minutos = estatisticas.get('Min', 48.0)
    if minutos == 0.0:
        minutos = 40.0
        
    for key, value in estatisticas.items():
        if key in cols_volume:
            estat_convertida[key] = value / minutos
        else:
            estat_convertida[key] = value
            
    return estat_convertida

# Adicione o parâmetro 'banco' nas funções de consulta
def get_jogos_temporada(ano, banco):
    time.sleep(0.005)
    # AQUI ESTÁ O SEGREDO: Ele vai usar o banco que o script mandou ('db' ou 'nba_db')
    conn = conectar(banco)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE ano = %s ORDER BY data"
        cursor.execute(query, (ano,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def calcular_media_estatisticas(jogos, equipe, num_jogos_passados_media=8):
    if len(jogos) > num_jogos_passados_media:
        jogos = jogos[-num_jogos_passados_media:]
    
    total_estatisticas = None
    for jogo in jogos:
        if jogo['equipe_casa'] == equipe:
            estatisticas = json.loads(jogo['estatisticas_casa'])
        else:
            estatisticas = json.loads(jogo['estatisticas_visitantes'])
            
        estatisticas = converter_para_minuto(estatisticas)
            
        if total_estatisticas is None:
            total_estatisticas = estatisticas
        else:
            for key, value in estatisticas.items():
                total_estatisticas[key] += value
    
    num_jogos = len(jogos)
    if num_jogos > 0:
        for key in total_estatisticas:
            total_estatisticas[key] /= num_jogos
    
    return total_estatisticas

# Passe o banco para as buscas de médias
def get_media_estatisticas_time_teste(equipe, data_jogo, temporada, banco, num_jogos_passados_media=8):
    conn = conectar(banco)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT equipe_casa, equipe_visitante, estatisticas_casa, estatisticas_visitantes FROM jogos
            WHERE (equipe_casa = %s OR equipe_visitante = %s) 
            AND data < %s
            AND ano = %s
            ORDER BY data
        """
        cursor.execute(query, (equipe, equipe, data_jogo, temporada))
        jogos = cursor.fetchall()
        
        if jogos:
            return calcular_media_estatisticas(jogos, equipe, num_jogos_passados_media)
        else:
            return {}
    finally:
        cursor.close()
        conn.close()

def get_media_estatisticas_time_treino(equipe, data_jogo, temporada, banco, num_jogos_passados_media=8):
    conn = conectar(banco)
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT equipe_casa, equipe_visitante, estatisticas_casa, estatisticas_visitantes FROM jogos
            WHERE (equipe_casa = %s OR equipe_visitante = %s) 
            AND data < %s 
            AND ano = %s 
            ORDER BY data
        """
        cursor.execute(query, (equipe, equipe, data_jogo, temporada))
        jogos = cursor.fetchall()
        
        if jogos:
            return calcular_media_estatisticas(jogos, equipe, num_jogos_passados_media)
        else:
            return {}
    finally:
        cursor.close()
        conn.close()

# O formatar médias também precisa saber de qual banco puxar o histórico
def formatar_medias(jogos, isTreino, banco, num_jogos_passados_media=10):
    jogos_nova = copy.deepcopy(jogos)
    
    for jogo in jogos_nova:
        equipe_casa = jogo['equipe_casa']
        equipe_visitante = jogo['equipe_visitante']
        data_jogo = jogo['data']
        temporada_atual = jogo['ano'] 
        
        if isTreino:
            media_casa = get_media_estatisticas_time_treino(equipe_casa, data_jogo, temporada_atual, banco, num_jogos_passados_media)
            media_visitante = get_media_estatisticas_time_treino(equipe_visitante, data_jogo, temporada_atual, banco, num_jogos_passados_media)
        else:
            media_casa = get_media_estatisticas_time_teste(equipe_casa, data_jogo, temporada_atual, banco, num_jogos_passados_media)
            media_visitante = get_media_estatisticas_time_teste(equipe_visitante, data_jogo, temporada_atual, banco, num_jogos_passados_media)
        
        jogo['estatisticas_casa'] = media_casa
        jogo['estatisticas_visitantes'] = media_visitante
        
    return jogos_nova