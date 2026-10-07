import os
from dotenv import load_dotenv
import mysql.connector
import json
import time
import copy

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração de conexão com o banco de dados usando variáveis de ambiente
db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_DATABASE')
}

def converter_para_minuto(estatisticas):
    """
    Recebe o dicionário de estatísticas de um jogo e converte
    as métricas de volume absoluto para taxas por minuto.
    """
    estat_convertida = {}
    
    # Lista exclusiva das variáveis de volume absoluto
    cols_volume = [
        'Pts', '3P', '2P', 'LL', 'RT', 'RO', 'RD', 'AS', 'ER', 
        '3PC', '3PT', '2PC', '2PT', 'LLC', 'LLT', 'EN', 
        'BR', 'TO', 'FC', 'ET', 'VI', 'EF'
    ]
    
    # Busca a minutagem. Se for 0.0 (como em jogos antigos), usa 200.0 (40 min * 5 jogadores) ou 240 pra nba
    minutos = estatisticas.get('Min', 48.0)
    if minutos == 0.0:
        minutos = 40.0
        
    for key, value in estatisticas.items():
        if key in cols_volume:
            # Divide pelo tempo de quadra
            estat_convertida[key] = value / minutos
        else:
            # Mantém intactas as taxas (ex: 3P%, IA%) e metadados (JO, Min)
            estat_convertida[key] = value
            
    return estat_convertida

def get_all_jogos():
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos ORDER BY data"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe(equipe):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_casa = %s OR equipe_visitante = %s ORDER BY data"
        cursor.execute(query, (equipe, equipe))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe_casa(equipe):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_casa = %s ORDER BY data"
        cursor.execute(query, (equipe,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe_visitante(equipe):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_visitante = %s ORDER BY data"
        cursor.execute(query, (equipe,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def calcular_media_estatisticas(jogos, equipe, num_jogos_passados_media=8):
    # Limita os jogos aos últimos num_jogos_passados_media jogos
    if len(jogos) > num_jogos_passados_media:
        jogos = jogos[-num_jogos_passados_media:]
    
    total_estatisticas = None
    for jogo in jogos:
        # Verifica se a equipe é a equipe da casa ou visitante
        if jogo['equipe_casa'] == equipe:
            estatisticas = json.loads(jogo['estatisticas_casa'])
        else:
            estatisticas = json.loads(jogo['estatisticas_visitantes'])
            
        # APLICA A CONVERSÃO ANTES DE SOMAR À MÉDIA GERAL
        estatisticas = converter_para_minuto(estatisticas)
            
        # Inicializa o total_estatisticas se for a primeira iteração
        if total_estatisticas is None:
            total_estatisticas = estatisticas
        else:
            # Acumula os valores das estatísticas convertidas
            for key, value in estatisticas.items():
                total_estatisticas[key] += value
    
    # Calcula a média
    num_jogos = len(jogos)
    if num_jogos > 0:
        for key in total_estatisticas:
            total_estatisticas[key] /= num_jogos
    
    return total_estatisticas

def get_media_estatisticas_time_teste(equipe, data_jogo, temporada , num_jogos_passados_media=8):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
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

def get_media_estatisticas_time_treino(equipe, data_jogo, temporada, num_jogos_passados_media=8):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)
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

def get_jogos_temporada(ano):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE ano = %s ORDER BY data"
        cursor.execute(query, (ano,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def formatar_medias(jogos, isTreino, num_jogos_passados_media=10):
    # Cria uma cópia real dos dados para evitar bugs de memória
    jogos_nova = copy.deepcopy(jogos)
    
    for jogo in jogos_nova:
        equipe_casa = jogo['equipe_casa']
        equipe_visitante = jogo['equipe_visitante']
        data_jogo = jogo['data']
        temporada_atual = jogo['ano'] 
        
        if isTreino:
            media_casa = get_media_estatisticas_time_treino(equipe_casa, data_jogo, temporada_atual, num_jogos_passados_media)
            media_visitante = get_media_estatisticas_time_treino(equipe_visitante, data_jogo, temporada_atual, num_jogos_passados_media)
        else:
            media_casa = get_media_estatisticas_time_teste(equipe_casa, data_jogo, temporada_atual, num_jogos_passados_media)
            media_visitante = get_media_estatisticas_time_teste(equipe_visitante, data_jogo, temporada_atual, num_jogos_passados_media)
        
        jogo['estatisticas_casa'] = media_casa
        jogo['estatisticas_visitantes'] = media_visitante
        
    return jogos_nova

# Exemplo de uso
if __name__ == "__main__":
    temporada = '2008-2009'
    # jogos = get_jogos_temporada(temporada)
    # jogos = get_jogos_equipe("Minas")
 
    # for jogo in jogos:
    #     print("\n", jogo)