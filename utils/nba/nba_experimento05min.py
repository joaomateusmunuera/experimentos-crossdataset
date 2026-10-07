import pandas as pd
import os
from dados import get_jogos_temporada
from dados import formatar_medias

def save_to_csv(data, filepath):
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def descompactar_estatisticas(jogos):
    dados_formatados = []

    for jogo in jogos:
        casa_estatisticas = jogo['estatisticas_casa']
        visitante_estatisticas = jogo['estatisticas_visitantes']

        dados = {
            'placar_casa': jogo['placar_casa'],
            'placar_visitante': jogo['placar_visitante'],
            'data': jogo['data'],
            'round': jogo['round'],
            'stage': jogo['stage'],
            'ano': jogo['ano'],
            'equipe_casa': jogo['equipe_casa'],
            'equipe_visitante': jogo['equipe_visitante'],

            'resultado': 1 if jogo['placar_casa'] > jogo['placar_visitante'] else 0,

            'Pts_casa': casa_estatisticas.get('Pts', 0),
            '3P_casa': casa_estatisticas.get('3P', 0),
            '2P_casa': casa_estatisticas.get('2P', 0),
            'LL_casa': casa_estatisticas.get('LL', 0),
            'RT_casa': casa_estatisticas.get('RT', 0),
            'RO_casa': casa_estatisticas.get('RO', 0),
            'RD_casa': casa_estatisticas.get('RD', 0),
            'AS_casa': casa_estatisticas.get('AS', 0),
            'ER_casa': casa_estatisticas.get('ER', 0),
            'IA%_casa': casa_estatisticas.get('IA%', 0),
            '3PC_casa': casa_estatisticas.get('3PC', 0),
            '3PT_casa': casa_estatisticas.get('3PT', 0),
            '3P%_casa': casa_estatisticas.get('3P%', 0),
            '2PC_casa': casa_estatisticas.get('2PC', 0),
            '2PT_casa': casa_estatisticas.get('2PT', 0),
            '2P%_casa': casa_estatisticas.get('2P%', 0),
            'LLC_casa': casa_estatisticas.get('LLC', 0),
            'LLT_casa': casa_estatisticas.get('LLT', 0),
            'LL%_casa': casa_estatisticas.get('LL%', 0),
            'EN_casa': casa_estatisticas.get('EN', 0),
            'BR_casa': casa_estatisticas.get('BR', 0),
            'B/E_casa': casa_estatisticas.get('B/E', 0),
            'TO_casa': casa_estatisticas.get('TO', 0),
            'FC_casa': casa_estatisticas.get('FC', 0),
            'T/FC_casa': casa_estatisticas.get('T/FC', 0),
            'ET_casa': casa_estatisticas.get('ET', 0),
            'VI_casa': casa_estatisticas.get('VI', 0),
            'EF_casa': casa_estatisticas.get('EF', 0),

            'Pts_visitante': visitante_estatisticas.get('Pts', 0),
            '3P_visitante': visitante_estatisticas.get('3P', 0),
            '2P_visitante': visitante_estatisticas.get('2P', 0),
            'LL_visitante': visitante_estatisticas.get('LL', 0),
            'RT_visitante': visitante_estatisticas.get('RT', 0),
            'RO_visitante': visitante_estatisticas.get('RO', 0),
            'RD_visitante': visitante_estatisticas.get('RD', 0),
            'AS_visitante': visitante_estatisticas.get('AS', 0),
            'ER_visitante': visitante_estatisticas.get('ER', 0),
            'IA%_visitante': visitante_estatisticas.get('IA%', 0),
            '3PC_visitante': visitante_estatisticas.get('3PC', 0),
            '3PT_visitante': visitante_estatisticas.get('3PT', 0),
            '3P%_visitante': visitante_estatisticas.get('3P%', 0),
            '2PC_visitante': visitante_estatisticas.get('2PC', 0),
            '2PT_visitante': visitante_estatisticas.get('2PT', 0),
            '2P%_visitante': visitante_estatisticas.get('2P%', 0),
            'LLC_visitante': visitante_estatisticas.get('LLC', 0),
            'LLT_visitante': visitante_estatisticas.get('LLT', 0),
            'LL%_visitante': visitante_estatisticas.get('LL%', 0),
            'EN_visitante': visitante_estatisticas.get('EN', 0),
            'BR_visitante': visitante_estatisticas.get('BR', 0),
            'B/E_visitante': visitante_estatisticas.get('B/E', 0),
            'TO_visitante': visitante_estatisticas.get('TO', 0),
            'FC_visitante': visitante_estatisticas.get('FC', 0),
            'T/FC_visitante': visitante_estatisticas.get('T/FC', 0),
            'ET_visitante': visitante_estatisticas.get('ET', 0),
            'VI_visitante': visitante_estatisticas.get('VI', 0),
            'EF_visitante': visitante_estatisticas.get('EF', 0)
        }
        dados_formatados.append(dados)
    return dados_formatados


def gerar_arquivos_inter_incremental(temporadas_passadas, temporada_atual, qtd_jogos_base, base_path):
    """
    Gera o treino composto por: (TODAS as temporadas passadas) + (jogos da temporada atual até o índice)
    E o teste composto por: (apenas o jogo do índice atual)
    """
    jogos_treino_passados_formatados = []

    # 1. Carrega e formata todo o histórico de temporadas passadas da NBA
    for temp in temporadas_passadas:
        jogos_temp = get_jogos_temporada(temp)
        jogos_temp_fmt = formatar_medias(jogos_temp, True, 15)
        jogos_treino_passados_formatados.extend(descompactar_estatisticas(jogos_temp_fmt))

    # 2. Carrega a temporada atual de teste
    jogos_atual_treino = get_jogos_temporada(temporada_atual)
    jogos_atual_teste = get_jogos_temporada(temporada_atual)

    jogos_atual_treino_fmt = formatar_medias(jogos_atual_treino, True, 15)
    jogos_atual_teste_fmt = formatar_medias(jogos_atual_teste, False, 15)

    indice = qtd_jogos_base
    num_arquivo = 1

    # 3. Gera os arquivos incrementais para a temporada atual
    while indice < len(jogos_atual_treino_fmt):
        # Treino = Histórico das temporadas passadas + Jogos da temporada atual até o 'indice'
        treino_atual_parcial = descompactar_estatisticas(jogos_atual_treino_fmt[0:indice])
        treino_completo = jogos_treino_passados_formatados + treino_atual_parcial

        # Teste = Apenas o jogo de teste do 'indice'
        teste = descompactar_estatisticas([jogos_atual_teste_fmt[indice]])

        # MUDANÇA AQUI: Alterada a pasta de salvamento para nba_experimento_05_minuto
        final_path = os.path.join(base_path, 'data', 'nba_experimento_05_minuto', temporada_atual, f'{indice}-1')

        save_to_csv(treino_completo, f'{final_path}/treino_{num_arquivo}.csv')
        save_to_csv(teste, f'{final_path}/teste_{num_arquivo}.csv')

     #   K_salto = 10
        indice += 1
        num_arquivo += 1

    print(f"-> Concluído: {temporada_atual} | Treino base com {len(temporadas_passadas)} temporadas passadas.")


if __name__ == "__main__":
    # MUDANÇA AQUI: Temporadas modernas da NBA
    temporadas_nba = [
        '2008-09', '2009-10', '2010-11', '2011-12',
        '2012-13', '2013-14', '2014-15', '2015-16',
        '2016-17', '2017-18', '2018-19', '2019-20',
        '2020-21', '2021-22', '2022-23', '2023-24','2024-25'
    ]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    qtd_jogos_base = 15

    print(">> Iniciando a extração do Experimento 05 (NBA Combinatório por Minuto)...")
    
    # A partir da 2ª temporada (índice 1), a temporada atual ganha o histórico de todas as anteriores
    for i in range(16,17):
        temps_passadas = temporadas_nba[:i]
        temp_atual = temporadas_nba[i]

        gerar_arquivos_inter_incremental(temps_passadas, temp_atual, qtd_jogos_base, base_path)
        
    print("\n>> Extração por minuto do Experimento 5 (NBA) finalizada com sucesso!")