import os
import pandas as pd
from dados import get_jogos_temporada, formatar_medias

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

def extrair_experimento_4_nba(temporadas, base_path, num_jogos_passados=15):
    """
    Gera as bases do Experimento 04 para a NBA (Inter-temporadas Expansivo).
    Treino: Todas as temporadas anteriores acumuladas.
    Teste: A temporada atual.
    Estrutura: data/nba_experimento_04_minuto/<num_jogos_passados>/janela_<numero>
    """
    for i in range(1, len(temporadas)):
        temps_passadas = temporadas[:i]
        temp_teste = temporadas[i]
        
        nome_janela = f"janela_{i:02d}"
        
        print(f"-> Extraindo {nome_janela}: Treinando com {len(temps_passadas)} temporada(s) | Testando em {temp_teste}")
        
        jogos_treino_acumulado = []
        for temp_treino in temps_passadas:
            jogos_treino_raw = get_jogos_temporada(temp_treino)
            jogos_treino_fmt = formatar_medias(jogos_treino_raw, True, num_jogos_passados)
            jogos_treino_acumulado.extend(descompactar_estatisticas(jogos_treino_fmt))
            
        jogos_teste_raw = get_jogos_temporada(temp_teste)
        jogos_teste_fmt = formatar_medias(jogos_teste_raw, False, num_jogos_passados)
        jogos_teste = descompactar_estatisticas(jogos_teste_fmt)
        
        # MUDANÇA: Pasta nba_experimento_04_minuto
        dir_saida = os.path.join(base_path, 'data', 'nba_experimento_04_minuto', str(num_jogos_passados), nome_janela)
        
        save_to_csv(jogos_treino_acumulado, os.path.join(dir_saida, 'treino.csv'))
        save_to_csv(jogos_teste, os.path.join(dir_saida, 'teste.csv'))
        
    print("\n>> Extração do Experimento 4 (NBA Inter-temporadas) finalizada com sucesso!")

if __name__ == "__main__":
    # Temporadas Modernas da NBA
    temporadas_nba = [
        '2008-09', '2009-10', '2010-11', '2011-12',
        '2012-13', '2013-14', '2014-15', '2015-16',
        '2016-17', '2017-18', '2018-19', '2019-20',
        '2020-21', '2021-22', '2022-23', '2023-24'
    ]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    extrair_experimento_4_nba(temporadas_nba, base_path, num_jogos_passados=15)