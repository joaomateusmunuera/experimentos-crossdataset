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


def gerar_arquivos_acumulada_intratemporada(temporadas, base_path, qtd_jogos_base=15):
    """
    Gera as bases do Experimento 03 para a NBA (Acurácia Acumulada - Intratemporada).
    Treino: Inicia com os primeiros N jogos da temporada e cresce +1 a cada iteração.
    Teste: O jogo exato logo após a janela de treino atual.
    """
    for temporada_atual in temporadas:
        
        # 1. Carrega a temporada atual para gerar treino e teste
        jogos_raw = get_jogos_temporada(temporada_atual)
        
        # Ignora temporadas vazias (caso a string da lista não exista no banco)
        if not jogos_raw:
            continue
            
        jogos_treino_fmt = formatar_medias(jogos_raw, True, qtd_jogos_base)
        jogos_teste_fmt = formatar_medias(jogos_raw, False, qtd_jogos_base)

        indice = qtd_jogos_base
        num_arquivo = 1

        print(f"-> Extraindo NBA {temporada_atual} | Total de jogos possíveis: {len(jogos_treino_fmt)}")

        # 2. Loop de janela expansiva (Walk-Forward) dentro da temporada
        while indice < len(jogos_treino_fmt):
            # Treino = Expande a cada iteração
            treino_atual = descompactar_estatisticas(jogos_treino_fmt[0:indice])
            
            # Teste = Apenas o jogo alvo
            teste_atual = descompactar_estatisticas([jogos_teste_fmt[indice]])

            # Pasta de saída exclusiva para a NBA
            final_path = os.path.join(base_path, 'data', 'nba_experimento_03_minuto', temporada_atual, f'{indice}-1')

            # Salva os arquivos iterativos
            save_to_csv(treino_atual, os.path.join(final_path, f'treino_{num_arquivo}.csv'))
            save_to_csv(teste_atual, os.path.join(final_path, f'teste_{num_arquivo}.csv'))

            indice += 1
            num_arquivo += 1

    print("\n>> Extração por minuto da NBA (Experimento 3) finalizada com sucesso!")


if __name__ == "__main__":
    # Ajuste as temporadas conforme os dados que você subiu no nba_db
    temporadas_nba = [
        '2008-09', '2009-10', '2010-11', '2011-12',
        '2012-13', '2013-14', '2014-15', '2015-16',
        '2016-17', '2017-18', '2018-19', '2019-20',
        '2020-21', '2021-22', '2022-23', '2023-24'
    ]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # K=15 (se for outro valor para a média dos jogos recentes do time, basta alterar aqui)
    qtd_jogos_base = 15

    gerar_arquivos_acumulada_intratemporada(temporadas_nba, base_path, qtd_jogos_base)