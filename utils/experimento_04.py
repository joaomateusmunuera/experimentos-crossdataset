import os
import pandas as pd
from dados import get_jogos_temporada
from dados import formatar_medias

os.environ['PROJECT_PATH'] = os.getcwd()

def save_to_csv(data, filename):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

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

            # (1 = vitoria do time da casa, 0 = derrota)    
            'resultado': 1 if jogo['placar_casa'] > jogo['placar_visitante'] else 0,
            
            # Estatísticas da equipe da casa
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
            
            # Estatísticas da equipe visitante
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

def process_expanding_window(idx_janela, temporadas_treino, temporadas_teste, num_jogos_passados=15):
    project_path = os.environ['PROJECT_PATH']
    
    path_treino = os.path.join(project_path, f'data/experimento_04/{num_jogos_passados}/janela_{idx_janela:02d}/treino.csv')
    path_teste = os.path.join(project_path, f'data/experimento_04/{num_jogos_passados}/janela_{idx_janela:02d}/teste.csv')

    todos_jogos_treino_formatados = []
    todos_jogos_teste_formatados = []

    # 1. Coletar e processar todas as temporadas do bloco de TREINO
    for temp in temporadas_treino:
        jogos = get_jogos_temporada(temp)
        jogos_formatados = formatar_medias(jogos, True, num_jogos_passados)
        todos_jogos_treino_formatados.extend(descompactar_estatisticas(jogos_formatados))

    # 2. Coletar e processar a única temporada do bloco de TESTE
    for temp in temporadas_teste:
        jogos = get_jogos_temporada(temp)
        jogos_formatados = formatar_medias(jogos, False, num_jogos_passados)
        todos_jogos_teste_formatados.extend(descompactar_estatisticas(jogos_formatados))

    save_to_csv(todos_jogos_treino_formatados, path_treino)
    save_to_csv(todos_jogos_teste_formatados, path_teste)
    print(f"-> Janela {idx_janela:02d} criada. Treino: {temporadas_treino} | Teste: {temporadas_teste}")


if __name__ == "__main__":
    # Lista de temporadas ordenadas cronologicamente
    temporadas = [
        "2008-2009", "2009-2010", "2011-2012", "2012-2013",
        "2013-2014", "2014-2015", "2015-2016", "2016-2017", 
        "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"
    ]
    
    num_janelas = len(temporadas) - 1  # O limite garante que a última janela teste a última temporada
    
    for i in range(1, num_janelas + 1):
        # Treino: Pega de 0 até o índice atual i (Ex: na janela 1, pega a temporada de índice 0)
        temps_treino = temporadas[:i]
        
        # Teste: Transforma em lista contendo APENAS a temporada imediatamente seguinte [i]
        # Usando colchetes criamos uma lista unitária, mantendo a compatibilidade com o laço "for temp in temporadas_teste"
        temps_teste = [temporadas[i]]
        
        print(f"\nIniciando Processamento da Janela {i:02d}...")
        process_expanding_window(
            idx_janela=i, 
            temporadas_treino=temps_treino, 
            temporadas_teste=temps_teste,
            num_jogos_passados=15
        )