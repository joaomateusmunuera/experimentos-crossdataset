import os
import pandas as pd
from dados import get_jogos_temporada, formatar_medias 

def save_to_csv(data, filepath):
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def formatar_ano_nbb(ano_nba_curto):
    """Converte '2008-09' (Formato NBA) para '2008-2009' (Formato NBB)."""
    partes = ano_nba_curto.split('-')
    ano_inicio = partes[0]
    # Reconstrói o século XX/XXI para o segundo ano (ex: '09' -> '2009')
    seculo = ano_inicio[:2]
    ano_fim = seculo + partes[1]
    return f"{ano_inicio}-{ano_fim}"

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

def verificar_compatibilidade(num_jogos_passados=15):
    print("\n[!] A verificar compatibilidade entre NBA e NBB...")
    
    # Testa os formatos invertidos
    jogos_nba = get_jogos_temporada('2009-10', banco='nba_db')[:5]
    jogos_nbb = get_jogos_temporada('2009-2010', banco='db')[:5]
    
    print(f" -> Jogos achados na NBA ('nba_db'): {len(jogos_nba)}")
    print(f" -> Jogos achados no NBB ('db'): {len(jogos_nbb)}")
    
    if not jogos_nba or not jogos_nbb:
        print("[X] Erro: Dados insuficientes para testar compatibilidade.")
        return False
        
    nba_fmt = descompactar_estatisticas(formatar_medias(jogos_nba, True, banco='nba_db', num_jogos_passados_media=num_jogos_passados))
    nbb_fmt = descompactar_estatisticas(formatar_medias(jogos_nbb, False, banco='db', num_jogos_passados_media=num_jogos_passados))
    
    colunas_nba = list(nba_fmt[0].keys())
    colunas_nbb = list(nbb_fmt[0].keys())
    
    if colunas_nba == colunas_nbb:
        print("[OK] Datasets 100% compatíveis! Estrutura de colunas alinhada.")
        return True
    else:
        print("[X] Incompatibilidade detectada nas colunas!")
        return False

def extrair_experimento_cross_dataset(temporadas, base_path, num_jogos_passados=15):
    for i in range(1, len(temporadas)):
        temps_passadas = temporadas[:i]
        temp_teste = temporadas[i]
        
        nome_janela = f"janela_{i:02d}"
        print(f"\n-> Extraindo {nome_janela}: Testando no NBB ({temp_teste}) | Treinando com NBA ({temps_passadas})")
        
        # 1. MONTAR O TREINO (NBA) - Usa o formato curto padrão da lista
        jogos_treino_acumulado = []
        for temp_treino in temps_passadas:
            jogos_treino_raw = get_jogos_temporada(temp_treino, banco='nba_db')
            
            if jogos_treino_raw:
                jogos_treino_fmt = formatar_medias(jogos_treino_raw, isTreino=True, banco='nba_db', num_jogos_passados_media=num_jogos_passados)
                jogos_treino_acumulado.extend(descompactar_estatisticas(jogos_treino_fmt))
            
        # 2. MONTAR O TESTE (NBB) 
        # Converte o ano da NBA para o formato longo do NBB
        temp_teste_nbb = formatar_ano_nbb(temp_teste)
        jogos_teste_raw = get_jogos_temporada(temp_teste_nbb, banco='db')
        
        if jogos_teste_raw:
            jogos_teste_fmt = formatar_medias(jogos_teste_raw, isTreino=False, banco='db', num_jogos_passados_media=num_jogos_passados)
            jogos_teste = descompactar_estatisticas(jogos_teste_fmt)
        else:
            jogos_teste = []
            print(f"   [Aviso] Temporada {temp_teste_nbb} não retornou dados para teste no NBB.")
        
        # 3. SALVAMENTO (Pasta específica para NBA -> NBB)
        dir_saida = os.path.join(base_path, 'data', 'cross_dataset_nba_para_nbb', str(num_jogos_passados), nome_janela)
        
        if jogos_treino_acumulado and jogos_teste:
            save_to_csv(jogos_treino_acumulado, os.path.join(dir_saida, 'treino_nba.csv'))
            save_to_csv(jogos_teste, os.path.join(dir_saida, 'teste_nbb.csv'))
            print(f"   [Salvo] Treino: {len(jogos_treino_acumulado)} jogos | Teste: {len(jogos_teste)} jogos")
        else:
            print(f"   [Erro] Janela {nome_janela} falhou ao salvar. Faltou Treino ou Teste.")
            
    print("\n>> Extração do Experimento Cross-Dataset (NBA -> NBB) finalizada com sucesso!")

if __name__ == "__main__":
    # Base estrutural padronizada com o formato curto da NBA
    temporadas_cronologicas = [
        '2008-09', '2009-10', '2011-12', '2012-13', 
        '2013-14', '2014-15', '2015-16', '2016-17', 
        '2018-19', '2019-20', '2020-21', '2021-22', 
        '2022-23', '2023-24', '2024-25'
    ]
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    
    if verificar_compatibilidade(num_jogos_passados=15):
        extrair_experimento_cross_dataset(temporadas_cronologicas, base_path, num_jogos_passados=15)