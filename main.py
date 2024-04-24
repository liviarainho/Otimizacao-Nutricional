import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pulp import LpVariable, LpProblem, lpSum, LpMinimize, LpStatus

# Função para carregar dados dos arquivos
def carregar_dados():
    df_alimentos = pd.read_excel("Informações Nutricionais - P1 O&S.xlsx")
    df_necessidades = pd.read_excel("Quantidade Necessária.xlsx")
    return df_alimentos, df_necessidades

# Função para preparar dados de alimentos
def preparar_dados_alimentos(df_alimentos):
    colunas_transformar = ['Energia (kcal)', 'Proteina (g)', 'Lipideos (g)', 'Carboidrato (g)',
                           'Calcio (mg)', 'Ferro (mg)', 'Vitamina A (mg)', 'Vitamina C (mg)']
    
    for coluna in colunas_transformar:
        df_alimentos[coluna] = pd.to_numeric(df_alimentos[coluna], errors='coerce')
        df_alimentos[coluna] = df_alimentos[coluna].fillna(0)
        df_alimentos[coluna] = df_alimentos[coluna].astype(int)
    
    return df_alimentos

# Função para separar índices dos alimentos por período
def separar_indices_alimentos(df_alimentos):
    cafe, lanche, almoco, jantar = [], [], [], []
    
    for i, row in df_alimentos.iterrows():
        periodo = row["Período"]
        if periodo == "Café da Manhã":
            cafe.append(i)
        elif periodo == "Lanche":
            lanche.append(i)
        elif periodo == "Almoço":
            almoco.append(i)
        elif periodo == "Jantar":
            jantar.append(i)
            
    return cafe, lanche, almoco, jantar

# Função para treinar modelos de regressão linear para previsões
def treinar_modelos(df_necessidades, sexo):
    dados_filtrados = df_necessidades[df_necessidades["Sexo"] == sexo]
    entrada = ["Peso (kg)"]
    saidas = ['Proteina (g)', 'Carboidrato (g)', 'Lipideos (g)', 'Vitamina A (mg)', 'Vitamina C (mg)', 'Calcio (mg)', 'Ferro (mg)']
    
    models = {}
    for saida in saidas:
        model = LinearRegression()
        model.fit(dados_filtrados[entrada], dados_filtrados[saida])
        models[saida] = model
    
    return models

# Função para fazer previsões com base nos modelos treinados
def fazer_previsoes(models, peso):
    entrada = ["Peso (kg)"]
    predictions = {}
    for saida, model in models.items():
        prediction = model.predict(pd.DataFrame(data=[[peso]], columns=entrada))
        predictions[saida] = prediction[0]
        
    return predictions

# Função para otimizar a dieta com base nas restrições
def otimizar_dieta(df_alimentos, cafe, lanche, almoco, jantar, nutriente_necessario):
    alimentos_selecionados = []
    
    # Criar problema de otimização
    prob = LpProblem("Dieta", LpMinimize)
    quantidade = LpVariable.dicts("Quantidade", df_alimentos.index, lowBound=0, upBound=1, cat='Integer')
    
    # Minimizar calorias totais
    prob += lpSum([df_alimentos.loc[i, "Energia (kcal)"] * quantidade[i] for i in df_alimentos.index])
    
    # Restrições nutricionais
    for nutriente, quant in nutriente_necessario.items():
        prob += lpSum([df_alimentos.loc[j, nutriente] * quantidade[j] for j in df_alimentos.index]) >= quant
        
    # Restrições de refeições
    prob += lpSum([quantidade[i] for i in cafe]) == 4
    prob += lpSum([quantidade[i] for i in lanche]) == 3
    prob += lpSum([quantidade[i] for i in almoco]) == 5
    prob += lpSum([quantidade[i] for i in jantar]) == 5
    
    # Restringir alimentos repetidos
    prob += lpSum([quantidade[i] for i in alimentos_selecionados]) == 0
    
    # Resolver problema de otimização
    prob.solve()
    
    # Atualizar lista de alimentos selecionados
    alimentos_selecionados.extend([i for i in df_alimentos.index if quantidade[i].varValue > 0])
    
    # Resultados da otimização
    resultados = {
        "status": LpStatus[prob.status],
        "cafe": [(df_alimentos.loc[i, 'Nome'], quantidade[i].varValue) for i in cafe if quantidade[i].varValue > 0],
        "lanche": [(df_alimentos.loc[i, 'Nome'], quantidade[i].varValue) for i in lanche if quantidade[i].varValue > 0],
        "almoco": [(df_alimentos.loc[i, 'Nome'], quantidade[i].varValue) for i in almoco if quantidade[i].varValue > 0],
        "jantar": [(df_alimentos.loc[i, 'Nome'], quantidade[i].varValue) for i in jantar if quantidade[i].varValue > 0]
    }
    
    total_calories = sum([df_alimentos.loc[i, "Energia (kcal)"] * quantidade[i].varValue for i in df_alimentos.index if quantidade[i].varValue > 0])
    resultados["total_calories"] = total_calories
    
    return resultados, alimentos_selecionados

# Função principal para a aplicação Streamlit
def main():
    st.title("Otimização de Dieta")
    
    # Carregar dados
    df_alimentos, df_necessidades = carregar_dados()
    
    # Preparar dados de alimentos
    df_alimentos = preparar_dados_alimentos(df_alimentos)
    
    # Separar índices de alimentos por período
    cafe, lanche, almoco, jantar = separar_indices_alimentos(df_alimentos)
    
    # Selecionar sexo e peso para previsões
    sexo = st.selectbox("Sexo", ["Mulher", "Homem"])
    peso = st.slider("Peso (kg)", 40, 150, 62)
    
    # Treinar modelos de regressão linear
    models = treinar_modelos(df_necessidades, sexo)
    
    # Fazer previsões
    predictions = fazer_previsoes(models, peso)
    
    # Exibir previsões
    st.subheader("Previsões de Nutrientes")
    for saida, value in predictions.items():
        st.write(f"{saida}: {value:.2f}")
    
    # Configurar restrições nutricionais
    nutriente_necessario = {
        "Proteina (g)": predictions["Proteina (g)"],
        "Carboidrato (g)": predictions["Carboidrato (g)"],
        "Lipideos (g)": predictions["Lipideos (g)"],
        "Vitamina A (mg)": predictions["Vitamina A (mg)"],
        "Vitamina C (mg)": predictions["Vitamina C (mg)"],
        "Calcio (mg)": predictions["Calcio (mg)"],
        "Ferro (mg)": predictions["Ferro (mg)"]
    }
    
    # Otimizar a dieta
    resultados, alimentos_selecionados = otimizar_dieta(df_alimentos, cafe, lanche, almoco, jantar, nutriente_necessario)
    
    # Exibir resultados da otimização
    st.subheader("Resultados da Otimização")
    if resultados["status"] == "Optimal":
        st.write("Solução Ótima Encontrada:")
        
        st.write("\nCafé da Manhã:")
        for nome, quantidade in resultados["cafe"]:
            st.write(f"{nome}: {quantidade}")
        
        st.write("\nLanche:")
        for nome, quantidade in resultados["lanche"]:
            st.write(f"{nome}: {quantidade}")
        
        st.write("\nAlmoço:")
        for nome, quantidade in resultados["almoco"]:
            st.write(f"{nome}: {quantidade}")
        
        st.write("\nJantar:")
        for nome, quantidade in resultados["jantar"]:
            st.write(f"{nome}: {quantidade}")
        
        st.write(f"\nTotal de Calorias: {resultados['total_calories']:.2f}")
    else:
        st.write("Não foi possível encontrar uma solução viável.")

if __name__ == "__main__":
    main()

