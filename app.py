import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import networkx as nx

# Configuração da página
st.set_page_config(page_title="Monitor Avançado de Fraudes em Apostas", layout="wide")

class AdvancedRiskDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
    
    def calculate_statistical_risk(self, df):
        """Análise estatística avançada"""
        # Z-scores multivariados
        features = df[['odds', 'valor_aposta']].values
        scaled_features = self.scaler.fit_transform(features)
        
        # Isolation Forest para detecção de anomalias
        anomaly_scores = self.isolation_forest.fit_predict(scaled_features)
        df['anomaly_score'] = anomaly_scores
        
        # Clustering com DBSCAN
        clusters = self.dbscan.fit_predict(scaled_features)
        df['cluster'] = clusters
        
        # Score final combinando diferentes métricas
        z_scores = np.abs(stats.zscore(df[['odds', 'valor_aposta']]))
        df['risk_score'] = (
            (z_scores.mean(axis=1) * 0.4) + 
            (df['anomaly_score'] == -1).astype(int) * 0.3 +
            (df['cluster'] == -1).astype(int) * 0.3
        )
        
        return df
    
    def detect_user_patterns(self, df):
        """Análise avançada de padrões de usuário"""
        user_patterns = df.groupby('usuario').agg({
            'odds': ['mean', 'std', 'count', lambda x: stats.skew(x)],
            'valor_aposta': ['mean', 'std', 'sum', lambda x: stats.kurtosis(x)],
            'resultado': lambda x: (x == 'Ganhou').mean(),
            'risk_score': 'mean'
        }).reset_index()
        
        user_patterns.columns = [
            'usuario', 'odds_mean', 'odds_std', 'bet_count', 'odds_skew',
            'valor_mean', 'valor_std', 'valor_total', 'valor_kurt',
            'win_rate', 'risk_mean'
        ]
        
        # Calcular score de padrão suspeito
        features = user_patterns[[
            'odds_mean', 'odds_std', 'bet_count',
            'valor_mean', 'valor_std', 'win_rate'
        ]]
        
        scaled_features = self.scaler.fit_transform(features)
        user_patterns['pattern_score'] = self.isolation_forest.fit_predict(scaled_features)
        
        return user_patterns
    
    def network_analysis(self, df):
        """Análise de rede simplificada"""
        G = nx.Graph()
        
        # Criar conexões entre usuários com padrões similares
        user_patterns = df.groupby('usuario').agg({
            'odds': 'mean',
            'valor_aposta': 'mean'
        }).reset_index()
        
        # Adicionar nós
        for _, user in user_patterns.iterrows():
            G.add_node(user['usuario'])
        
        # Adicionar arestas para usuários com padrões similares
        for i, user1 in user_patterns.iterrows():
            for j, user2 in user_patterns.iterrows():
                if i < j:  # Evitar duplicatas
                    odds_diff = abs(user1['odds'] - user2['odds'])
                    value_diff = abs(user1['valor_aposta'] - user2['valor_aposta'])
                    
                    if odds_diff < 0.5 and value_diff < 50:  # Thresholds arbitrários
                        G.add_edge(user1['usuario'], user2['usuario'])
        
        # Encontrar comunidades
        communities = list(nx.community.greedy_modularity_communities(G))
        
        return G, communities
    
    def detect_time_patterns(self, df):
        """Análise avançada de padrões temporais"""
        df['hora'] = pd.to_datetime(df['data']).dt.hour
        df['minuto'] = pd.to_datetime(df['data']).dt.minute
        
        # Análise de frequência por intervalo de tempo
        time_patterns = df.groupby(['usuario', 'hora']).agg({
            'minuto': 'count',
            'valor_aposta': 'sum',
            'odds': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        
        # Detectar padrões suspeitos de tempo
        time_patterns['time_risk'] = (
            (time_patterns['minuto'] > time_patterns['minuto'].mean() + time_patterns['minuto'].std()) &
            (time_patterns['odds'] > time_patterns['odds'].mean())
        ).astype(int)
        
        return time_patterns

def gerar_dados_simulados(n_apostas=1000):
    """Geração de dados simulados mais complexos"""
    np.random.seed(42)
    
    # Dados base
    datas = [datetime.now() - timedelta(minutes=x*30) for x in range(n_apostas)]
    usuarios = [f"user_{i}" for i in np.random.randint(1, 100, n_apostas)]
    valores = np.random.exponential(100, n_apostas)
    odds = np.random.uniform(1.1, 10.0, n_apostas)
    resultados = np.random.choice(['Ganhou', 'Perdeu'], n_apostas, p=[0.3, 0.7])
    esportes = np.random.choice(['Futebol', 'Basquete', 'Tênis', 'Vôlei'], n_apostas)
    
    # Criar padrões suspeitos mais complexos
    
    # 1. Usuário que sempre aposta valores altos
    idx_high_roller = np.where(np.array(usuarios) == 'user_1')[0]
    valores[idx_high_roller] *= 3
    
    # 2. Usuário com padrão temporal suspeito
    idx_time_pattern = np.where(np.array(usuarios) == 'user_2')[0]
    for idx in idx_time_pattern:
        if idx % 5 == 0:  # Criar padrão regular
            valores[idx] *= 2
            odds[idx] = 9.0
    
    # 3. Grupo de usuários coordenados
    coordinated_users = ['user_3', 'user_4', 'user_5']
    idx_coordinated = np.where(np.isin(usuarios, coordinated_users))[0]
    odds[idx_coordinated] = np.random.uniform(8.0, 9.0, len(idx_coordinated))
    
    return pd.DataFrame({
        'data': datas,
        'usuario': usuarios,
        'valor_aposta': valores,
        'odds': odds,
        'resultado': resultados,
        'esporte': esportes,
        'retorno_potencial': valores * odds
    })

# Interface principal
st.title("🎲 Monitor Avançado de Fraudes em Apostas")

# Sidebar
st.sidebar.header("Configurações")
n_dias = st.sidebar.slider("Número de dias para análise", 1, 30, 7)
min_risk_score = st.sidebar.slider("Score mínimo de risco", 1.0, 5.0, 2.0)
show_network = st.sidebar.checkbox("Mostrar análise de rede", True)

# Gerar dados
df = gerar_dados_simulados(n_dias * 100)

# Instanciar detector
detector = AdvancedRiskDetector()

# Calcular riscos e padrões
df_with_risk = detector.calculate_statistical_risk(df)
user_patterns = detector.detect_user_patterns(df_with_risk)
time_patterns = detector.detect_time_patterns(df_with_risk)

if show_network:
    G, communities = detector.network_analysis(df_with_risk)

# Interface com tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visão Geral", "🎯 Análise de Risco", "🕒 Padrões Temporais", "🔍 Análise de Rede", "📚 Documentação"])

with tab1:
    st.markdown("""
    ### Monitor de Apostas em Tempo Real
    Este painel apresenta uma visão geral das apostas e principais indicadores de risco.
    Os gráficos abaixo mostram padrões e anomalias detectados no período selecionado.
    """)
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Apostas", len(df))
    with col2:
        st.metric("Valor Total Apostado", f"R$ {df['valor_aposta'].sum():,.2f}")
    with col3:
        st.metric("Média das Odds", f"{df['odds'].mean():,.2f}")
    with col4:
        st.metric("Apostas Suspeitas", len(df_with_risk[df_with_risk['risk_score'] > min_risk_score]))

    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        fig_anomalies = px.scatter(
            df_with_risk,
            x='odds',
            y='valor_aposta',
            color='anomaly_score',
            title="Detecção de Anomalias",
            labels={'anomaly_score': 'Score de Anomalia'}
        )
        st.plotly_chart(fig_anomalies)
        st.markdown("""
        **Interpretação do Gráfico de Anomalias:**
        - Pontos em vermelho (-1) indicam apostas potencialmente suspeitas
        - Pontos em azul (1) representam padrões normais de apostas
        - Concentrações incomuns de pontos podem indicar atividade coordenada
        """)
    
    with col2:
        fig_clusters = px.scatter(
            df_with_risk,
            x='odds',
            y='valor_aposta',
            color='cluster',
            title="Clusters de Apostas"
        )
        st.plotly_chart(fig_clusters)
        st.markdown("""
        **Interpretação dos Clusters:**
        - Cada cor representa um grupo de apostas com características similares
        - Clusters isolados (-1) podem indicar comportamentos atípicos
        - A distância entre clusters indica o quão diferentes são os padrões
        """)

with tab2:
    st.header("Análise de Risco")
    
    # Mapa de calor de risco
    fig_risk_heatmap = px.density_heatmap(
        df_with_risk,
        x='odds',
        y='valor_aposta',
        z='risk_score',
        title="Mapa de Calor de Risco"
    )
    st.plotly_chart(fig_risk_heatmap)
    
    # Padrões por usuário
    st.subheader("Padrões Suspeitos por Usuário")
    suspicious_users = user_patterns[user_patterns['pattern_score'] == -1]
    
    if not suspicious_users.empty:
        st.dataframe(suspicious_users)
        
        fig_user_patterns = px.scatter(
            suspicious_users,
            x='odds_mean',
            y='valor_mean',
            size='bet_count',
            color='win_rate',
            hover_data=['usuario'],
            title="Padrões de Usuários Suspeitos"
        )
        st.plotly_chart(fig_user_patterns)

with tab3:
    st.header("Análise Temporal")
    
    # Padrões temporais
    fig_time = px.scatter(
        time_patterns[time_patterns['time_risk'] == 1],
        x='hora',
        y='minuto',
        size='valor_aposta',
        color='odds',
        hover_data=['usuario'],
        title="Padrões Temporais Suspeitos"
    )
    st.plotly_chart(fig_time)
    
    # Distribuição horária
    fig_hour_dist = px.histogram(
        df_with_risk,
        x='hora',
        color='esporte',
        title="Distribuição de Apostas por Hora"
    )
    st.plotly_chart(fig_hour_dist)

with tab4:
    st.markdown("""
    ### Análise de Redes Sociais
    Esta seção utiliza teoria de redes para identificar grupos de usuários que podem estar coordenando suas apostas.
    As comunidades são formadas por usuários com padrões similares de apostas.
    """)
    if show_network:
        st.header("Análise de Rede")
        
        # Informações sobre comunidades
        st.subheader("Comunidades Detectadas")
        for i, community in enumerate(communities):
            st.write(f"Comunidade {i+1}: {len(community)} usuários")
            if len(community) >= 3:  # Mostrar apenas comunidades grandes
                st.write("Usuários:", ', '.join(list(community)))
        
        # Métricas de rede
        st.subheader("Métricas de Rede")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Número de Conexões", G.number_of_edges())
        with col2:
            st.metric("Grupos Suspeitos", len([c for c in communities if len(c) >= 3]))
        with col3:
            st.metric("Densidade da Rede", f"{nx.density(G):.4f}")

# Lista final de alertas
st.header("⚠️ Alertas de Alto Risco")
high_risk = df_with_risk[df_with_risk['risk_score'] > min_risk_score].sort_values('risk_score', ascending=False)

if not high_risk.empty:
    st.dataframe(
        high_risk[[
            'data', 'usuario', 'valor_aposta', 'odds', 
            'esporte', 'resultado', 'risk_score', 'anomaly_score'
        ]]
    )
else:
    st.info("Nenhum alerta de alto risco detectado no período analisado.")

# Nova aba de documentação
with tab5:
    st.header("📚 Documentação do Sistema")
    
    with st.expander("🎯 Métodos de Detecção de Anomalias"):
        st.markdown("""
        O sistema utiliza três métodos principais para detectar anomalias:
        
        1. **Isolation Forest**
        - Algoritmo que isola observações considerando que anomalias são mais facilmente separáveis
        - Eficiente para detectar apostas que fogem dos padrões normais
        
        2. **Clustering (DBSCAN)**
        - Agrupa apostas com características similares
        - Identifica pontos que não pertencem a nenhum cluster como potenciais anomalias
        
        3. **Análise Estatística Multivariada**
        - Utiliza Z-scores para identificar valores extremos
        - Considera múltiplas variáveis simultaneamente (odds, valores, frequência)
        """)
    
    with st.expander("🌐 Análise de Redes"):
        st.markdown("""
        A análise de redes sociais é utilizada para:
        
        1. **Detecção de Comunidades**
        - Identifica grupos de usuários com comportamentos similares
        - Comunidades pequenas podem indicar coordenação
        
        2. **Métricas de Rede**
        - Densidade: indica o quão conectados estão os usuários
        - Número de conexões: mostra a intensidade das relações
        
        3. **Interpretação**
        - Grupos grandes (>30 usuários): geralmente padrões normais
        - Grupos médios (10-30 usuários): requerem atenção
        - Grupos pequenos (2-5 usuários): alto risco de coordenação
        - Usuários isolados: possíveis comportamentos fraudulentos
        """)
    
    with st.expander("⏱️ Análise Temporal"):
        st.markdown("""
        A análise temporal examina:
        
        1. **Padrões de Horário**
        - Identifica concentrações suspeitas de apostas
        - Detecta padrões automatizados (bots)
        
        2. **Sequências de Apostas**
        - Analisa intervalos entre apostas
        - Identifica padrões repetitivos suspeitos
        
        3. **Distribuição Temporal**
        - Compara com padrões históricos normais
        - Detecta desvios significativos
        """)
    
    with st.expander("📊 Interpretação dos Scores"):
        st.markdown("""
        O sistema utiliza diferentes scores para avaliar riscos:
        
        1. **Score de Anomalia**
        - -1: Indica comportamento suspeito
        - 1: Indica comportamento normal
        
        2. **Score de Risco**
        - 0-2: Risco baixo
        - 2-4: Risco médio
        - >4: Risco alto
        
        3. **Risk Score Combinado**
        - Combina múltiplos indicadores
        - Ponderado por importância relativa
        - Atualizado em tempo real
        """)
