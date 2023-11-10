# Imports para manipulação de dados e cálculos numéricos
import numpy as np
import pandas as pd

# Imports relacionados a ajustes de data e hora
import datetime

# Imports para manipulação de sistema operacional
import os

# Imports para pré-processamento e transformação de dados
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# Imports para codificação de variáveis categóricas
from category_encoders import TargetEncoder

# Imports relacionados a algoritmos de aprendizado de máquina
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Imports para construção de pipelines e transformações de colunas
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imports relacionados a métricas de avaliação de modelos
from sklearn.metrics import accuracy_score

class Features:
    def numeric(self):
       return ['Código Terceiro', 'numero_dia_acordada', 'Terceiro Centralizador', 'faixa_de_peso']
    
    def categorical(self):
       return ['LINHA', 'Cidade', 'UF']
    
    def target(self):
        return ['Data PCP', 'numero_dia_pcp']
    
    def decision(self):
        return ['Situação da Entrega']

    def reference(self):
        return ['Data de Embarque']
    
    def avulsos(self):
        return ['Data Acordada', 'Peso Líquido Estimado']
    def id(self):
        return ['Pedido', 'pedido_compartilhado']
    
class Hiperparametros:
    def get_hiperparametros(self):
        return {
            'Random Forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__splitter': ['best', 'random'],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['linear', 'rbf', 'poly'],
                    'classifier__gamma': ['scale', 'auto', 0.1, 1],
                    'classifier__degree': [2, 3, 4]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'classifier__n_neighbors': [3, 5, 7, 9],
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'classifier__p': [1, 2]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(),
                'params': {
                    'classifier__learning_rate': [0.1, 0.2],
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 4]
                                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(),
                'params': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2]
                }
            }
        }

class TransformadorDados:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.limites_peso = [0, 500, 1500, 5000, float('inf')]
        self.rotulos_peso = [500, 1500, 5000, 28000]
        self.features = Features()

    def transformar(self, df, treino, semana_inicia_em_domingo=False):
        if treino:
            df = df[df[self.features.target()[0]] == df[self.features.reference()[0]]]
            df = df[df[self.features.decision()[0]].isin(['Faturada', 'Pendente'])]
            df[self.features.target()[0]] = pd.to_datetime(df[self.features.target()[0]], format='%d/%m/%Y', errors='coerce')
            if semana_inicia_em_domingo:
                df[self.features.target()[1]] = (df[self.features.target()[0]].dt.dayofweek + 1) % 7
            else:
                df[self.features.target()[1]] = df[self.features.target()[0]].dt.dayofweek
            colunas_desejadas = [self.features.target()[0]] + self.features.categorical() + [self.features.numeric()[0]] + [self.features.numeric()[2]] + [self.features.target()[1]] + self.features.avulsos() + [self.features.id()[0]]
        else:
            colunas_desejadas = self.features.categorical() + [self.features.numeric()[0]] + [self.features.numeric()[2]] + self.features.avulsos() + [self.features.id()[0]]

        df = df[colunas_desejadas]

        df[self.features.avulsos()[0]] = pd.to_datetime(df[self.features.avulsos()[0]], format='%d/%m/%Y', errors='coerce')
        df[self.features.numeric()[1]] = df[self.features.avulsos()[0]].dt.dayofweek

        # Remove linhas com valores NaN na coluna 'Peso Líquido Estimado'
        df.dropna(subset=[self.features.avulsos()[1]], inplace=True)
        df[self.features.avulsos()[1]] = pd.to_numeric(df[self.features.avulsos()[1]], errors='coerce')
        df[self.features.avulsos()[1]].fillna(0, inplace=True)

        # Verificar se o pedido é um pedido que se repete
        df[self.features.id()[1]] = df.duplicated(subset=[self.features.id()[0]]).astype(int)
        

        # Agora você pode aplicar pd.cut
        df[self.features.numeric()[3]] = pd.cut(df[self.features.avulsos()[1]], bins=self.limites_peso, labels=self.rotulos_peso)
        df.dropna(subset=self.features.numeric()[3], inplace=True)
        df[self.features.numeric()[3]] = df[self.features.numeric()[3]].astype(int)

        df[self.features.numeric()[0]] = df[self.features.numeric()[0]].astype(int)
        df[self.features.numeric()[2]] = df[self.features.numeric()[2]].astype(int)

        if treino: df.dropna(inplace=True)

        return df

    def recortar_dataframe(self, df, num_dias):
        data_mais_recente = df[self.features.target()[0]].max()
        data_inicio = data_mais_recente - datetime.timedelta(days=num_dias)
        df_recortado = df.loc[(df[self.features.target()[0]] >= data_inicio) & (df[self.features.target()[0]] <= data_mais_recente)]
        return df_recortado
class Aprendizado:
    def __init__(self, df_treino:pd.DataFrame, df_predizer:pd.DataFrame, dir_previsao, nome_arquivo='previsao_embarques'):
        self.seed = 10
        self.num_dias = 40
        self.kfold = KFold(n_splits=10, random_state=self.seed, shuffle=True)
        self.features = Features()
        self.numeric_features = [self.features.numeric()[0], self.features.numeric()[2], self.features.numeric()[3], self.features.id()[1]]
        self.categorical_features = self.features.categorical()
        self.hiperparametros = Hiperparametros()
        self.transformador = TransformadorDados()
        self.df_treino = df_treino
        self.df_predizer = df_predizer
        self.dir_previsao = dir_previsao
        self.nome_arquivo = nome_arquivo
        self.semana_inicia_em_domingo = True

        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler()),
            ('std_scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_encoder', TargetEncoder())
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ])
        
        self.X_treino = None
        self.y_treino = None
        self.X_predizer = None
        self.informacoes_melhor_modelo = None
        self.melhor_modelo = None
        self.modelo_otimizado = None
        
    def dividir_dados_treino(self):
        df = self.transformador.transformar(self.df_treino, treino=True, semana_inicia_em_domingo = self.semana_inicia_em_domingo)
        df_recortado = self.transformador.recortar_dataframe(df, num_dias=self.num_dias)
        X = pd.DataFrame(df_recortado, columns = self.numeric_features + self.categorical_features)
        y = pd.DataFrame(df_recortado, columns = [self.features.target()[1]])
        self.X_treino = X
        self.y_treino = y.values

        # Salvar os DataFrames em um arquivo Excel
        with pd.ExcelWriter('dados_treino.xlsx') as writer:
            self.df_treino.to_excel(writer, sheet_name='df_treino')
            X.to_excel(writer, sheet_name='X_treino')
            y.to_excel(writer, sheet_name='y_treino')

        #print(X)
        #print(y)

    def dividir_dados_predicao(self):
        df = self.transformador.transformar(self.df_predizer, treino=False)
        self.X_predizer = pd.DataFrame(df, columns = self.numeric_features + self.categorical_features)


        if not os.path.exists(self.dir_previsao):
            os.makedirs(self.dir_previsao)
            print(f'A pasta {self.dir_previsao} foi criada com sucesso.')
        else:
            print(f'A pasta {self.dir_previsao} já existe.')

        file_path = os.path.join(self.dir_previsao, ('dados_predicao'+'.xlsx'))
        self.X_predizer.to_excel(file_path, index=False)

    def identificar_melhor_modelo(self):
        melhor_score = 0
        for nome_modelo, params in self.hiperparametros.get_hiperparametros().items():
            modelo = params['model']
            hiperparametros = params['params']
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', modelo)])
            score = cross_val_score(pipeline, self.X_treino, self.y_treino, cv=self.kfold, error_score='raise')
            score_medio = np.mean(score)

            if score_medio > melhor_score:
                melhor_score = score_medio
                self.informacoes_melhor_modelo = (nome_modelo, modelo, score_medio, score, hiperparametros)
                self.hiperparametros = hiperparametros
                self.melhor_modelo = modelo  

    def imprimir_informacoes_modelo_otimizado(self):
        print('A acurácia do modelo é: %.2f%%' % (self.modelo_otimizado.score(self.X_treino,self.y_treino) *100))
  
    def imprimir_informacoes_modelo(self):
        nome_modelo, modelo, score_medio, scores, hiperparametros = self.informacoes_melhor_modelo
        print("Nome do modelo:", nome_modelo)
        print("Modelo:", modelo)
        print(f"Score médio: {score_medio:.2f}")
        print("Scores unitários:")
        for i, score in enumerate(scores):
            print(f"    Fold {i+1}: {score:.2f}")
        print("Hiperparâmetros:")
        for parametro, valores in hiperparametros.items():
            print(f"    {parametro}: {valores}")
            
    
    def data_por_semana_ano(self, ano, semana, dia_semana_data_pcp):

        print(dia_semana_data_pcp)
        print(f'Parâmetros recebidos: ano={ano}, semana={semana}, dia_semana_data_pcp={dia_semana_data_pcp}, semana_inicia_em_domingo={self.semana_inicia_em_domingo}')
        
        if self.semana_inicia_em_domingo:
            data_referencia = datetime.datetime.strptime(f'{ano}-W{semana}-0', '%Y-W%W-%w')
            mapeamento_dias_semana = {
                '0': 6,
                '1': 0,
                '2': 1,
                '3': 2,
                '4': 3,
                '5': 4,
                '6': 5
            }
        else:
            data_referencia = datetime.datetime.strptime(f'{ano}-W{semana}-1', '%Y-W%W-%w')
            mapeamento_dias_semana = {
                '0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6
            }

        print(mapeamento_dias_semana)
        def encontrar_data(dia_semana_data_pcp):
            dia_semana_ingles = mapeamento_dias_semana.get(dia_semana_data_pcp, None)

            if dia_semana_ingles is not None:
                dias_diferenca = dia_semana_ingles - data_referencia.weekday()
                #if dias_diferenca < 0 and self.semana_inicia_em_domingo:
                    #dias_diferenca += 7  # Ajuste para considerar semana começando em domingo
                data_prevista = data_referencia + datetime.timedelta(days=dias_diferenca)
                return data_prevista.strftime('%d/%m/%Y')
            else:
                return "Dia da semana inválido"

        if isinstance(dia_semana_data_pcp, str):
            return encontrar_data(dia_semana_data_pcp)
        else:
            return [encontrar_data(str(dia)) for dia in dia_semana_data_pcp]

    def prever_e_salvar_data(self, ano, semana):
        previsoes = self.modelo_otimizado.predict(self.X_predizer)
        df_previsoes = pd.DataFrame(previsoes, columns=['dia_semana_data_pcp'])
        df_previsoes['dia_semana_data_pcp'] = df_previsoes['dia_semana_data_pcp'].astype(str)  # Convertendo para str

        df_previsoes['data_pcp'] = self.data_por_semana_ano(ano, semana, df_previsoes['dia_semana_data_pcp'])

        df_final = pd.concat([self.df_predizer, df_previsoes['dia_semana_data_pcp'], df_previsoes['data_pcp']], axis=1)
        df_final = df_final.reset_index(drop=True)
        file_path = os.path.join(self.dir_previsao, (self.nome_arquivo+'.xlsx'))
        df_final.to_excel(file_path, index=False)
        print(df_final)
    
    
    def prever_e_salvar(self):
        previsoes = self.modelo_otimizado.predict(self.X_predizer)
        df_previsoes = pd.DataFrame(previsoes, columns=['dia_semana_data_pcp'])
        df_final = pd.concat([self.df_predizer, df_previsoes], axis=1)
        #df_final = df_final.reset_index(drop=True)
        file_path = os.path.join(self.dir_previsao, (self.nome_arquivo+'.xlsx'))
        df_final.to_excel(file_path, index=False)
        #print(df_final)
    
    def otimizar_modelo_com_hiperparametros(self):
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', self.melhor_modelo)])
        modelo = GridSearchCV(estimator=pipeline, param_grid=self.hiperparametros, cv=self.kfold)
        modelo.fit(self.X_treino, self.y_treino)
        self.modelo_otimizado = modelo.best_estimator_

import warnings 
warnings.filterwarnings('ignore')

def execute():
    df_treino = pd.read_excel("base_completa_sem_filtro.xlsx", sheet_name="completo")
    df_teste = pd.read_excel("jj.xlsx", sheet_name="Sheet")
    diretorio_previsao = 'previsoes/'
    aprendizado = Aprendizado(df_treino, df_teste, diretorio_previsao)

    aprendizado.dividir_dados_treino()
    aprendizado.identificar_melhor_modelo()
    aprendizado.imprimir_informacoes_modelo()
    aprendizado.otimizar_modelo_com_hiperparametros()

    aprendizado.dividir_dados_predicao()

    hoje = datetime.date.today()
    proxima_semana = hoje + datetime.timedelta(7)

    numero_proxima_semana = proxima_semana.isocalendar()[1]
    ano_proxima_semana = proxima_semana.isocalendar()[0]

    print(f"O número da próxima semana é {numero_proxima_semana} e o ano é {ano_proxima_semana}.")

    modelo = aprendizado.modelo_otimizado

    aprendizado.prever_e_salvar()
    aprendizado.prever_e_salvar_data(ano_proxima_semana, numero_proxima_semana)