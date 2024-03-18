import pandas as pd

class Mconfusao:
    def __init__(self, items, incon = False):
        # items : lista com o nome de cada item avaliado
        # incon : se terá ou não espaço para casos inconclusivos

        self.itens_list = items     # lsita com os itens avaliados na matriz de confusão
        self.incon = incon          # Se aceita ou não inconclusivo
        self.incon_var = "I"        # nome do inconclusivo

        # criação do dataframe que será usado como matriz de confusão
        self.matriz = pd.DataFrame(
            columns= items.copy() + [self.incon_var] if incon else items,
            index= items
        ).applymap(lambda x: 0)
        
        return
    
    def insert_row (self, item_linha, *ans):
        # inseri todos os valores correspondentes a uma linha da matriz de confusão
        # na ordem que foram inseridos

        # completa a resposta caso possua menos valores que o necessário
        while len(ans) < len(self.itens_list):
            ans += (0,)
    
        self.matriz.loc[item_linha] = ans + (0,) if len(ans) == len(self.itens_list) and self.incon else ans
        return
    
    def add_one (self, item_linha, item_coluna):
        # adiciona +1 ao contador do item desejado
        """
        item_linha = resposta correta
        item_coluna = resposta prevista
        """

        self.matriz.loc[item_linha][item_coluna] += 1
        return
    
    def insert_value (self, item_linha, item_coluna, value):
        # inseri o valor desejado ao item desejado

        self.matriz.loc[item_linha][item_coluna] = value
        return
    
    def analytics (self):
        # calcula o total de amostras
        self.total = (self.matriz.sum()).sum()
        
        # calcula Positivos Verdadeiros
        self.TP = 0
        for i in self.itens_list:
            self.TP += self.matriz.loc[i, i]

        # calcula inconclusivos
        if self.incon:
            self.I = self.matriz.sum(axis='index')[-1]
        
        # calcula Falsos Negativos
        self.FN = self.total - self.TP - self.I if self.incon else self.total - self.TP

        # Acurácia e Precisão gerais
        self.Accuracy = self.TP/self.total
        self.Precision = self.TP/(self.total - (self.I if self.incon else 0))
        
        return
    
    def great_analysis (self):

        ind_sum = self.matriz.sum(axis='columns')   # soma das linhas
        col_sum = self.matriz.sum(axis='index')     # soma das colunas

        # dataframe das estatisticas
        self.statistics = pd.DataFrame(
            columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            index = self.itens_list
        ).applymap(lambda x: 0)

        for i in range(len(self.itens_list)):
            prec = self.matriz.iloc[i, i]/(ind_sum[i] - (self.matriz.iloc[i, -1] if self.incon else 0))
            rec = self.matriz.iloc[i, i]/col_sum[i]

            self.statistics.loc[self.itens_list[i]] = [
                self.matriz.iloc[i, i]/ind_sum[i],      # Accuracy
                prec,                                   # Precision
                rec,                                    # Recall
                2*prec*rec/(prec + rec)                 # F1-Score
            ]
            
        print(self.statistics.applymap(lambda x: round(x, 3)))
        return

    def render(self, type = 'abs'):
        """
        type:   abs = valores absolutos
                per = valores percentuais
        """
        
        self.analytics()

        match(type):
            case 'abs':
                # visualização da matriz de confusão absoluta
                print(self.matriz)
            case 'per':
                # renderização percentual
                print(self.matriz.applymap(lambda x: round(x/self.total, 3)))
        
        # estatisticas gerais
        print(f"Total amostras: {self.total}")
        print(f"Precisão: {round(self.Precision, 3)}")
        print(f"Acurácia: {round(self.Accuracy, 3)}")
        if self.incon:
            print(f"Inconclusivos: {round(self.I/self.total, 3)}")
        return

if __name__ == "__main__":
    # exemplo/teste
    m = Mconfusao(['A', 'B', 'C', 'D', 'E'], incon=True)
    
    m.insert_value('D', 'D', 29)
    m.insert_value('E', 'E', 21)
    m.insert_value('C', 'C', 30)
    m.insert_value('E', 'C', 2)

    # resultados não inconclusivos
    # m.insert_row('A', 30, 0, 0, 0, 0)
    # m.insert_row('B', 0, 30, 0, 0, 0)
    
    # inconclusivo
    m.insert_row('A', 30, 0, 0, 0, 0, 0)
    m.insert_row('B', 0, 30, 0, 0, 0, 0)
    m.insert_value('E', 'I', 7)
    m.add_one('D', 'I')

    m.render('per')
    m.great_analysis()
    #m.render('abs')