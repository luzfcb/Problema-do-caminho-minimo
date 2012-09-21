#!/usr/bin/env python
# -*- coding: utf8 -*-
# Testado unicamente no Ubuntu 12.04 64bits, Python 2.7.3

from __future__ import print_function #import da nova versao do metodo print, compativel com python 3

import argparse
import random

import timeit
import time
import math

import pickle
import sys
import os


pygraph_instalado = True
try:
#    import gv
#
#    from pygraph.classes.graph import graph
#    from pygraph.classes.digraph import digraph
#    from pygraph.algorithms.searching import breadth_first_search
#    from pygraph.readwrite.dot import write


    #necessario para o metodo gerar_visualizacao()
    import pydot
    from PIL import Image # eh necessario instalacao do PIL com suporte a libjpej e libpng

except :
    print('pygraph , Python Imaging Library(PIL), pydot pode nao estar instalado')
    pygraph_instalado = False

class SupressorImpressao:

    def write(self, _in):
        pass



class Caminho(object):
    global ma
    global t

    #inicializador, em python você não precisa utilizar o construtor diretamente
    #voce simplesmente sobrescreve o metodo inicializador
    def __init__(self, numero_vertices=10, matriz={}, debug=False, iniciar_indices_em_zero=False, gerar_visul=False):

        self.iniciar_indices_em_zero = iniciar_indices_em_zero
        if self.iniciar_indices_em_zero is False:
            #faz indices comecarem em 1 e nao em zero
            self.numero_vertices = numero_vertices + 1
            self.indice_inicial = 1
        else:
            self.numero_vertices = numero_vertices
            self.indice_inicial = 0

#        print('teset: ',self.numero_vertices)
        self.infinito = sys.maxint
        self.lista_de_vertices = xrange(self.indice_inicial, self.numero_vertices)
        self.matriz = matriz
        self.debug = debug
        self.ha_aresta_negativa = False
        self.gerar_visul = gerar_visul


#
#    def gerar_img(self):
#
#        if pygraph_instalado:
#            # Graph creation
#            gr = graph()
#
#
#            # Add nodes and edges
#            gr.add_nodes(["Portugal","Spain","France","Germany","Belgium","Netherlands","Italy"])
#            gr.add_nodes(["Switzerland","Austria","Denmark","Poland","Czech Republic","Slovakia","Hungary"])
#            gr.add_nodes(["England","Ireland","Scotland","Wales"])
#
#            gr.add_edge(edge=("Portugal", "Spain"), wt=22)
#            gr.add_edge(("Spain","France"), 50)
#            gr.add_edge(("France","Belgium"))
#            gr.add_edge(("France","Germany"))
#            # Draw as PNG
#            dot = write(gr)
#            gvv = gv.readstring(dot)
#            gv.layout(gvv,'dot')
#            gv.render(gvv,'png','europe.png')

    #utiliza o pydot para gerar um grafo, utilizando a notacao do Graphviz
    def gerar_visualizacao(self, nome_arq=None, gerar_arq_dot=False, mater_nos_nao_utilizados=False):
        if self.gerar_visul is True:
            if pygraph_instalado:
                grafo = pydot.Dot(
                    'Grafo Direcionado',
                    graph_type='digraph',
                    simplify=False,
                    layout="dot",
                    #layout="fdp",
                    #overlap=None,
                    overlap="scale",
                    splines=True,
                    #splines=True,
                    #size=10000,
                    #nojustify=True,
                    #repulsiveforce=0.5,
                    #target=4
                    #overlap="prism"
                    #sep=True
                )

                nos = {}
                arestas = []
                qnt = len(self.lista_de_vertices)
                if mater_nos_nao_utilizados:
                    for verticeX in self.lista_de_vertices:
                        nos[verticeX] = pydot.Node(name=verticeX, #shape='doublecircle'
                        )


                for verticeX in self.lista_de_vertices:
                    for verticeY in self.lista_de_vertices:
                        if self.matriz[verticeX,verticeY] != 0:
                            if mater_nos_nao_utilizados:

                                aresta = pydot.Edge(src=nos[verticeX], dst=nos[verticeY], label=self.matriz[verticeX,verticeY],
                                    labelfontcolor='green', color='red', fontcolor ='green' ,
                                    repulsiveforce=0.5)
                            else:
                                aresta = pydot.Edge(src=str(verticeX), dst=str(verticeY), label=self.matriz[verticeX,verticeY],
                                    labelfontcolor='green', color='red', fontcolor ='green' ,
                                    repulsiveforce=0.5)

                            arestas.append(aresta)
                if mater_nos_nao_utilizados:
                    for no in nos.itervalues():
                        grafo.add_node(no)

                for aresta in arestas:
                    grafo.add_edge(aresta)
                n_arq = ''
                if nome_arq is None:
                    n_arq = ('%s.png' % len(self.lista_de_vertices))

                if '.png' not in n_arq:
                    n_arq = n_arq.join('.png')

                grafo.write_png(n_arq)
                if gerar_arq_dot:
                    grafo.write_dot(n_arq.replace('.png', '.dot') )
                #a = grafo.read_dot(('%s.dot' % len(self.iterador_lista_vertices)) )
                #print(a)
                #im=Image.open( ('%s.png' % len(self.lista_de_vertices)))
                #im.show()


    #gera um Grafo com numero de vertices
    def gerarGrafoAleatorio(self, tamanho=None, nao_direcionado=True, ):
        if tamanho is not None:
            self.__init__(numero_vertices = tamanho, gerar_visul=self.gerar_visul)

        for i in self.lista_de_vertices:
            for j in self.lista_de_vertices:
                self.matriz[i,j] = 0
        num_criacoes = 0
        max_criacoes = len(self.lista_de_vertices) + len(self.lista_de_vertices) + (len(self.lista_de_vertices)) /2
        for i in self.lista_de_vertices:
            for j in self.lista_de_vertices:
                if i != j:
                    numero_aleatorio = int(random.randrange(0, 100))

                    if num_criacoes <  max_criacoes:

                        if (numero_aleatorio % 3) ==0:#print numero_aleatorio
                            if nao_direcionado is True:
                                self.matriz[i,j] = numero_aleatorio
                                self.matriz[j,i] = numero_aleatorio
                            else:
                                self.matriz[i,j] = numero_aleatorio
                            #matriz[j,i] = numero_aleatorio
                            if self.debug is True:
                                print( i,j,':', numero_aleatorio)
                        else:
                            if nao_direcionado is True:
                                self.matriz[i,j] = 0
                                self.matriz[j,i] = 0
                            else:
                                self.matriz[i,j] = 0
                            #matriz[j,i] = numero_aleatorio
                            if self.debug is True:
                                print( i,j,':', 0)

                        num_criacoes += 1
                else:
                    self.matriz[i,j] = 0
                    self.matriz[j,i] = 0

        if self.debug is True:
            print( self.matriz.keys())
            print(self.matriz.keys()[1][1])

    def comparar_algoritmos(self,origem, destino, metrica_utilizar_saltos=False):
        #suprimindo impressao
        e = SupressorImpressao()
        saida_anterior = sys.stdout
        sys.stdout = e
        percurso, todos_os_caminhos, tempo_exec = self.caminho_mais_curto(origem, destino)

        percurso_bellman, todos_os_caminhos_bellman, tempo_exec_bellman = self.caminho_mais_curto(origem,destino, utilizar_bellman_ford=True)

        percurso_salto, todos_os_caminhos_salto, tempo_exec_salto = self.caminho_mais_curto(origem, destino, metrica_utilizar_saltos=True)

        percurso_bellman_salto, todos_os_caminhos_bellman_salto, tempo_exec_bellman_salto = self.caminho_mais_curto(origem,destino, metrica_utilizar_saltos=True, utilizar_bellman_ford=True)


        #restaurando stdout para o padrao do sistema
        #sys.stdout = sys.__stdout__
        sys.stdout = saida_anterior
        print('dijkstra     : %s, executado em %ss' % (percurso, tempo_exec))
        print('bellman_ford : %s, executado em %ss' % (percurso_bellman, tempo_exec_bellman))
        print('dijkstra metrica salto    : %s, executado em %ss' % (percurso_salto, tempo_exec_salto))
        print('bellman_ford metrica salto: %s, executado em %ss' % (percurso_bellman_salto, tempo_exec_bellman_salto))



    #serializa em arquivo a instancia das variaveis matriz e numero_vertices
    def gravar_matriz_dat(self):
        file_name = 'backup_matriz.dat'
        the_file = open(file_name, 'wb')
        pickle.dump(self.matriz, the_file)
        the_file.close()

        file_name2 = 'backup_num_vertices.dat'
        the_file2 = open(file_name2, 'wb')
        pickle.dump(self.numero_vertices, the_file2)
        the_file2.close()
    #le arquivos serializados
    def ler_matriz_dat(self):
        file_name = 'backup_matriz.dat'
        the_file = open(file_name)
        self.ma = pickle.load(the_file)
        self.matriz = self.ma
        file_name2 = 'backup_num_vertices.dat'
        the_file2 = open(file_name2)
        self.t = pickle.load(the_file2)
        self.numero_vertices = self.t

    #imprime a matriz, caso seja passado parametro nome_arq, ele gera um arquivo com a matriz
    def imprimeMatriz(self):

        for i in self.lista_de_vertices:
            print('[', end='')
            for j in self.lista_de_vertices:
                print('[', self.matriz[i,j] , ']','',sep='', end = '')
            print(']')



    #imprime a matriz
    def imprimeMatriz_adjacencia(self):

        print(self.numero_vertices)
        for i in self.lista_de_vertices:
            for j in self.lista_de_vertices:
                print(self.matriz[i,j] , ' ','',sep='', end = '')
            print('')

    def salvarMatrizAdjacencia(self, nome_arq):
        print(nome_arq)
        arq = file(nome_arq, 'w')


        arq.write(str('%s\n' % (self.numero_vertices -1)))
        for i in self.lista_de_vertices:
            for j in self.lista_de_vertices:
                arq.write( '%s ' % self.matriz[i,j])
            arq.write('\n')
        arq.close()

    def le_arquivo(self,nome_do_arquivo):

        self.arq = open(nome_do_arquivo)
        tamanho_matriz = int(self.arq.readline())
        print('Quantidade de Vertices:', tamanho_matriz)
        matriz = []
        ma ={}

        linha = self.indice_inicial
        coluna = self.indice_inicial
        nome_arq = nome_do_arquivo.split(os.path.sep)[-1]
        print('Iniciando leitura do arquivo', nome_arq)
        t_inicial = timeit.default_timer()
        aresta_negativa = False
        while 1:
            coluna = self.indice_inicial
            linha_arq = self.arq.readline()

            if linha_arq == "":
                break
            pos = linha_arq.replace('\r\n', '').split()
            for a in pos:

                ma[linha,coluna] = int(a)
                if ma[linha,coluna] < 0:
                    aresta_negativa = True
                coluna += 1

            linha += 1
        t_final = timeit.default_timer()
        print('Leitura do arquivo %s terminada em %ss' % (nome_arq, (t_final - t_inicial)))

        self.__init__(tamanho_matriz, ma, self.debug, self.iniciar_indices_em_zero)
        self.ha_aresta_negativa = aresta_negativa

        return [tamanho_matriz,ma]

    def adicionar_aresta(self, vertice_origem, vertice_destino, custo, direcionada=True):
        if vertice_destino not in self.lista_de_vertices or vertice_destino not in self.lista_de_vertices:
            print('Vertice Origem ou destino fora dos vertices cadastrados')
            return None

        if direcionada:
            self.matriz[vertice_origem, vertice_destino] = custo
        else:
            self.matriz[vertice_origem, vertice_destino] = custo
            self.matriz[vertice_destino, vertice_origem] = custo

    def remover_aresta(self, vertice_origem, vertice_destino, custo, direcionada=True):
        if vertice_destino not in self.lista_de_vertices or vertice_destino not in self.lista_de_vertices:
            print('Vertice Origem ou destino fora dos vertices cadastrados')
            return None

        if direcionada:
            self.matriz[vertice_origem, vertice_destino] = 0
        else:
            self.matriz[vertice_origem, vertice_destino] = 0
            self.matriz[vertice_destino, vertice_origem] = 0

    def buscaProfundidade(self,origem, destino, visitado=None):


        ordem = []

        if visitado is None:
            visitado = {}
            for vertice in self.lista_de_vertices:
                visitado[vertice] = False

        visitado[origem] = True
        ordem.append(origem)
        print(origem, sep=',', end = '')
        for vertice in [i for i in self.lista_de_vertices if self.matriz[origem,i] != 0 ]:
            if visitado[vertice] is False:
                self.buscaProfundidade(origem, destino, visitado)


    #http://www.php2python.com/wiki/function.microtime/
    def microtime(self, get_as_float = False) :
        if get_as_float:
            return time.time()
        else:
            return '%f %d' % math.modf(time.time())
    #http://code.activestate.com/recipes/358361-non-exponential-floating-point-representation/
    def non_exp_repr(self,x):

        """Return a floating point representation without exponential notation.

        Result is a string that satisfies:
            float(result)==float(x) and 'e' not in result.

        >>> non_exp_repr(1.234e-025)
        '0.00000000000000000000000012339999999999999'
        >>> non_exp_repr(-1.234e+018)
        '-1234000000000000000.0'

        >>> for e in xrange(-50,51):
        ...     for m in (1.234, 0.018, -0.89, -75.59, 100/7.0, -909):
        ...         x = m * 10 ** e
        ...         s = non_exp_repr(x)
        ...         assert 'e' not in s
        ...         assert float(x) == float(s)

        """
        s = repr(float(x))
        e_loc = s.lower().find('e')
        if e_loc == -1:
            return s

        mantissa = s[:e_loc].replace('.', '')
        exp = int(s[e_loc+1:])

        assert s[1] == '.' or s[0] == '-' and s[2] == '.', "Unsupported format"
        sign = ''
        if mantissa[0] == '-':
            sign = '-'
            mantissa = mantissa[1:]

        digitsafter = len(mantissa) - 1     # num digits after the decimal point
        if exp >= digitsafter:
            return sign + mantissa + '0' * (exp - digitsafter) + '.0'
        elif exp <= -1:
            return sign + '0.' + '0' * (-exp - 1) + mantissa
        ip = exp + 1                        # insertion point
        return sign + mantissa[:ip] + '.' + mantissa[ip:]
#------------------------------------------------------------------------------------------

    def menor_indice_de_distancia(self,q, dist):
            menor = self.infinito
            verti = -1

            for i in q:
                if dist[i] < menor:
                    menor = dist[i]
                    verti = i

            return verti



    def dijkstra(self,origem, metrica_utilizar_saltos=False):
        #inicilizacao
        distancia = {}
        anterior = {}

        for vertice in self.lista_de_vertices:
            distancia[vertice] = self.infinito
            anterior[vertice] = None
        distancia[origem] = 0

        q = [i for i in self.lista_de_vertices]

        t_inicial = timeit.default_timer()
        #Processamento de Caminhos minimos
        while len(q) > 0:
            u = self.menor_indice_de_distancia(q, distancia)
            if u == -1:
                break
            if distancia[u] == self.infinito:
                break
            q.remove(u)
            visinhos = [i for i in self.lista_de_vertices if self.matriz[u,i] > 0 and i in q]

            for visinho in visinhos:
                if metrica_utilizar_saltos is False:
                    alt = distancia[u] + self.matriz[u, visinho]
                else:
                    alt = distancia[u] + 1

                if alt < distancia[visinho]:
                    distancia[visinho] = alt
                    anterior[visinho] = u

        t_final = timeit.default_timer()

        return distancia, anterior, (t_final - t_inicial)



    def bellman_ford(self,origem, metrica_utilizar_saltos=False):
        #inicilizacao
        distancia = {}
        anterior = {}

        for vertice in self.lista_de_vertices:
            distancia[vertice] = self.infinito
            anterior[vertice] = None
        distancia[origem] = 0
        ciclo_negativo = False
        q = [i for i in self.lista_de_vertices]
        #print('iniciando bellman_ford...')
        t_inicial = self.microtime(True)

        #Processamento de Caminhos minimos
        for cada_vertice in self.lista_de_vertices:
            for u in self.lista_de_vertices:
                for v in self.lista_de_vertices:
                    if self.matriz[u,v] != 0:
                        distan = None
                        if metrica_utilizar_saltos is False:
                            distan = distancia[u] + self.matriz[u,v]
                        else:
                            distan = distancia[u] + 1
                        if distancia[v] > distan:
                            distancia[v] = distan
                            anterior[v] = u
        #Checando ciclo negativo
        print('Checando ciclo negativo')
        for u in self.lista_de_vertices:
            for v in self.lista_de_vertices:
                if self.matriz[u,v] != 0:
                    distan = None
                    if metrica_utilizar_saltos is False:
                        distan = distancia[u] + self.matriz[u,v]
                    else:
                        distan = distancia[u] + 1
                    if distancia[v] > distan:
                        t_final = self.microtime(True)
                        #print('terminando bellman_ford...')
                        ciclo_negativo = True
                        return distancia, anterior, (t_final - t_inicial), ciclo_negativo
        t_final = self.microtime(True)
        #print('terminando bellman_ford...')
        return distancia, anterior, (t_final - t_inicial), ciclo_negativo


    def caminho_mais_curto(self,origem, destino, metrica_utilizar_saltos=False, utilizar_bellman_ford=False):
        if origem < self.indice_inicial or origem > self.numero_vertices -1 \
           or destino < self.indice_inicial or destino > self.numero_vertices-1:

            print('Origem ou Destino nao sao vertices validos')
            return 'nao validao', 'nao validao','nao validao'
        ciclo_negativo = False
        percurso = []

        if metrica_utilizar_saltos:
            print('Utilizando Metrica de Saltos')

        if self.ha_aresta_negativa is True or utilizar_bellman_ford is True:
            print('Utilizando bellman_ford')
            distancia, anterior, tempo_total , ciclo_negativo= self.bellman_ford(origem, metrica_utilizar_saltos)
        else:
            print('Utilizando dijkstra')
            distancia, anterior, tempo_total = self.dijkstra(origem, metrica_utilizar_saltos)
        print('Trassando caminho..')
        #obtendo percurso
        if destino is not None:
            if ciclo_negativo is not True:
                while anterior[destino] is not None :
                    percurso.append(destino)
                    destino = anterior[destino]
            else:
                print('Ha ciclo negativo')

        percurso.append(origem)
        percurso.reverse()
        if 'e-' in str(tempo_total):
            tempo_total = (self.non_exp_repr(tempo_total))

        print('Executado em: %ss\nCaminho:\n%s' % (tempo_total,percurso))
        return percurso, distancia, tempo_total



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grafos e Algoritmos\nCriado por Fabio C. Barrionuevo')
    parser.add_argument('-a', action='store', dest='arquivo',
        #default=[],
        help='Arquivo com matriz de adjacencia',
    )
    parser.add_argument('-o', action='store', dest='v_origem', type=int,
        #default=[],
        help='Vertice origem',
    )
    parser.add_argument('-d', action='store', dest='v_destino', type=int,
        #default=[],
        help='Vertice destino',
    )
    parser.add_argument('-s', action='store', dest='arq_dest',
        #default=[],
        help='Salvar saida no arquivo destino',
    )
    parser.add_argument('-g', action='store', dest='num_vertices', type=int,
        #default=[],
        help="Gerar grafo aleatorio baseano no em num_vertices, e salvar matriz de adjacencia em arquivo nomeado como o matriz_adcancencia_'num_vertices'.txt",
    )
    parser.add_argument('-nd', action='store', dest='direcionado', type=int,
        #default=[],
        help="0 para nao direcionado, 1 para direcionado, default = 0",
    )
    parser.add_argument('-v', action='store', dest='visul', type=int,
        #default=[],
        help="0 para nao gerar arquivo png com a visualizacao do grafo , 1 para gerar arquivo png com a visualizacao do grafo, default = 0",
    )
    result = parser.parse_args()

    DIRETORIO_ARQUIVO = os.path.abspath(os.path.dirname(__file__))

    nome_saida = result.arq_dest
    gerar_visul = False
    if result.visul is not None:
        gerar_visul = True

    if result.arq_dest is not None:
        if '.txt' not in result.arq_dest:
            nome_saida.join('.txt')
        #redirecionando saida STDOUT, que eh utilizada pelo comando print, para um arquivo
        print('Processando caminhos e salvando no arquivo %s\nDependendo do numero de vertices e arestas do isso pode demorar muito tempo.\nAguarde...' % nome_saida)
        sys.stdout = file(nome_saida, 'w')

    if result.arquivo is not None and result.v_origem is not None and result.v_destino is not None:
        caminho = Caminho(debug=False, gerar_visul=gerar_visul)
        caminho.le_arquivo(result.arquivo)
        caminho.caminho_mais_curto(result.v_origem, result.v_destino)
        caminho.buscaProfundidade(result.v_origem, result.v_destino)
        caminho.gerar_visualizacao()

    elif result.num_vertices is not None:
        caminho = Caminho(numero_vertices=result.num_vertices, debug=False, gerar_visul=gerar_visul)
        if result.direcionado is not None:
            caminho.gerarGrafoAleatorio(nao_direcionado=False)
        else:
            caminho.gerarGrafoAleatorio()
        caminho.salvarMatrizAdjacencia(str('matriz_adcancencia_%s.txt' % result.num_vertices))
        caminho.gerar_visualizacao()
        del caminho
    else:
        print("Execute 'python %s -h', para verificar mais opcoes\nNao foram passados todos os parametros, executando o padrao..." % __file__)
        time.sleep(3)

        #----------------------------------------------------
        print( '#----------------------------------------------------')
        caminho = Caminho(debug=False, gerar_visul=gerar_visul)
        ARQ = os.path.join(DIRETORIO_ARQUIVO, 'matrix.txt')
        caminho.le_arquivo(ARQ)
        #caminho.caminho_mais_curto(1,6)
        caminho.comparar_algoritmos(1,6)
        caminho.gerar_visualizacao()
        del caminho
        #----------------------------------------------------
        print( '#----------------------------------------------------')
        caminho = Caminho(debug=False, gerar_visul=gerar_visul)
        ARQ = os.path.join(DIRETORIO_ARQUIVO, '50.txt')
        caminho.le_arquivo(ARQ)
        #caminho.caminho_mais_curto(10,40)
        caminho.comparar_algoritmos(10,40)
        caminho.gerar_visualizacao()
        del caminho
        #----------------------------------------------------
        print( '#----------------------------------------------------')
        caminho = Caminho(debug=False, gerar_visul=gerar_visul)
        ARQ = os.path.join(DIRETORIO_ARQUIVO, '250.txt')
        caminho.le_arquivo(ARQ)
        #caminho.caminho_mais_curto(10,220)
        caminho.comparar_algoritmos(10,220)
        caminho.gerar_visualizacao()
        del caminho
        #----------------------------------------------------
        print( '#----------------------------------------------------')
        caminho = Caminho(debug=False, gerar_visul=gerar_visul)
        ARQ = os.path.join(DIRETORIO_ARQUIVO, '500.txt')
        caminho.le_arquivo(ARQ)
        #caminho.caminho_mais_curto(1,350)
        caminho.comparar_algoritmos(1,350)
        caminho.gerar_visualizacao()
        del caminho
        #----------------------------------------------------
        print( '#----------------------------------------------------')
        caminho = Caminho(debug=False, gerar_visul=gerar_visul)
        ARQ = os.path.join(DIRETORIO_ARQUIVO, '750_2.txt')
        caminho.le_arquivo(ARQ)
        #caminho.caminho_mais_curto(1,700)
        caminho.comparar_algoritmos(1,700)
        caminho.gerar_visualizacao()
        del caminho
        #----------------------------------------------------
        print( '#----------------------------------------------------')
        caminho = Caminho(debug=False, gerar_visul=gerar_visul)
        ARQ = os.path.join(DIRETORIO_ARQUIVO, '1000.txt')
        caminho.le_arquivo(ARQ)
        #caminho.caminho_mais_curto(1,900)
        caminho.comparar_algoritmos(1,900)
        caminho.gerar_visualizacao()
        del caminho
