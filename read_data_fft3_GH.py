# -*- coding: utf-8 -*-
# Python 3.6
# Created by A. Brhian and M. Ridenti

# Comand in CMD: "cd.." (sobe pastas) "cd/d G:" (altera HD)
# Comand in CMD: [cd Meu Drive\ITA_Brhian\Script]
# Comand in CMD: python read_data_fft3.py data_test input_samples

# The data from csv need to be separated by spacing " " and the decimal represented by point "."

# Last update on 08/31/2023 by A. BRHIAN

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats
import pymannkendall as mk
from scipy import signal
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# Start time the program
ini_prog = time.time()#;print(ini_prog)

# Warning message
if len(sys.argv) != 3:
    #print('Program accepts only two arguments, the relative directory path where the station data files are stored and the parameter input file, with extension .dat')
    print('Program accepts only two arguments, the relative directory path where the station data files are stored and the relative directory path where the files inputs are stored')
    exit()

###################################################################################
#
# Functions
#
###################################################################################

def month_days(month):
    """
    Defining the acumulated days in each month
    """
    if month == 1:
        days = 0
    elif month == 2:
        days = 31
    elif month == 3:
        days = 59
    elif month == 4:
        days = 90
    elif month == 5:
        days = 120
    elif month == 6:
        days = 151
    elif month == 7:
        days = 181
    elif month == 8:
        days = 212
    elif month == 9:
        days = 243
    elif month == 10:
        days = 273
    elif month == 11:
        days = 304
    elif month == 12:
        days = 334
    return days

def station_days(estacao):
    """
    Label some period accoording their seasons
    """
    ano = 366 #referent on 2014
    if estacao == "su":# summer
        int_estacao = np.concatenate((np.arange(354.5, 366+0.5, 0.5),
                                     np.arange(0, 79+0.5, 0.5)))
    elif estacao == "fa":# fall
        int_estacao = np.arange(79.5, 171+0.5, 0.5)
    elif estacao == "wi":# winter
        int_estacao = np.arange(171.5, 264+0.5, 0.5)
    elif estacao == "sp":# spring
        int_estacao = np.arange(264.5, 354+0.5, 0.5)
    elif estacao == "al":# all year
        int_estacao = np.arange(0, ano+0.5, 0.5)
    elif estacao == "fw":# fall-winter
        int_estacao = np.arange(79.5, 264+0.5, 0.5)
    elif estacao == "ss":# spring-summer
        int_estacao = np.concatenate((np.arange(264.5, 354+0.5, 0.5),
                                      np.arange(354.5, 366 + 0.5, 0.5),
                                      np.arange(0, 79 + 0.5, 0.5)))
    return int_estacao

def func1(x, a, b, c):
    """
    Definition of the parabolic fuction for the background fitting
    """
    return a*x**2 + b*x + c

def func2(x, A, B, la):
    """
    Definition of the cosine and sine fuction for GW fitting
    """
    return A * np.cos(2 * np.pi * x / la) + B * np.sin(2 * np.pi * x / la)

def func3(x, phase, A, la):
    """
    Definition of the cosine fuction for GW fitting
    """
    return A * np.cos(2 * np.pi * x / la + phase)

def func(x, a, b, c, A, B, la):
    """
    Definition of the cosine + parabolic fuction for GW complete fitting
    """
    return a * x ** 2 + b * x + c + A * np.cos(2 * np.pi * x / la) + B * np.sin(2 * np.pi * x / la)

def call_old_fitting(lamb_f, h_f, w_f):
    """"
    Definition to call the old fitting routine (fit parabola first, then fit sine wave)
    """
    qui2_f = np.zeros(shape=(len(lamb_f),))
    phase_val_f = np.zeros(shape=(len(lamb_f),))
    amp_val_f = np.zeros(shape=(len(lamb_f),))
    for l in range(0, len(lamb_f)):
        popt_f, pcov_f = curve_fit(lambda x, A, B: func2(x, A, B, lamb_f[l]), h_f, w_f)
        w_fit = func2(h_f, popt_f[0], popt_f[1], lamb_f[l])
        qui2_f[l] = np.sum((w_fit - w_f) ** 2)
        phase_val_f[l] = np.arctan2(-popt_f[1], popt_f[0]) * 180 / np.pi
        amp_val_f[l] = np.sqrt(popt_f[1] ** 2 + popt_f[0] ** 2)
    return qui2_f, phase_val_f, amp_val_f

def salvaVariavel(variavel, mean, h_reg):
    """
    Save information of winds, temperature and pression
    The labels are: "No"=northward wind, "Ea"=eastward wind, "Te"=temperature, and "Pr"=pression
    """

    # Salve the data from variable in bd_Var.csv

    # Creating the head of table
    intervalo = [50 * x for x in range(601)]# Altitudes coleted (601-1)*50=30000
    todoAlt = str(intervalo)
    todoAlt = str.replace(todoAlt, ",", ";")
    todoAlt = str.replace(todoAlt, "[", "")
    todoAlt = str.replace(todoAlt, "]", "")
    todoAlt = str.replace(todoAlt, " ", "")
    cabecalho = "aerodromo;ano;camada;estacao;variavel;data_compilacao;hora_compilacao;"+todoAlt+"\n"

    # Reading (try) or creating (except) a archive csv
    try:
        analyticVar2 = open(directory[:-10] + "analytic" + "\\" + "bd_Var" + ".csv", "r")
        lista = analyticVar2.readlines()
        analyticVar2.close()
        print("\n /!\ A planilha \n\t", analyticVar2, "\n foi LIDA com sucesso!\n")
    except:
        analyticVar2 = open(directory[:-10] + "analytic" + "\\" + "bd_Var" + ".csv", "w")
        analyticVar2.writelines(cabecalho)
        analyticVar2.close()
        analyticVar2 = open(directory[:-10] + "analytic" + "\\" + "bd_Var" + ".csv", "r")
        lista = analyticVar2.readlines()
        analyticVar2.close()
        print("\n /!\ A nova planilha \n\t", analyticVar2, "\n foi CRIADA!\n")

    # Preparing the data that will be salve
    now = datetime.now() # Time of save
    now_esp = now.strftime("%Y/%m/%d;%H:%M:%S")
    variavel = variavel

    dadosColetado = [] # Putting the data of station in all days of the year
    cont = 0
    for i in range(len(intervalo)):
        #print(intervalo[i], h_reg[cont],intervalo[i] == h_reg[cont])
        if intervalo[i] == h_reg[cont]:
            dadosColetado.insert(i, mean[cont]) # If in day HAVE info
            if cont < (len(h_reg)-1):
                cont += 1
        else:
            dadosColetado.insert(i, []) # If in day DO NOT HAVE info

    dadosColetado = str.replace(str(dadosColetado), ",", ";") # Formating string
    dadosColetado = str.replace(dadosColetado, "[", "")
    dadosColetado = str.replace(dadosColetado, "]", "")
    dadosColetado = str.replace(dadosColetado, " ", "")

    indicador_ini = aer_cam_ano_est[0:4]+";"+str(year)+";"+cam+";"+ESTACAO+";"+variavel+";"+now_esp
    salvar = indicador_ini+";"+dadosColetado+"\n"

    # Delete old info and update data
    existe=False
    for i in range(len(lista)): # This program locate the aerodrome in archive
        # print(indicador_ini[:14], lista[i][:14], indicador_ini[:14] == lista[i][:14],
        #       '\n', indicador_ini[:34], lista[i][:34], indicador_ini[:34] != lista[i][:34])
        if indicador_ini[:16] == lista[i][:16] and indicador_ini[:36] != lista[i][:36]:
            lista.pop(i)
            lista.append(salvar)
            lista.sort()
            existe = True
            print(" /!\ Os valores da \n\t", aer_cam_ano_est+variavel, "\n foram ATUALIZADAS no arquivo!\n")
            break # The aerodrome will be recorded only one time

    # If the archive do not have of aerodrome
    try:
        if existe == False and lista[0] == cabecalho:
            lista.append(salvar)
            lista.sort()
            print(" /!\ Os valores da \n\t", aer_cam_ano_est+variavel, "\n foram ADICIONADAS no arquivo!\n")
    except: # If the archive is empty
        if lista == []:
            lista.append(cabecalho)
            lista.append(salvar)
            lista.sort()
            print(" /?\ O arquivo estava vazio e uma lista foi CRIADA!\n")

    # Recording new list in csv
    analyticVar = open(directory[:-10] + "analytic" + "\\" + "bd_Var" + ".csv", "w")
    analyticVar.writelines(lista)
    analyticVar.close()
    print (" /!\ A planilha: \n\t", analyticVar, "\n acaba de ser FECHADA!\n")

    return None

def salvaFFT(variavel_FFT, value_FFT, day_FFT):
    """"
    Function for saving variables from FFT

    The variables started with: 'nw_', 'ew_' ou 'te_';
    and end with: 'lam', 'pha', 'amp' ou 'rel' 'par_a', 'par_b', andy 'par_c'
    """

    # Saving data of variable in (bd_FFT.csv)
    # Creating the head of table
    intervalo = [0.5 * x for x in range((366 + 2) * 2-1)] # Day of collect
    todoAlt = str(intervalo)
    todoAlt = str.replace(todoAlt, ",", ";")
    todoAlt = str.replace(todoAlt, "[", "")
    todoAlt = str.replace(todoAlt, "]", "")
    todoAlt = str.replace(todoAlt, " ", "")
    cabecalho = "aerodromo;ano;camada;estacao;variavel_FFT;data_compilacao;hora_compilacao;multDia;"+todoAlt+"\n"

    # Reading (try) or creating (except) a archive csv
    try:
        analyticFFT2 = open(directory[:-10] + "analytic" + "\\" + "bd_FFT" + ".csv", "r")
        lista = analyticFFT2.readlines()
        analyticFFT2.close()
        print("\n /!\ A planilha \n\t", analyticFFT2, "\n foi LIDA com sucesso!\n")
    except:
        analyticFFT2 = open(directory[:-10] + "analytic" + "\\" + "bd_FFT" + ".csv", "w")
        analyticFFT2.writelines(cabecalho)
        analyticFFT2.close()
        analyticFFT2 = open(directory[:-10] + "analytic" + "\\" + "bd_FFT" + ".csv", "r")
        lista = analyticFFT2.readlines()
        analyticFFT2.close()
        print("\n /!\ A nova planilha \n\t", analyticFFT2, "\n foi CRIADA!\n")

    # Preparing the data that will be salve
    now = datetime.now()# Time of save
    now_esp = now.strftime("%Y/%m/%d;%H:%M:%S")
    variavel = variavel_FFT

    dadosColetado = [] # Putting the data of station in all days of the year
    cont = multDia = 0
    for i in range(len(intervalo)):
        #print(i, cont, 'intervalo', intervalo[i], 'dayFFT', day_FFT[cont], 'value', np.round(value_FFT[cont], 3))
        while cont > 1 and \
            day_FFT[cont] == day_FFT[cont-1]: # Check if had more of one measurement in a day/hour
            multDia += 1
            cont += 1
        if intervalo[i] == day_FFT[cont]:
            dadosColetado.insert(i, value_FFT[cont]) # If in day HAVE info
            if cont < len(day_FFT)-1:
                cont += 1
        else:
            dadosColetado.insert(i, []) # If in day DO NOT HAVE info

    dadosColetado = str.replace(str(dadosColetado), ",", ";") # Formating string
    dadosColetado = str.replace(dadosColetado, "[", "")
    dadosColetado = str.replace(dadosColetado, "]", "")
    dadosColetado = str.replace(dadosColetado, " ", "")

    indicador_ini = aer_cam_ano_est[0:4]+";"+str(year)+";"+cam+";"+ESTACAO+";"+variavel+";"+now_esp+";"+str(multDia)
    salvar = indicador_ini+";"+dadosColetado+"\n"

    # Delete old info and update data
    existe=False
    for i in range(len(lista)): # This program locate the aerodrome in archive
        # print(variavel, indicador_ini[:21], '\n', lista[i][:21])
        # print('\n\t', indicador_ini[:41], '\n\t', lista[i][:41])
        if indicador_ini[:21] == lista[i][:21] and indicador_ini[:41] != lista[i][:41]:
            lista.pop(i)
            lista.append(salvar)
            lista.sort()
            existe = True
            print(" /!\ Os valores da \n\t", aer_cam_ano_est+variavel, "\n foram ATUALIZADAS no arquivo!\n")
            break  # The aerodrome will be recorded only one time

    # If the archive do not have of aerodrome
    try:
        if existe == False and lista[0] == cabecalho:
            lista.append(salvar)
            lista.sort()
            print(" /!\ Os valores da \n\t", aer_cam_ano_est+variavel, "\n foram ADICIONADAS no arquivo!\n")
    except:# If the archive is empty
        if lista == []:
            lista.append(cabecalho)
            lista.append(salvar)
            lista.sort()
            print(" /?\ O arquivo estava vazio e uma lista foi CRIADA!\n")

    # Recording new list in csv
    analyticFFT = open(directory[:-10] + "analytic" + "\\" + "bd_FFT" + ".csv", "w")
    analyticFFT.writelines(lista)
    analyticFFT.close()
    print(" /!\ A planilha: \n\t", analyticFFT, "\n acaba de ser FECHADA!\n")

    return None

def salvaEst(R_pearson, p_value, slope, intercept, n, std_err):
    """"
    Function for saving statistics about K and P energy
    """

    # Saving data of variable in (bd_Est.csv)
    # Creating the head of table
    cabecalho = "aerodromo;ano;camada;estacao;data_compilacao;hora_compilacao;R_pearson;p_value;slope;intercept;n;std_err\n"

    # Reading (try) or creating (except) a archive csv
    try:
        analyticEst2 = open(directory[:-10] + "analytic" + "\\" + "bd_Est" + ".csv", "r")
        lista = analyticEst2.readlines()
        analyticEst2.close()
        print("\n /!\ A planilha \n\t", analyticEst2, "\n foi LIDA com sucesso!\n")
    except:
        analyticEst2 = open(directory[:-10] + "analytic" + "\\" + "bd_Est" + ".csv", "w")
        analyticEst2.writelines(cabecalho)
        analyticEst2.close()
        analyticEst2 = open(directory[:-10] + "analytic" + "\\" + "bd_Est" + ".csv", "r")
        lista = analyticEst2.readlines()
        analyticEst2.close()
        print("\n /!\ A nova planilha \n\t", analyticEst2, "\n foi CRIADA!\n")

    # Preparing the data that will be salve
    now = datetime.now()  # Time of save
    now_esp = now.strftime("%Y/%m/%d;%H:%M:%S")

    indicador_ini = aer_cam_ano_est[0:4] + ";" + str(year) + ";" + cam + ";" + ESTACAO + ";" +now_esp
    salvar = indicador_ini + ";" + str(R_pearson) + ";" + str(p_value) + ";" + \
             str(slope) + ";" + str(intercept) + ";" + str(n) + ";" + str(std_err) +"\n"

    # Delete old info and update data
    existe = False
    for i in range(len(lista)): # This program locate the aerodrome in archive
        if indicador_ini[:14] == lista[i][:14] and indicador_ini[:34] != lista[i][:34]:
            lista.pop(i)
            lista.append(salvar)
            lista.sort()
            existe = True
            print(" /!\ Os valores da \n\t", aer_cam_ano_est + "Est", "\n foram ATUALIZADAS no arquivo!\n")
            break  # The aerodrome will be recorded only one time

    # If the archive do not have of aerodrome
    try:
        if existe == False and lista[0] == cabecalho:
            lista.append(salvar)
            lista.sort()
            print(" /!\ Os valores da \n\t", aer_cam_ano_est + "Est", "\n foram ADICIONADAS no arquivo!\n")
    except: # If the archive is empty
        if lista == []:
            lista.append(cabecalho)
            lista.append(salvar)
            lista.sort()
            print(" /?\ O arquivo estava vazio e uma lista foi CRIADA!\n")

    # Recording new list in csv
    analyticEst = open(directory[:-10] + "analytic" + "\\" + "bd_Est" + ".csv", "w")
    analyticEst.writelines(lista)
    analyticEst.close()
    print(" /!\ A planilha: \n\t", analyticEst, "\n acaba de ser FECHADA!\n")

    return None

def salvaEne(energia, tipo, dia, tendencia, beta0, beta1):
    """
    Function that salve K, P, and M energy
    """

    # Saving data of variable in (bd_Ene.csv)
    # Creating the head of table
    intervalo = [0.5 * x for x in range((366 + 2) * 2-1)]#Dias de coleta
    todoDia = str(intervalo)
    todoDia = str.replace(todoDia, ",", ";")
    todoDia = str.replace(todoDia, "[", "")
    todoDia = str.replace(todoDia, "]", "")
    todoDia = str.replace(todoDia, " ", "")
    cabecalho = "aerodromo;ano;camada;estacao;tipo;data_compilacao;hora_compilacao;num_dados;erro_dados;MK_slope;MK_p;lin_slope;lin_intercept;"\
                +todoDia+"\n"

    # Reading (try) or creating (except) a archive csv
    try:
        analyticEne2 = open(directory[:-10] + "analytic" + "\\" + "bd_Ene" + ".csv", "r")
        lista = analyticEne2.readlines()
        analyticEne2.close()
        print (" /!\ A planilha \n\t", analyticEne2, " foi LIDA com sucesso!\n")
    except:
        analyticEne2 = open(directory[:-10] + "analytic" + "\\" + "bd_Ene" + ".csv", "w")
        analyticEne2.writelines(cabecalho)
        analyticEne2.close()
        analyticEne2 = open(directory[:-10] + "analytic" + "\\" + "bd_Ene" + ".csv", "r")
        lista = analyticEne2.readlines()
        analyticEne2.close()
        print(" /!\ A nova planilha \n\t", analyticEne2, " foi CRIADA!\n")

    # Preparing the data that will be salve
    now = datetime.now() # Time of save
    now_esp = now.strftime("%Y/%m/%d;%H:%M:%S")

    num_dados = str(len(energia)) # Amount of collected data

    dadosColetado = [] # Putting the data of station in all days of the year
    cont = erroDia = 0
    for i in range(len(intervalo)):
        # print (intervalo[i] == dia[cont], "i= ", i,
        #        "intervalo[i]= ", intervalo[i],
        #        "cont= ", cont,
        #        "dia[cont]= ", dia[cont])
        if np.array(intervalo[i]) == dia[cont]:
            dadosColetado.insert(i, energia[cont]) # If in day HAVE info
            if cont < (len(dia)-1):
                cont += 1
            while dia[cont] == dia[cont-1]:
                print(" Foi encontrado multiplicidade de dados no dia[cont]= ", dia[cont], tipo)
                cont += 1
                erroDia += 1 # Count days with multiplicity
                print(" Este arquivo contem ", erroDia, " dia(s) com multiplicidade")
        else:
            dadosColetado.insert(i,[]) # If in day DO NOT HAVE info

    erroDia = str(erroDia)
    dadosColetado = str.replace(str(dadosColetado), ",", ";") # Formating string
    dadosColetado = str.replace(dadosColetado, "[", "")
    dadosColetado = str.replace(dadosColetado, "]", "")
    dadosColetado = str.replace(dadosColetado, " ", "")

    indicador_ini = aer_cam_ano_est[0:4]+";"+str(year)+";"+cam+";"+ESTACAO+";"+tipo

    estatisticas = tendencia.trend+";"+str(tendencia.p)+";"+str(beta1)+";"+str(beta0)
    estatisticas = str.replace(estatisticas, "[", "")
    estatisticas = str.replace(estatisticas, "]", "")
    estatisticas = str.replace(estatisticas, " ", "")

    # "estacao;ano;camada;tipo;data_compilacao;hora_compilacao;num_dados;erro_dados;MK_slope;MK_p;lin_slope;lin_intercept;" \
    # + todoDia + "\n"
    salvar = indicador_ini+";"+now_esp+";"+num_dados+";"+erroDia+";"+estatisticas+";"\
             +dadosColetado+"\n"

    # Delete old info and update data
    existe=False
    for i in range(len(lista)):#Esse programa localiza a estacao no arquivo
        # print("Iteracao: ", i, " de ", len(lista))
        # print(" Programa", indicador_ini, "\nArquivo", lista[i][0:16])
        # print(" Programa2", salvar[:37], "\nArquivo2", lista[i][:37])
        if indicador_ini == lista[i][0:16] and salvar[:37] != lista[i][:37]:
            lista.pop(i)
            lista.append(salvar)
            lista.sort()
            existe = True
            print(" /!\ Os valores de energia da \n\t", aer_cam_ano_est+tipo, "\n foram ATUALIZADAS!\n")
            break #Permite que a estacao seja gravada somente uma vez

    # If the archive do not have of aerodrome
    try:
        if existe == False and lista[0] == cabecalho:
            lista.append(salvar)
            lista.sort()
            print(" /!\ Dados da \n\t", aer_cam_ano_est+tipo, "\n foram ADICIONADAS no arquivo!\n")
    except: # If the archive is empty
        if lista == []:
            lista.append(cabecalho)
            lista.append(salvar)
            lista.sort()
            print(" /?\ O arquivo estava vazio e uma lista foi CRIADA!\n")

    # Recording new list in csv
    analyticEne = open(directory[:-10] + "analytic" + "\\" + "bd_Ene" + ".csv", "w")
    analyticEne.writelines(lista)
    analyticEne.close()
    print(" /!\A planilha: \n\t", analyticEne, " acaba de ser FECHADA!\n")

    return None

def salvaBrunt(variavel_N, value_N, day_N):
    """
    Function that salve the N of Brunt-Vaisala frequency estimated

    The variables are 'N_mean' and 'N_sd' for mean and standard deviation
    """

    # Saving data of variable in (bd_Brunt.csv)
    # Creating the head of table
    intervalo = [0.5 * x for x in range((366 + 2) * 2-1)]#Dias de coleta
    todoAlt = str(intervalo)
    todoAlt = str.replace(todoAlt, ",", ";")
    todoAlt = str.replace(todoAlt, "[", "")
    todoAlt = str.replace(todoAlt, "]", "")
    todoAlt = str.replace(todoAlt, " ", "")
    cabecalho = "aerodromo;ano;camada;estacao;variavel_N;data_compilacao;hora_compilacao;multDia;"+todoAlt+"\n"

    # Reading (try) or creating (except) a archive csv
    try:
        analyticBrunt = open(directory[:-10] + "analytic" + "\\" + "bd_Brunt" + ".csv", "r")
        lista = analyticBrunt.readlines()
        analyticBrunt.close()
        print("\n /!\ A planilha \n\t", analyticBrunt, "\n foi LIDA com sucesso!\n")
    except:
        analyticBrunt = open(directory[:-10] + "analytic" + "\\" + "bd_Brunt" + ".csv", "w")
        analyticBrunt.writelines(cabecalho)
        analyticBrunt.close()
        analyticBrunt = open(directory[:-10] + "analytic" + "\\" + "bd_Brunt" + ".csv", "r")
        lista = analyticBrunt.readlines()
        analyticBrunt.close()
        print("\n /!\ A nova planilha \n\t", analyticBrunt, "\n foi CRIADA!\n")

    # Preparing the data that will be salve
    now = datetime.now() # Time of save
    now_esp = now.strftime("%Y/%m/%d;%H:%M:%S")
    variavel = variavel_N

    dadosColetado = [] # Putting the data of station in all days of the year
    cont = multDia = 0
    for i in range(len(intervalo)):
        #print(i, cont, 'intervalo', intervalo[i], 'dayN', day_N[cont], 'value', np.round(value_N[cont], 3))
        while cont > 1 and \
            day_N[cont] == day_N[cont-1]: # Check if had more of one measurement in a day/hour
            multDia += 1
            cont += 1
        if intervalo[i] == day_N[cont]:
            dadosColetado.insert(i, value_N[cont]) # If in day HAVE info
            if cont < len(day_N)-1:
                cont += 1
        else:
            dadosColetado.insert(i, []) # If in day DO NOT HAVE info

    dadosColetado = str.replace(str(dadosColetado), ",", ";") # Formating string
    dadosColetado = str.replace(dadosColetado, "[", "")
    dadosColetado = str.replace(dadosColetado, "]", "")
    dadosColetado = str.replace(dadosColetado, " ", "")

    indicador_ini = aer_cam_ano_est[0:4]+";"+str(year)+";"+cam+";"+ESTACAO+";"+variavel+";"+now_esp+";"+str(multDia)
    salvar = indicador_ini+";"+dadosColetado+"\n"

    # Delete old info and update data
    existe=False
    for i in range(len(lista)): # This program locate the aerodrome in archive
        # print(variavel, indicador_ini[:21], '\n', lista[i][:21])
        # print('\n\t', indicador_ini[:41], '\n\t', lista[i][:41])
        if indicador_ini[:21] == lista[i][:21] and indicador_ini[:41] != lista[i][:41]:
            lista.pop(i)
            lista.append(salvar)
            lista.sort()
            existe = True
            print(" /!\ Os valores da \n\t", aer_cam_ano_est+variavel, "\n foram ATUALIZADAS no arquivo!\n")
            break # The aerodrome will be recorded only one time

    # If the archive do not have of aerodrome
    try:
        if existe == False and lista[0] == cabecalho:
            lista.append(salvar)
            lista.sort()
            print(" /!\ Os valores da \n\t", aer_cam_ano_est+variavel, "\n foram ADICIONADAS no arquivo!\n")
    except:# If the archive is empty
        if lista == []:
            lista.append(cabecalho)
            lista.append(salvar)
            lista.sort()
            print(" /?\ O arquivo estava vazio e uma lista foi CRIADA!\n")

    # Recording new list in csv
    analyticBrunt = open(directory[:-10] + "analytic" + "\\" + "bd_Brunt" + ".csv", "w")
    analyticBrunt.writelines(lista)
    analyticBrunt.close()
    print(" /!\ A planilha: \n\t", analyticBrunt, "\n acaba de ser FECHADA!\n")

    return None

# Physical constants
Rcte = 8.314463 # Ideal gas universal constant in SI
gamma_i = 1.400 # Typical air adiabatic index
M_air = 28.9647E-03 # Molar mass of air
grav = 9.80665 # Gravitacional cceleration m/s^2

# Number of aerodromes
# ATENTION: modify this value according to your aerodrome list
n_station = 33

# Time data: specify here the maximum number of data in one year (reccomended upper value 400,000)
# n_time = 400000
n_time = 1100000

# Read station list
arr = []
file = open('station_list.txt', 'r')
# Read line into array
for lines in file.readlines():
    # Ddd a new sublist
    #arr.append([])
    # Loop over the elements, splited by whitespace
    stat = lines.split()[0]
    arr.append(stat)
    for i in lines.split()[1:]:
        # Convert to integer and append to the last
        # Element of the list
        arr.append(float(i))
# print('arr=',arr)

# Variables to store station information: code, latitude, longitude and altitude
stat_list = np.chararray(n_station, itemsize=4)
lat_list = np.zeros(n_station, dtype='float64')
lon_list = np.zeros(n_station, dtype='float64')
alt_list = np.zeros(n_station, dtype='float64')
for i in range(0, n_station):
    stat_list[i] = arr[4*i]
    lat_list[i] = arr[4*i+1]
    lon_list[i] = arr[4*i+2]
    alt_list[i] = arr[4*i+3]
file.close

# Custom type for data handling and storing
tp_temp = np.dtype([('stat', 'U4'), ('lon', 'f8'), ('lat', 'f8'), ('alt', 'f8'),\
                         ('mat', 'f8', (n_time, 12))])
data_ptemp = np.zeros(n_station,dtype=tp_temp)
tp_wind = np.dtype([('stat', 'U4'), ('lon', 'f8'), ('lat', 'f8'), ('alt', 'f8'),\
                         ('mat', 'f8', (n_time, 11))])
data_pwind = np.zeros(n_station, dtype=tp_wind)

###################################################################################
#
# Execute all aerodromes/years/layer/station
#
##################################################################################

listAerodrome2014 = ['sbat', 'sbbr', 'sbbv', 'sbcf', 'sbcg',
                     'sbcr', 'sbct', 'sbcy', 'sbcz', 'sbfi',
                     'sbfl', 'sbfn', 'sbgl', 'sblo', 'sbmn',
                     'sbmq', 'sbmt', 'sbmy', 'sbnt', 'sbpa',
                     'sbpv', 'sbrb', 'sbsl', 'sbsm', 'sbsn',
                     'sbts', 'sbtt', 'sbua', 'sbug', 'sbul',
                     'sbvh', 'sbvt']

stepLSM = 10 # Steps of Least Squared Minimum

for AERODROME in listAerodrome2014:
    YEAR = '2014' # If necessary, change!
    for CAM in ['T', 'S']: # T = troposphere and S = Lower Stratosphere
        for ESTACAO in ['al']: # or ['al', 'fw', 'ss']:
            ###################################################################################
            #
            # Read parameters from input file:
            # The relevant input parameters parameters are read from the input file
            # (e.g. input.dat) where the year and the atmosphere height interval are defined
            #
            ##################################################################################

            # Changing of variable to localize the archives .dat
            if CAM == "T":
                CAM = "trop"
            if CAM == "S":
                CAM = "strat"

            # Checking the directory of archive of input_path and open
            directory = os.getcwd()
            input_path = directory + '\\' + str(sys.argv[2]) + '\\'#<<<<# directory + '\\' +
            filenameS = sorted(os.listdir(input_path)) # List all the files inside the input_path
            l_fileS = len(filenameS)#;print(l_fileS)

            i = 0
            while (i < l_fileS):
                 filename = filenameS[i]#;print(ii,filename)
                 nome = AERODROME + "_" + CAM + "_" + YEAR + ".dat"
                 if filename == nome:
                     input_path = input_path+nome
                     break
                 i = i+1

            #Checking if the archive is in /input_samples/
            try:
                file = open(input_path, 'r')
            except:
                print('\n /!\ The archive was not find in /input_samples/! :( ')
                #exit()
                break

            print('\n The input file is \n\t' + (input_path))
            lines = file.readlines()
            # Year of the data set that should be analysed
            year = int(lines[0].strip())
            print('\nyear   : ', year)
            # Program  only samples data between h_min and h_max
            h_min = float(lines[1].strip()) # for troposphere, h_min depends on the station (ssbr: 1100). For low stratosphere, use h_min=18000.
            print('h_min  : ', h_min)
            h_max = float(lines[2].strip()) # for troposphere, h_max = 11000; for low stratosphere, h_max = 25000
            print('h_max  : ', h_max)
            h_lim = float(lines[3].strip()) # data for which h(max) < h_lim should not be analysed. For troposphere, h_lim = 10000; for low stratosphere, h_lim = 24000
            print('h_lim  : ', h_lim)
            h_min_fft = float(lines[4].strip())
            print('h_min  : ', h_min_fft)
            h_max_fft = float(lines[5].strip())
            print('h_min  : ', h_max_fft)
            cam = lines[6].strip()
            print('stratum: ', cam)
            file.close
            #____________________________ raw_input("Press [enter] to continue.")

            ###################################################################################
            #
            # Read data                                                                       #
            # Note: all the data files must be named ending with '_tempo.csv' or '_vento.csv' #
            # ATTENTION: only data files should be stored in the data directory
            #
            ###################################################################################
            directory = os.getcwd()
            rel_path = '\\' + str(sys.argv[1]) + '\\'
            print('\n The data files are stored in the directory: \n\t' + directory + rel_path)
            directory = directory + rel_path
            sufix_t = '_temp.csv'
            sufix_w = '_vento.csv'
            filenames = sorted(os.listdir(directory)) # List all the files inside the directory
            #print('\n The archives are: \n\t', filenames, '\n')#<<<<# Linha adicionada em 13/07/2020
            l_files = len(filenames)

            II = 0
            while II < l_files: # Finding rhe archive AERODROMO+sufix_t
                filename = filenames[II]
                if filename not in AERODROME+sufix_t:
                    II = II + 1
                if filename in AERODROME+sufix_t:
                    break
                if II >= l_files:
                    print(' /!\ No data was found in the specified directory!')
                    exit()

            #print('\n The archieve ', filenames[II], ' has the index', II, '\n')

            ii = II
            j = k = 0 #j finding info of aerodrome;ii finding of archive in the past;k the archives derired
            stat_index = []
            while (ii < II+2):
                 filename = filenames[ii]
                 #print(' The ii =', ii, ' has filename:', filename)#<<<<# Linha adicionada em 13/07/2020
                 aerodrome = str(stat_list[j])[2:6]
                 str_t = aerodrome + sufix_t
                 str_w = aerodrome + sufix_w
                 print(' Searching for data files: ' + str_t + ' and ' + str_w)

                 if str_t in filename:
                   k=1
                   ii = ii + 1
                   print('\n Getting TEMPERATURE data from aerodrome ' + aerodrome.upper()+':')
                 elif str_w in filename:
                   k=2
                   ii = ii + 1
                   print('\n Getting WIND data from aerodrome ' + aerodrome.upper()+':')
                 else:
                   if aerodrome not in filename:
                       #print('\t The aerodrome j =', j, 'is: ', aerodrome, ', and filename is: ', filename, '\n')
                       j = j + 1
                   if j >= n_station:#<<<<# Modificado em 11/02/2021
                       print (' /!\ No data was found in the specified directory!')
                       file.close
                       exit()
                   continue

                 # Saving the data spreadsheet in 'arr'
                 if (k==1 or k==2):
                     arr = []
                     file = open(directory + filename, 'r')
                     print('\t'+directory+filename)
                     # Read line into array
                     if k == 1:
                         n_col = 12
                     elif k == 2:
                         n_col = 11
                     for lines in file.readlines()[1:]:
                         #print(lines);exit()
                         # Add a new sublist
                         arr.append([])#;print(lines)
                         # Loop over the elemets, split by whitespace
                         for i in lines.split()[2:n_col+2]:
                             # print(' i= ',i)
                             # Convert to float and append to the last
                             # Element of the list
                             arr[-1].append(float(i))

                 # Shape of 'arr': a(row), b(column)
                 a, b = np.shape(arr)
                 print('\ta = ', a, ' b = ', b, '\n')

                 # Transfoming variable arr(list) into array
                 neoarr = np.zeros(shape=(n_time, b))
                 neoarr[0:a, 0:b] = np.asarray(arr)
                 #print(neoarr);exit()

                 if k == 1:
                     data_ptemp['stat'][j] = stat_list[j]
                     data_ptemp['lon'][j] = lon_list[j]
                     data_ptemp['lat'][j] = lat_list[j]
                     data_ptemp['alt'][j] = alt_list[j]
                     data_ptemp['mat'][j] = neoarr
                 elif k == 2:
                     data_pwind['stat'][j] = stat_list[j]
                     data_pwind['lon'][j] = lon_list[j]
                     data_pwind['lat'][j] = lat_list[j]
                     data_pwind['alt'][j] = alt_list[j]
                     data_pwind['mat'][j] = neoarr
                     stat_index.append(j)
                     j = 0
                     file.close
                     break

                 file.close

            jj = stat_index[0] # For now we should only analyse one station: the last station for which data was found in the data directory
            len_sbat = np.shape(data_pwind['mat'][jj])

            aer_cam_ano_est = AERODROME+cam+str(year)+ESTACAO # The exit is, for example : sbtaT2009

            # Search for the index list where daily measurement starts

            # Wind data case
            arr = []
            for i in range(0, n_time-1): # [...,9] is pression
                if (data_pwind['mat'][jj][i, 0] == year and \
                       data_pwind['mat'][jj][i+1, 9] - data_pwind['mat'][jj][i, 9] > -3000.0):
                    continue
                elif (data_pwind['mat'][jj][i, 0] == year and \
                      data_pwind['mat'][jj][i+1, 9] - data_pwind['mat'][jj][i, 9] < -3000.0):
                    arr.append(int(i))
            # Identify when (day and hour in spreadsheet) was colected
            a = np.shape(arr)
            arr_init_w = np.zeros(shape=a)
            arr_init_w = np.asarray(arr)
            #print(' arr_init_w: ',arr_init_w) # The number of each colect in a year
            len_init_w = arr_init_w.size # The amount of collect in a year
            print(' Number of WIND data points in '+YEAR+': \n\t' + str(len_init_w))

            # Temperature data case
            arr = []
            for i in range(0, n_time-1): # [...,10] is altitude
                if (data_ptemp['mat'][jj][i,0] == year and \
                       data_ptemp['mat'][jj][i+1,10] - data_ptemp['mat'][jj][i,10] > -3000.0):
                    continue
                elif (data_ptemp['mat'][jj][i,0] == year and \
                      data_ptemp['mat'][jj][i+1,10] - data_ptemp['mat'][jj][i,10] < -3000.0):
                    arr.append(int(i))
            # Identify when (day and hour in spreadsheet) was colected
            a = np.shape(arr)
            arr_init_t = np.zeros(shape=a)
            arr_init_t = np.asarray(arr)
            #;print(' arr_init_t: ',arr_init_t) # The number of each colect in a year
            len_init_t = arr_init_t.size # The amount of collect in a year
            print(' Number of TEMPERATURE data points in '+YEAR+': \n\t' + str(len_init_t), "\n")

            ###################################################################################
            #
            # Data Analysis:                                                                  #
            # Here each WIND and TEMPERATURE profile is FFT analysed. The wave kinetic energy #
            # is estimated from the square sum of FFT components.  After that, the optimum    #
            # wavelength is determined from a least square fitting.
            #
            ###################################################################################

            h_reg = np.arange(h_min, h_max, 50)            # horizontal grid for plotting purposes
            L_fft = 50                                     # FFT step
            h_fft = np.arange(h_min_fft, h_max_fft, L_fft) # horizontal grid for the domain of FFT analysis. For low stratosphere use 18000 and 24000. For troposphere, choose inteval based on h_min and h_lim
            thr_exp = 0.7                                  # if the main FFT squared component is higher than 70% of the total energy, then fit is plotted

            lamb_nw    = np.zeros(shape=(len_init_w,))# store optmized lambda value for northerly wind variation
            phase_nw   = np.zeros(shape=(len_init_w,))# store optmized phase value for northerly wind variation
            amp_nw     = np.zeros(shape=(len_init_w,))# store optmized amplitude value for northerly wind variation
            rel_err_wn = np.zeros(shape=(len_init_w,))# ratio between qui2 squared to northerly wind std
            wn_energy  = np.zeros(shape=(len_init_w,))# store kinetic energy associated with the northerly component of the wind
            a_nw    = np.zeros(shape=(len_init_w,))# store optmized 'a'x^2+bx+c
            b_nw    = np.zeros(shape=(len_init_w,))# store optmized ax^2+'b'x+c
            c_nw    = np.zeros(shape=(len_init_w,))# store optmized ax^2+bx+'c'

            lamb_ew    = np.zeros(shape=(len_init_w,)) # store optmized lambda value for easterly wind variation
            phase_ew   = np.zeros(shape=(len_init_w,)) # store optmized phase value for easterly wind variation
            amp_ew     = np.zeros(shape=(len_init_w,)) # store optmized amplitude value for easterly wind variation
            rel_err_ew = np.zeros(shape=(len_init_w,)) # ratio between qui2 squared to northerly wind std
            we_energy  = np.zeros(shape=(len_init_w,)) # store kinetic energy associated with the easterly component of the wind
            a_ew    = np.zeros(shape=(len_init_w,))# store optmized 'a'x^2+bx+c
            b_ew    = np.zeros(shape=(len_init_w,))# store optmized ax^2+'b'x+c
            c_ew    = np.zeros(shape=(len_init_w,))# store optmized ax^2+bx+'c'

            lamb_t          = np.zeros(shape=(len_init_t, ))# store optmized lambda value for temperature variation
            brunt_t_mean    = np.zeros(shape=(len_init_t,))# store mean of N (Brunt-Vaisala) frequency from temperature
            brunt_t_sd      = np.zeros(shape=(len_init_t,))  # store std of N (Brunt-Vaisala) frequency from temperature
            phase_t         = np.zeros(shape=(len_init_t, ))# store optmized phase value for temperature variation
            amp_t           = np.zeros(shape=(len_init_t, ))# store optmized amplitude value for temperature variation
            rel_err_t       = np.zeros(shape=(len_init_t, ))# ratio between qui2 squared to northerly wind std
            a_t          = np.zeros(shape=(len_init_t,))# store optmized 'a'x^2+bx+c
            b_t          = np.zeros(shape=(len_init_t,))# store optmized ax^2+'b'x+c
            c_t          = np.zeros(shape=(len_init_t,))# store optmized ax^2+bx+'c'

            temp_energy     = np.zeros(shape=(len_init_t, ))# store "kinetic energy" associated with temperature
            temp_pot_energy = np.zeros(shape=(len_init_t, ))# store potential energy associated with temperature

            # Northerly wind case <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
            summ = np.zeros(shape=(h_reg.size,))
            n_n = np.zeros(shape=(h_reg.size,))
            u_zeros = np.zeros(shape=(h_reg.size,))
            u_ones = np.ones(shape=(h_reg.size,))
            print('\t----------->Analysing NORTHERLY winds<-----------\n')
            for i in range(0, len_init_w-1):#9 is altitude, 8 is wind, 7 is direction of wind
                continue
                ###################################################
                #   Verification of day
                ###################################################
                month = data_pwind['mat'][jj][arr_init_w[i], 1]
                day   = data_pwind['mat'][jj][arr_init_w[i], 2]
                hour  = data_pwind['mat'][jj][arr_init_w[i], 3]
                days  = month_days(month)
                days  = days + day
                if i>0 and hour == data_pwind['mat'][jj][arr_init_w[i-1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[i-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if i>0 and i<len_init_w-1 and\
                        hour == data_pwind['mat'][jj][arr_init_w[i+1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[i+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    continue

                print(' NORTHERLY i =', i, 'day =', days, round(i/(len_init_w-2)*100, 2), "% t(min) =", round((time.time()-ini_prog)/60, 2),aer_cam_ano_est)
                h = data_pwind['mat'][jj][arr_init_w[i]:(arr_init_w[i+1]-1), 9]
                theta = np.pi*data_pwind['mat'][jj][arr_init_w[i]:(arr_init_w[i+1]-1), 7]/180
                w = np.multiply(np.cos(theta),data_pwind['mat'][jj][arr_init_w[i]:(arr_init_w[i+1]-1), 8])
                if len(h) < 2:
                    continue
                ###################################################
                #   Data Interpolation
                ###################################################
                f = interp1d(h, w, kind='linear', bounds_error=False, fill_value=0.0) #Interpolate missing data
                f_reg = f(h_reg)
                summ = summ + f_reg
                n_n = n_n + np.where(f_reg == 0, u_zeros, u_ones)
                h_str = h[np.where((h >= h_min) & (h <= h_max))]
                w_str = w[np.where((h >= h_min) & (h <= h_max))]
                if np.amax(h) < h_lim or w_str.all() == 0 or np.size(h_str) == 0 or h_str.size < 6:
                    continue
                #print(' NORTHERLY i = ', i, 'tam = ', h_str.size)
                ###################################################
                #   Parabolic fiting - background correction
                ###################################################
                popt1, pcov1 = curve_fit(func1, h_str, w_str, absolute_sigma=False, maxfev=800)
                w_fit_1 = func1(h_str, *popt1)
                w_net = w_str - w_fit_1
                ###################################################
                #   FFT - Fast Fourier Transform - Power Spectrum
                ###################################################
                f_fft = interp1d(h_str, w_net, kind='linear', bounds_error=False, fill_value=0.0)
                w_fft = f_fft(h_fft)
                F_fft = np.fft.fft(w_fft)
                N_fft = w_fft.size
                AF_fft = np.abs(F_fft)[:N_fft // 2] * 2 / N_fft
                max_value = np.amax(AF_fft)
                i_max = np.where(AF_fft == max_value)
                fft_coef_im = F_fft[i_max]
                wn_energy[i] = 0.5 * np.sum(AF_fft**2)
                explan = max_value**2/wn_energy[i]
                ilamb = np.fft.fftfreq(N_fft, L_fft)[:N_fft // 2]#np.linspace(1000.0/h_max_fft, 1000.0/L_fft, N_fft)
                if ilamb[i_max] != 0:
                    lamb_max_fft = 1 / ilamb[i_max]
                else:
                    lamb_max_fft = max(h_fft) - min(h_fft)
                ratio_im = np.imag(fft_coef_im)/np.real(fft_coef_im)
                phase_max = np.arctan(ratio_im)*180/np.pi
                #########################################################
                #   LSF - Least Squares Fitting - Find optimal wavelength
                #########################################################
                qui2old = 0
                l_lower = 500
                l_upper = 12000
                lamb_arr = np.arange(l_lower, l_upper+10, stepLSM)
                #lamb_arr = np.arange(l_lower, l_upper + 10, 100)
                qui2      = np.zeros(shape=(len(lamb_arr),))
                phase_val = np.zeros(shape=(len(lamb_arr),))
                amp_val   = np.zeros(shape=(len(lamb_arr),))
                a_val = np.zeros(shape=(len(lamb_arr),))
                b_val = np.zeros(shape=(len(lamb_arr),))
                c_val = np.zeros(shape=(len(lamb_arr),))
                for j in range(0, len(lamb_arr)):
                    popt, pcov = curve_fit(lambda x, a, b, c, A, B: func(x, a, b, c, A, B, lamb_arr[j]), h_str, w_str)
                    w_fit_wave = func(h_str, popt[0], popt[1], popt[2], popt[3], popt[4], lamb_arr[j])
                    qui2[j] = np.sum((w_fit_wave - w_str)**2)
                    phase_val[j] = np.arctan2(-popt[4], popt[3]) * 180 / np.pi
                    amp_val[j]   = np.sqrt(popt[3] ** 2 + popt[4] ** 2)
                    a_val[j] = popt[0]
                    b_val[j] = popt[1]
                    c_val[j] = popt[2]
                index_min = np.where(qui2 == np.amin(qui2))[0][0]
                lamb_max = lamb_arr[index_min]
                phase_nw[i] = phase_val[index_min]
                amp_nw[i]   = amp_val[index_min]
                lamb_nw[i]  = lamb_max
                a_nw[i] = a_val[index_min]
                b_nw[i] = b_val[index_min]
                c_nw[i] = c_val[index_min]
                ngl = w_str.size - 5
                rel_err_wn[i] = np.sqrt(qui2[index_min]/ngl)/np.std(w_str)

                popt, pcov = curve_fit(lambda x, a, b, c, A, B: func(x, a, b, c, A, B, lamb_max), h_str, w_str)
                w_fit_total = func(h_str, popt[0], popt[1], popt[2], popt[3], popt[4], lamb_max)
                w_parabolic = func1(h_str, popt[0], popt[1], popt[2])
                w_net_fit = w_str - w_parabolic
                w_fit_sin = func2(h_str, popt[3], popt[4], lamb_max)

                #########################################################
                #   Refine FFT - Fast Fourier Transform - Power Spectrum
                #########################################################
                f_fft = interp1d(h_str, w_net_fit, kind='linear', bounds_error=False, fill_value=0.0)
                w_fft = f_fft(h_fft)
                F_fft = np.fft.fft(w_fft)
                N_fft = w_fft.size
                AF_fft = np.abs(F_fft)[:N_fft // 2] * 2 / N_fft
                max_value = np.amax(AF_fft)
                i_max = np.where(AF_fft == max_value)
                fft_coef_im = F_fft[i_max]
                if 0.5 * np.sum(AF_fft ** 2) < 0.125 * (np.amax(w_net) - np.amin(w_net)) ** 2 and lamb_max != l_upper:
                    wn_energy[i] = 0.5 * np.sum(AF_fft ** 2)
                    explan = max_value ** 2 / wn_energy[i]
                    ilamb = np.fft.fftfreq(N_fft, L_fft)[:N_fft // 2]
                    ratio_im = np.imag(fft_coef_im) / np.real(fft_coef_im)
                    phase_max = np.arctan(ratio_im) * 180 / np.pi
                    if ilamb[i_max] != 0:
                        lamb_max_fft = 1 / ilamb[i_max]
                    else:
                        lamb_max_fft = max(h_fft) - min(h_fft)
                else:# In this case, call the old fitting
                    qui2_old, phase_val_old, amp_val_old = call_old_fitting(lamb_arr, h_str, w_net)
                    index_min = np.where(qui2_old == np.amin(qui2_old))
                    lamb_max = lamb_arr[index_min]
                    phase_nw[i] = phase_val_old[index_min]
                    amp_nw[i] = amp_val_old[index_min]
                    lamb_nw[i] = lamb_max
                    a_nw[i] = popt1[0]
                    b_nw[i] = popt1[1]
                    c_nw[i] = popt1[2]
                    ngl = w_str.size - 2
                    rel_err_wn[i] = np.sqrt(qui2_old[index_min] / ngl) / np.std(w_str)
                    popt, pcov = curve_fit(lambda x, A, B: func2(x, A, B, lamb_max), h_str, w_net)
                    w_fit_sin = func2(h_str, popt[0], popt[1], lamb_max)
                    w_parabolic = func1(h_str, popt1[0], popt1[1], popt1[2])
                    w_fit_total = w_parabolic + w_fit_sin
                    w_net_fit = w_net

                # Plot the most 'monochromatic like' waves
                # if explan >= thr_exp:
                #     # Plot wave and fitted curve
                #     fig, axs = plt.subplots(2)
                #     fig.set_size_inches(6, 8)
                #     axs[0].set(ylabel="Northerly Wind variation (m/s)")
                #     axs[0].set(xlabel="Altitude (m)")
                #     axs[0].plot(h_str, w_str, '-k')
                #     axs[0].plot(h_str, w_parabolic, '--k')
                #     axs[0].plot(h_str, w_fit_sin, '--b')
                #     axs[0].plot(h_str, w_fit_total)
                #     # Plot FFT spectrum
                #     axs[1].set(ylabel="Amplitude")
                #     axs[1].set(xlabel="Wave-number [1/m]")
                #     axs[1].bar(ilamb, AF_fft, width=1E-4)  # 1 / N is a normalization factor
                #     print('Iteration number: ', i, 'Lambda (m): ', lamb_max, 'Lambda FFT', lamb_max_fft, 'Energy ', wn_energy[i])
                #     print('Month, Day, Hour:', data_pwind['mat'][jj][arr_init_w[i], 1],
                #             data_pwind['mat'][jj][arr_init_w[i], 2],
                #             data_pwind['mat'][jj][arr_init_w[i], 3])
                #     plt.show()
                #     #____________________________
                #     #input("Press [enter] to continue.")
                #     plt.close(fig)

            # Plot anual average and save
            fig1 = plt.figure(1)
            mean = np.divide(summ, n_n)
            plt.plot(h_reg, mean, '-')
            plt.plot(h_reg, mean*0, '--', color="black")
            plt.title('Northerly Wind '+aer_cam_ano_est)
            plt.xlabel('altitude (m)')
            plt.ylabel('m/s')
            local = str('analytic' + '\\' + aer_cam_ano_est + 'Nwind.png')
            plt.savefig(local, format='png', dpi=300)
            fig1.show()
            #____________________________
            #input("Press [enter] to continue.")
            plt.close(fig1)
            # salvaVariavel("No", mean, h_reg)

            # Easterly wind case <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
            summ = np.zeros(shape=(h_reg.size,))
            n_n = np.zeros(shape=(h_reg.size,))
            u_zeros = np.zeros(shape=(h_reg.size,))
            u_ones = np.ones(shape=(h_reg.size,))
            print('\t----------->Analysing EASTERLY winds<-----------\n')
            for i in range(0, len_init_w-1):
                continue
                ###################################################
                #   Verification of day
                ###################################################
                month = data_pwind['mat'][jj][arr_init_w[i], 1]
                day   = data_pwind['mat'][jj][arr_init_w[i], 2]
                hour  = data_pwind['mat'][jj][arr_init_w[i], 3]
                days  = month_days(month)
                days  = days + day
                if i>0 and hour == data_pwind['mat'][jj][arr_init_w[i-1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[i-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if i>0 and i<len_init_w-1 and\
                        hour == data_pwind['mat'][jj][arr_init_w[i+1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[i+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    continue

                print(' EASTERLY i =', i, 'day =', days, round(i / (len_init_w - 2) * 100, 2), "% t(min) =",
                      round((time.time() - ini_prog)/60, 2), aer_cam_ano_est)
                h = data_pwind['mat'][jj][arr_init_w[i]:(arr_init_w[i+1]-1), 9]
                theta = np.pi*data_pwind['mat'][jj][arr_init_w[i]:(arr_init_w[i+1]-1), 7]/180
                w = np.multiply(np.sin(theta), data_pwind['mat'][jj][arr_init_w[i]:(arr_init_w[i+1]-1), 8])# np.sin
                if len(h) < 2:
                    continue
                ###################################################
                #   Data Interpolation
                ###################################################
                f = interp1d(h, w, kind='linear', bounds_error=False, fill_value=0.0)
                f_reg = f(h_reg)
                summ = summ + f_reg
                n_n = n_n + np.where(f_reg == 0, u_zeros, u_ones)
                h_str = h[np.where((h >= h_min) & (h <= h_max))]
                w_str = w[np.where((h >= h_min) & (h <= h_max))]
                if np.amax(h) < h_lim or w_str.all() == 0 or np.size(h_str) == 0 or h_str.size < 6:
                    continue
                #print(' EASTHERLY i = ', i, 'tam = ', h_str.size)
                ###################################################
                #   Parabolic fiting - background correction
                ###################################################
                popt1, pcov1 = curve_fit(func1, h_str, w_str, absolute_sigma=False, maxfev=800)
                w_fit_1 = func1(h_str, *popt1)
                w_net = w_str - w_fit_1
                ##################################################
                #   FFT - Fast Fourier Transform - Power Spectrum
                ##################################################
                f_fft = interp1d(h_str, w_net, kind='linear', bounds_error=False, fill_value=0.0)
                w_fft = f_fft(h_fft)
                F_fft = np.fft.fft(w_fft)
                N_fft = w_fft.size
                AF_fft = np.abs(F_fft)[:N_fft // 2] * 2 / N_fft
                max_value = np.amax(AF_fft)
                i_max = np.where(AF_fft == max_value)
                we_energy[i] = 0.5 * np.sum(AF_fft**2)#we_energy
                explan = max_value**2/we_energy[i]
                fft_coef_im = F_fft[i_max]
                ilamb = np.fft.fftfreq(N_fft, L_fft)[:N_fft // 2]
                if ilamb[i_max] != 0:
                    lamb_max_fft = 1 / ilamb[i_max]
                else:
                    lamb_max_fft = max(h_fft) - min(h_fft)
                ratio_im = np.imag(fft_coef_im)/np.real(fft_coef_im)
                phase_max = np.arctan(ratio_im) * 180 / np.pi
                #########################################################
                #   LSF - Least Squares Fitting - Find optimal wavelength
                #########################################################
                qui2old = 0
                l_lower = 500
                l_upper = 12000
                lamb_arr = np.arange(l_lower, l_upper+10, stepLSM)
                #lamb_arr = np.arange(l_lower, l_upper + 10, 100)
                qui2 = np.zeros(shape=(len(lamb_arr),))
                phase_val = np.zeros(shape=(len(lamb_arr),))
                amp_val   = np.zeros(shape=(len(lamb_arr),))
                a_val = np.zeros(shape=(len(lamb_arr),))
                b_val = np.zeros(shape=(len(lamb_arr),))
                c_val = np.zeros(shape=(len(lamb_arr),))
                for j in range(0, len(lamb_arr)):
                    popt, pcov = curve_fit(lambda x, a, b, c, A, B: func(x, a, b, c, A, B, lamb_arr[j]), h_str, w_str)
                    w_fit_wave = func(h_str, popt[0], popt[1], popt[2], popt[3], popt[4], lamb_arr[j])
                    qui2[j] = np.sum((w_fit_wave - w_str) ** 2)
                    phase_val[j] = np.arctan2(-popt[4], popt[3]) * 180 / np.pi
                    amp_val[j]   = np.sqrt(popt[3] ** 2 + popt[4] ** 2)
                    a_val[j] = popt[0]
                    b_val[j] = popt[1]
                    c_val[j] = popt[2]
                index_min = np.where(qui2 == np.amin(qui2))[0][0]
                lamb_max = lamb_arr[index_min]
                amp_ew[i] = amp_val[index_min]#amp_ew
                phase_ew[i] = phase_val[index_min]#phase_ew
                lamb_ew[i] = lamb_max#lamb_ew
                a_ew[i] = a_val[index_min]
                b_ew[i] = b_val[index_min]
                c_ew[i] = c_val[index_min]
                ngl = w_str.size - 5
                rel_err_ew[i] = np.sqrt(qui2[index_min] / ngl) / np.std(w_str)#rel_err_ew

                popt, pcov = curve_fit(lambda x, a, b, c, A, B: func(x, a, b, c, A, B, lamb_max), h_str, w_str)
                w_fit_total = func(h_str, popt[0], popt[1], popt[2], popt[3], popt[4], lamb_max)
                w_parabolic = func1(h_str, popt[0], popt[1], popt[2])
                w_net_fit = w_str - w_parabolic
                w_fit_sin = func2(h_str, popt[3], popt[4], lamb_max)

                #########################################################
                #   Refine FFT - Fast Fourier Transform - Power Spectrum
                #########################################################
                f_fft = interp1d(h_str, w_net_fit, kind='linear', bounds_error=False, fill_value=0.0)
                w_fft = f_fft(h_fft)
                F_fft = np.fft.fft(w_fft)
                N_fft = w_fft.size
                AF_fft = np.abs(F_fft)[:N_fft // 2] * 2 / N_fft
                max_value = np.amax(AF_fft)
                i_max = np.where(AF_fft == max_value)
                fft_coef_im = F_fft[i_max]
                if 0.5 * np.sum(AF_fft ** 2) < 0.125 * (np.amax(w_net) - np.amin(w_net)) ** 2 and lamb_max != l_upper:
                    we_energy[i] = 0.5 * np.sum(AF_fft ** 2)
                    explan = max_value ** 2 / we_energy[i]
                    ilamb = np.fft.fftfreq(N_fft, L_fft)[:N_fft // 2]
                    ratio_im = np.imag(fft_coef_im) / np.real(fft_coef_im)
                    phase_max = np.arctan(ratio_im) * 180 / np.pi
                    if ilamb[i_max] != 0:
                        lamb_max_fft = 1 / ilamb[i_max]
                    else:
                        lamb_max_fft = max(h_fft) - min(h_fft)
                else:  # In this case, call the old fitting
                    qui2_old, phase_val_old, amp_val_old = call_old_fitting(lamb_arr, h_str, w_net)
                    index_min = np.where(qui2_old == np.amin(qui2_old))
                    lamb_max = lamb_arr[index_min]
                    phase_ew[i] = phase_val_old[index_min]
                    amp_ew[i] = amp_val_old[index_min]
                    lamb_ew[i] = lamb_max
                    a_ew[i] = popt1[0]
                    b_ew[i] = popt1[1]
                    c_ew[i] = popt1[2]
                    ngl = w_str.size - 2
                    rel_err_ew[i] = np.sqrt(qui2_old[index_min] / ngl) / np.std(w_str)
                    popt, pcov = curve_fit(lambda x, A, B: func2(x, A, B, lamb_max), h_str, w_net)
                    w_fit_sin = func2(h_str, popt[0], popt[1], lamb_max)
                    w_parabolic = func1(h_str, popt1[0], popt1[1], popt1[2])
                    w_fit_total = w_parabolic + w_fit_sin
                    w_net_fit = w_net

                # # Plot the most 'monochromatic like' waves
                # if explan >= thr_exp:
                #     # Plot wave and fitted curve
                #     fig, axs = plt.subplots(2)
                #     fig.set_size_inches(6, 8)
                #     axs[0].set(ylabel="Easterly wind variation (m/s)") #Eastherly
                #     axs[0].set(xlabel="Altitude (m)")
                #     axs[0].plot(h_str, w_str, '-k')
                #     axs[0].plot(h_str, w_parabolic, '--k')
                #     axs[0].plot(h_str, w_fit_sin, '--b')
                #     axs[0].plot(h_str, w_fit_total)
                #     # Plot FFT spectrum
                #     axs[1].set(ylabel="Amplitude")
                #     axs[1].set(xlabel="Wave-number [1/m]")
                #     axs[1].bar(ilamb[:N_fft // 2], AF_fft, width=1E-4)# 1 / N is a normalization factor
                #     print('Iteration: ', i, 'Lambda (m): ', lamb_max, 'Lambda FFT', lamb_max_fft, 'Energy ', we_energy[i])
                #     plt.show()
                #     #____________________________
                #     # input("Press [enter] to continue.")
                #     plt.close(fig)

            # Plot anual average and save
            fig1 = plt.figure(1)
            mean = np.divide(summ, n_n)
            plt.plot(h_reg, mean, '-')
            plt.plot(h_reg, mean*0, '--', color="black")
            plt.title('Easterly Wind '+aer_cam_ano_est)
            plt.xlabel('altitude (m)')
            plt.ylabel('m/s')
            local = str('analytic' + '\\' + aer_cam_ano_est + 'Ewind.png')
            plt.savefig(local, format='png', dpi=300)
            fig1.show()
            #____________________________
            #input("Press [enter] to continue.")
            plt.close(fig1)
            # salvaVariavel("Ea", mean, h_reg)# Comentario temporario para algoritimo do N de Brunt-Vaisala

            # Temperature vs altitude analysis <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
            summ = np.zeros(shape=(h_reg.size,))
            n_n = np.zeros(shape=(h_reg.size,))
            u_zeros = np.zeros(shape=(h_reg.size,))
            u_ones = np.ones(shape=(h_reg.size,))
            day_N = np.zeros(shape=(len_init_t,))
            print('\t----------->Analysing TEMPERATURE<-----------\n')
            for i in range(0, len_init_t-1):
                ###################################################
                #   Verification of day
                ###################################################
                month = data_ptemp['mat'][jj][arr_init_t[i], 1]
                day   = data_ptemp['mat'][jj][arr_init_t[i], 2]
                hour  = data_ptemp['mat'][jj][arr_init_t[i], 3]
                days  = month_days(month)
                days  = days + day
                if i>0 and hour == data_ptemp['mat'][jj][arr_init_t[i-1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if i>0 and i<len_init_t-1 and\
                        hour == data_ptemp['mat'][jj][arr_init_t[i+1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    continue

                print(' TEMPERATURE i =', i, 'day =', days, round(i / (len_init_t - 2) * 100, 2), "% t(min) =",
                      round((time.time() - ini_prog)/60, 2), aer_cam_ano_est)
                h = data_ptemp['mat'][jj][arr_init_t[i]:(arr_init_t[i+1]-1), 10]
                w = data_ptemp['mat'][jj][arr_init_t[i]:(arr_init_t[i+1]-1), 7]
                if len(h) < 2:
                    continue
                ###################################################
                #   Data Interpolation
                ###################################################
                f = interp1d(h, w, kind='linear', bounds_error=False, fill_value=0.0)
                f_reg = f(h_reg)
                summ = summ + f_reg
                n_n = n_n + np.where(f_reg == 0, u_zeros, u_ones)
                h_str = h[np.where((h >= h_min) & (h <= h_max))]
                w_str = w[np.where((h >= h_min) & (h <= h_max))]
                if np.amax(h) < h_lim or w_str.all() == 0 or np.size(h_str) == 0 or h_str.size < 6:
                    continue
                #print(' TEMPERATURE i = ', i, 'tam = ', h_str.size)
                ###################################################
                #   Parabolic fiting - background correction
                ###################################################
                popt1, pcov1 = curve_fit(func1, h_str, w_str, absolute_sigma=False, maxfev=800)
                w_fit_1 = func1(h_str, *popt1)
                w_net = w_str - w_fit_1
                ###################################################
                #   FFT - Fast Fourier Transform - Power Spectrum
                ###################################################
                f_fft = interp1d(h_str, w_net, kind='linear', bounds_error=False, fill_value=0.0)
                w_fft = f_fft(h_fft)
                F_fft = np.fft.fft(w_fft)
                N_fft = w_fft.size
                AF_fft = np.abs(F_fft)[:N_fft // 2] * 2 / N_fft
                max_value = np.amax(AF_fft)
                i_max = np.where(AF_fft == max_value)
                fft_coef_im = F_fft[i_max]
                temp_energy[i] = 0.5 * np.sum(AF_fft**2)
                explan = max_value**2/temp_energy[i]
                ilamb = np.fft.fftfreq(N_fft, L_fft)[:N_fft // 2]
                if ilamb[i_max] != 0:
                    lamb_max_fft = 1 / ilamb[i_max]
                else:
                    lamb_max_fft = max(h_fft) - min(h_fft)
                ratio_im = np.imag(fft_coef_im) / np.real(fft_coef_im)
                phase_max = np.arctan(ratio_im)
                ###################################################
                #   FFT - Fast Fourier Transform - Potential energy
                ###################################################
                wp_str = 275.15*np.ones(shape=(w_str.size,))+w_str

                N_brunt = ((M_air*(gamma_i-1)/(gamma_i*Rcte)))**0.5 * np.reciprocal(np.sqrt(wp_str))
                brunt_t_mean[i] = np.mean(N_brunt)*grav#; print(np.mean(N_brunt))
                brunt_t_sd[i] = np.std(N_brunt)*grav
                day_N[i] = days

                print(aer_cam_ano_est)
                print('wp_str = ', np.reciprocal(np.sqrt(wp_str)), np.reciprocal(np.sqrt(wp_str)).size)
                print('N_brunt = ', N_brunt, '\t','N_brunt.size = ', N_brunt.size, 'h_str.size = ', h_str.size)
                print('days =',days,days.size)

                fig1 = plt.figure(1)
                plt.plot(N_brunt, h_str/1000)
                plt.xlabel('N (Brunt-Vaisälä)')
                plt.ylabel('km')
                fig1.show()
                # ____________________________
                input("Press [enter] to continue.")
                plt.close(fig1)

                exit()
                wp_pot = np.divide(np.divide(w_net, wp_str), N_brunt)
                #print 'wp_pot', wp_pot
                fp_fft = interp1d(h_str, wp_pot, kind='linear', bounds_error=False, fill_value=0.0)
                wp_fft = fp_fft(h_fft)
                Fp_fft = np.fft.fft(wp_fft)
                Np_fft = wp_fft.size
                AFp_fft = np.abs(Fp_fft)[:Np_fft // 2] * 2 / Np_fft
                temp_pot_energy[i] = 0.5*np.sum(AFp_fft**2)
                #########################################################
                #   LSF - Least Squares Fitting - Find optimal wavelength
                #########################################################
                qui2old = 0
                l_lower = 500
                l_upper = 12000
                lamb_arr = np.arange(l_lower, l_upper+10, stepLSM)
                #lamb_arr = np.arange(l_lower, l_upper + 10, 100)
                qui2 = np.zeros(shape=(len(lamb_arr),))
                phase_val = np.zeros(shape=(len(lamb_arr),))
                amp_val   = np.zeros(shape=(len(lamb_arr),))
                a_val = np.zeros(shape=(len(lamb_arr),))
                b_val = np.zeros(shape=(len(lamb_arr),))
                c_val = np.zeros(shape=(len(lamb_arr),))
                for j in range(0, len(lamb_arr)):
                    popt, pcov = curve_fit(lambda x, a, b, c, A, B: func(x, a, b, c, A, B, lamb_arr[j]), h_str, w_str)
                    w_fit_wave = func(h_str, popt[0], popt[1], popt[2], popt[3],popt[4], lamb_arr[j])
                    qui2[j] = np.sum((w_fit_wave - w_str) ** 2)
                    phase_val[j] = np.arctan2(-popt[4], popt[3]) * 180 / np.pi
                    amp_val[j] = np.sqrt(popt[3] ** 2 + popt[4] ** 2)
                    a_val[j] = popt[0]
                    b_val[j] = popt[1]
                    c_val[j] = popt[2]
                index_min = np.where(qui2 == np.amin(qui2))[0][0]
                lamb_max = lamb_arr[index_min]
                amp_t[i] = amp_val[index_min]
                phase_t[i] = phase_val[index_min]
                lamb_t[i] = lamb_max
                a_t[i] = a_val[index_min]
                b_t[i] = b_val[index_min]
                c_t[i] = c_val[index_min]
                ngl = w_str.size - 5
                rel_err_t[i] = np.sqrt(qui2[index_min] / ngl) / np.std(w_str)

                popt, pcov = curve_fit(lambda x, a, b, c, A, B: func(x, a, b, c, A, B, lamb_max), h_str, w_str)
                w_fit_total = func(h_str, popt[0], popt[1], popt[2], popt[3], popt[4], lamb_max)
                w_parabolic = func1(h_str, popt[0], popt[1], popt[2])
                w_net_fit = w_str - w_parabolic
                w_fit_sin = func2(h_str, popt[3], popt[4], lamb_max)

                #########################################################
                #   Refine FFT - Fast Fourier Transform - Power Spectrum
                #########################################################
                f_fft = interp1d(h_str, w_net_fit, kind='linear', bounds_error=False, fill_value=0.0)
                w_fft = f_fft(h_fft)
                F_fft = np.fft.fft(w_fft)
                N_fft = w_fft.size
                AF_fft = np.abs(F_fft)[:N_fft // 2] * 2 / N_fft
                max_value = np.amax(AF_fft)
                i_max = np.where(AF_fft == max_value)
                fft_coef_im = F_fft[i_max]
                # potential energy FFT
                wp_pot = np.divide(np.divide(w_net_fit, wp_str), N_brunt)
                fp_fft = interp1d(h_str, wp_pot, kind='linear', bounds_error=False, fill_value=0.0)
                wp_fft = fp_fft(h_fft)
                Fp_fft = np.fft.fft(wp_fft)
                Np_fft = wp_fft.size
                AFp_fft = np.abs(Fp_fft)[:Np_fft // 2] * 2 / Np_fft
                if 0.5 * np.sum(AF_fft ** 2) < 0.125 * (np.amax(w_net) - np.amin(w_net)) ** 2 and lamb_max != l_upper:
                    temp_energy[i] = 0.5 * np.sum(AF_fft ** 2)
                    temp_pot_energy[i] = 0.5 * np.sum(AFp_fft ** 2)
                    explan = max_value ** 2 / temp_energy[i]
                    ilamb = np.fft.fftfreq(N_fft, L_fft)[:N_fft // 2]
                    ratio_im = np.imag(fft_coef_im) / np.real(fft_coef_im)
                    phase_max = np.arctan(ratio_im) * 180 / np.pi
                    if ilamb[i_max] != 0:
                        lamb_max_fft = 1 / ilamb[i_max]
                    else:
                        lamb_max_fft = max(h_fft) - min(h_fft)
                else:  # in this case, call the old fitting
                    qui2_old, phase_val_old, amp_val_old = call_old_fitting(lamb_arr, h_str, w_net)
                    index_min = np.where(qui2_old == np.amin(qui2_old))
                    lamb_max = lamb_arr[index_min]
                    phase_t[i] = phase_val_old[index_min]
                    amp_t[i] = amp_val_old[index_min]
                    lamb_t[i] = lamb_max
                    a_t[i] = popt1[0]
                    b_t[i] = popt1[1]
                    c_t[i] = popt1[2]
                    ngl = w_str.size - 2
                    rel_err_t[i] = np.sqrt(qui2_old[index_min] / ngl) / np.std(w_str)
                    popt, pcov = curve_fit(lambda x, A, B: func2(x, A, B, lamb_max), h_str, w_net)
                    w_fit_sin = func2(h_str, popt[0], popt[1], lamb_max)
                    w_parabolic = func1(h_str, popt1[0], popt1[1], popt1[2])
                    w_fit_total = w_parabolic + w_fit_sin
                    w_net_fit = w_net

                # Plot the most 'monochromatic like' waves
                # if explan >= thr_exp:
                #     # Plot wave and fitted curve
                #     fig, axs = plt.subplots(2)
                #     fig.set_size_inches(6, 8)
                #     axs[0].set(ylabel="Temperature variation (K)")
                #     axs[0].set(xlabel="Altitude (m)")
                #     axs[0].plot(h_str, w_str, '-k')
                #     axs[0].plot(h_str, w_parabolic, '--k')
                #     axs[0].plot(h_str, w_fit_total)
                #     # Plot FFT spectrum
                #     axs[1].set(ylabel="Amplitude")
                #     axs[1].set(xlabel="Wave-number [1/m]")
                #     axs[1].bar(ilamb[:N_fft // 2], AF_fft, width=1E-4)  # 1 / N is a normalization factor
                #     print('Iteration number: ', i, 'Lambda (m): ', lamb_max, 'Lambda FFT', lamb_max_fft, 'Potential En: ', temp_pot_energy[i])
                #     plt.show()
                #     # ____________________________ raw_input("Press [enter] to continue.")
                #     plt.close(fig)


            print('Mean N = ',brunt_t_mean[brunt_t_mean != 0],'\t'
                  ,'size_N=',brunt_t_mean[brunt_t_mean != 0].size)
            print('Sd N = ', brunt_t_sd[brunt_t_mean != 0], '\t'
                  , 'size_N=', brunt_t_sd[brunt_t_mean != 0].size)
            print('day_N =',day_N[brunt_t_mean != 0],'\t','size_dayN', day_N[brunt_t_mean != 0].size)
            if(ESTACAO=='al'):
                salvaBrunt('N_mean', brunt_t_mean[brunt_t_mean != 0], day_N[brunt_t_mean != 0])
                salvaBrunt('N_sd', brunt_t_sd[brunt_t_mean != 0], day_N[brunt_t_mean != 0])


            # fig1 = plt.figure(1)
            #
            # # Plot anual average and save
            # mean = np.divide(summ, n_n)
            # plt.plot(h_reg, mean, '-')
            # plt.plot(h_reg, mean*0, '--', color="black")
            # plt.title('Temperature '+ aer_cam_ano_est)
            # plt.ylabel('$^\circ$C')
            # plt.xlabel('altitude (m)')
            # local = str('analytic' + '\\' + aer_cam_ano_est + 'Temp.png')
            # plt.savefig(local, format='png', dpi=300)
            # fig1.show()
            # #____________________________
            # #input("Press [enter] to continue.")
            # plt.close(fig1)
            # salvaVariavel("Te", mean, h_reg)# Comentario temporario para algoritimo do N de Brunt-Vaisala

            # Plot pressure vs altitude - no GW analysis <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
            summ = np.zeros(shape=(h_reg.size,))
            n_n = np.zeros(shape=(h_reg.size,))
            u_zeros = np.zeros(shape=(h_reg.size,))
            u_ones = np.ones(shape=(h_reg.size,))
            fig1 = plt.figure(1)
            print('\t----------->Analysing PRESSURE<-----------\n')
            for i in range(0, len_init_t-1):
                continue
                ###################################################
                #   Verification of day
                ###################################################
                month = data_ptemp['mat'][jj][arr_init_t[i], 1]
                day   = data_ptemp['mat'][jj][arr_init_t[i], 2]
                hour  = data_ptemp['mat'][jj][arr_init_t[i], 3]
                days  = month_days(month)
                days  = days + day
                if i>0 and hour == data_ptemp['mat'][jj][arr_init_t[i-1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if i>0 and i<len_init_t-1 and\
                        hour == data_ptemp['mat'][jj][arr_init_t[i+1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    continue

                print(' PRESSURE i =', i, 'day =', days, round(i / (len_init_t - 2) * 100, 2), "% t(min) =",
                      round((time.time() - ini_prog)/60, 2), aer_cam_ano_est)
                h = data_ptemp['mat'][jj][arr_init_t[i]:(arr_init_t[i+1]-1), 10]
                p = data_ptemp['mat'][jj][arr_init_t[i]:(arr_init_t[i+1]-1), 11]
                if len(h) < 2:
                    continue
                f = interp1d(h, p, kind='linear', bounds_error=False, fill_value=0.0)
                f_reg = f(h_reg)
                summ = summ + f_reg
                n_n = n_n + np.where(f_reg == 0, u_zeros, u_ones)
                plt.plot(h, p, '.k')
                #fig1.show()
                #input("Press [enter] to continue.")

            # Plot the anual mean pressure and save
            mean = np.divide(summ, n_n)
            plt.plot(h_reg, mean, '.', color="pink")
            plt.title('Pressure '+aer_cam_ano_est)
            plt.ylabel('Torr')
            plt.xlabel('altitude (m)')
            local = str('analytic' + '\\' + aer_cam_ano_est + 'Press.png')
            plt.savefig(local, format='png', dpi=300)
            #fig1.show()
            #____________________________
            #input("Press [enter] to continue.")
            plt.close(fig1)
            # salvaVariavel("Pr", mean, h_reg) # Comentario temporario para algoritimo do N de Brunt-Vaisala

            ###################################################################################
            #
            # Data Analysis:                                                                  #
            # Here each ENERGY is analysed.
            #
            ###################################################################################
            fig1 = plt.figure(1)

            # Plot the histogram of the total energy
            tot_energy = np.array([])
            day_number = np.array([])

            arr_kin_energy      = np.array([])
            arr_pot_energy      = np.array([])
            arr_kin_energy_corr = np.array([])
            arr_pot_energy_corr = np.array([])
            day_kin_number      = np.array([])
            day_pot_number      = np.array([])
            day_kin_numberFFT   = np.array([])
            day_pot_numberFFT   = np.array([])

            # Compute all valid Kinetic energy (!=0)
            days = 0
            for j in range(0, len_init_w-1):
                # print(j/2, wn_energy[j], we_energy[j])
                if wn_energy[j] == we_energy[j] == 0: # Is differente of energy days with the FFT
                    continue
                month = data_pwind['mat'][jj][arr_init_w[j], 1]
                day   = data_pwind['mat'][jj][arr_init_w[j], 2]
                hour  = data_pwind['mat'][jj][arr_init_w[j], 3]
                days  = month_days(month)
                days  = days + day
                if j>0 and hour == data_pwind['mat'][jj][arr_init_w[j-1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[j-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if j>0 and j<len_init_w-1 and\
                        hour == data_pwind['mat'][jj][arr_init_w[j+1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[j+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    continue
                kin_energy     = 0.5 * (wn_energy[j] + we_energy[j])
                arr_kin_energy = np.append(arr_kin_energy, kin_energy)
                day_kin_number = np.append(day_kin_number, days)
                # print("\t", days, kin_energy)

            # Compute all variables FFT of WIND
            days = 0
            for j in range(0, len_init_w):
                month = data_pwind['mat'][jj][arr_init_w[j], 1]
                day   = data_pwind['mat'][jj][arr_init_w[j], 2]
                hour  = data_pwind['mat'][jj][arr_init_w[j], 3]
                days  = month_days(month)
                days  = days + day
                if j>0 and hour == data_pwind['mat'][jj][arr_init_w[j-1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[j-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if j>0 and j<len_init_w-1 and\
                        hour == data_pwind['mat'][jj][arr_init_w[j+1], 3] and \
                        day == data_pwind['mat'][jj][arr_init_w[j+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    day_kin_numberFFT = np.append(day_kin_numberFFT, None)
                    continue
                day_kin_numberFFT = np.append(day_kin_numberFFT, days)
                #print('dia=', day, 'mes=', month, 'h=', hour, 'diaFFT=', day_kin_numberFFT[j])

            # Compute all valid Potential energy (!=0)
            days = 0
            for i in range(0, len_init_t-1):
                #print(i/2, temp_pot_energy[i])
                if temp_pot_energy[i] == 0: # Is differente of energy days with the FFT
                    continue
                month = data_ptemp['mat'][jj][arr_init_t[i], 1]
                day   = data_ptemp['mat'][jj][arr_init_t[i], 2]
                hour  = data_ptemp['mat'][jj][arr_init_t[i], 3]
                days  = month_days(month)
                days  = days + day
                if i>0 and hour == data_ptemp['mat'][jj][arr_init_t[i-1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if i>0 and i<len_init_t-1 and\
                        hour == data_ptemp['mat'][jj][arr_init_t[i+1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    continue
                pot_energy     = temp_pot_energy[i]
                arr_pot_energy = np.append(arr_pot_energy, pot_energy)
                day_pot_number = np.append(day_pot_number, days)
                #print("\t", days, pot_energy)

            # Compute all variables FFT of TEMPERATURE
            days = 0
            for i in range(0, len_init_t):
                month = data_ptemp['mat'][jj][arr_init_t[i], 1]
                day   = data_ptemp['mat'][jj][arr_init_t[i], 2]
                hour  = data_ptemp['mat'][jj][arr_init_t[i], 3]
                days  = month_days(month)
                days  = days + day
                if i>0 and hour == data_ptemp['mat'][jj][arr_init_t[i-1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i-1], 2] and \
                        6 < hour and hour < 18:
                    hour = 23
                if i>0 and i<len_init_t-1 and\
                        hour == data_ptemp['mat'][jj][arr_init_t[i+1], 3] and \
                        day == data_ptemp['mat'][jj][arr_init_t[i+1], 2] and \
                        hour >= 18:
                    hour = 11
                if hour >= 21:
                    days = days + 0.5
                if hour <= 3:
                    days = days + 0.5 - 1
                if days not in station_days(ESTACAO):
                    day_pot_numberFFT = np.append(day_pot_numberFFT, None)
                    continue
                day_pot_numberFFT = np.append(day_pot_numberFFT, days)
                # print('dia=', day, 'mes=', month, 'h=', hour, 'diaFFT=', day_pot_numberFFT[i])

            # Compute total energy and plot histogram
            days = 0
            for j in range(0, len_init_w-1):
                if wn_energy[j] == we_energy[j] == 0:
                    continue
                for i in range(0, len_init_t-1):
                    if temp_pot_energy[i] == 0:
                        continue
                    if data_pwind['mat'][jj][arr_init_w[j], 1] == data_ptemp['mat'][jj][arr_init_t[i], 1] and \
                            data_pwind['mat'][jj][arr_init_w[j], 2] == data_ptemp['mat'][jj][arr_init_t[i], 2] and \
                            data_pwind['mat'][jj][arr_init_w[j], 3] == data_ptemp['mat'][jj][arr_init_t[i], 3]:

                        month = data_pwind['mat'][jj][arr_init_w[j], 1]
                        day = data_pwind['mat'][jj][arr_init_w[j], 2]
                        hour = data_pwind['mat'][jj][arr_init_w[j], 3]
                        days = month_days(month)
                        days = days + day
                        # print('dia=', day, 'mes=', month, 'h=', hour, 'dias=', days)
                        if j > 0 and hour == data_pwind['mat'][jj][arr_init_w[j - 1], 3] and \
                                day == data_pwind['mat'][jj][arr_init_w[j - 1], 2] and \
                                6 < hour and hour < 18:
                            hour = 23
                        if j > 0 and j < len_init_t - 1 and \
                                hour == data_pwind['mat'][jj][arr_init_w[j + 1], 3] and \
                                day == data_pwind['mat'][jj][arr_init_w[j + 1], 2] and \
                                hour >= 18:
                            hour = 11
                        if hour >= 21:
                            days = days + 0.5
                        if hour <= 3:
                            days = days + 0.5 - 1

                        if days not in station_days(ESTACAO):
                            continue

                        kin_energy = 0.5 * (wn_energy[j] + we_energy[j])
                        pot_energy = temp_pot_energy[i]
                        temp_energy = kin_energy + pot_energy
                        tot_energy = np.append(tot_energy, temp_energy)
                        arr_kin_energy_corr = np.append(arr_kin_energy_corr, kin_energy)# K = P
                        arr_pot_energy_corr = np.append(arr_pot_energy_corr, pot_energy)# P = K
                        day_number = np.append(day_number, days) # The days that have K and P together
                        # print(i,j,days, kin_energy, pot_energy)

            # Plot the histogram of the total pseudo energy
            fig1 = plt.figure(1)
            n, bins, patches = plt.hist(tot_energy, bins='auto', color='red', ##0504aa
                                        alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel(r'Total Energy $(m^2/s^2)$')
            plt.ylabel('Frequency')
            plt.title('Total Energy Histogram '+aer_cam_ano_est)
            local = str('analytic' + '\\' + aer_cam_ano_est + 'histM.png')
            plt.savefig(local, format='png', dpi=300)
            fig1.show()
            #____________________________
            #input("Press [enter] to continue.")
            plt.close(fig1)

            # Plot the histogram of the total pseudo kinetic energy
            fig1 = plt.figure(1)
            n, bins, patches = plt.hist(arr_kin_energy, bins='auto', color='green',
                                        alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel(r'Kinetic Energy $(m^2/s^2)$')
            plt.ylabel('Frequency')
            plt.title('Kinetic Energy Histogram '+aer_cam_ano_est)
            local = str('analytic' + '\\' + aer_cam_ano_est + 'histK.png')
            plt.savefig(local, format='png', dpi=300)
            fig1.show()
            #____________________________
            #input("Press [enter] to continue.")
            plt.close(fig1)

            # Plot the histogram of the total potential energy
            fig1 = plt.figure(1)
            n, bins, patches = plt.hist(arr_pot_energy, bins='auto', color='orange',
                                        alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel(r'Potential Energy $(m^2/s^2)$')
            plt.ylabel('Frequency')
            plt.title('Potential Energy Histogram '+aer_cam_ano_est)
            local = str('analytic' + '\\' + aer_cam_ano_est + 'histP.png')
            plt.savefig(local, format='png', dpi=300)
            fig1.show()
            #____________________________
            # input("Press [enter] to continue.")
            plt.close(fig1)

            # XY-scatter plot to visualize correlation
            print('\t----------->Analysing KP - Scatter Plot<-----------\n')
            if arr_kin_energy_corr.size > 1:
                R_val = np.corrcoef(arr_kin_energy_corr, arr_pot_energy_corr)#;print(' R_val = ', R_val)
                ngl = arr_kin_energy_corr.size - 2
                t = R_val[0, 1]*np.sqrt(ngl/(1 - R_val[0, 1]**2))
                p = 1 - stats.t.cdf(t, df=ngl)
                slope, intercept, r_value, p_value, std_err = stats.linregress(arr_kin_energy_corr,
                                                                               arr_pot_energy_corr)# ;print('std_err = ', std_err)

                fig1 = plt.figure(1)
                plt.plot(arr_kin_energy_corr, arr_pot_energy_corr, '.')
                plt.plot(arr_kin_energy_corr, arr_kin_energy_corr*slope+intercept, color='red')
                ax = plt.gca()
                plt.title('Scatter Plot P and K ' + aer_cam_ano_est)
                plt.xlabel(r'Kinetic Energy $(m^2/s^2)$')
                plt.ylabel(r'Potential Energy $(m^2/s^2)$')
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = '\n'.join(('R = %f' % (R_val[0, 1], ),
                                     'p = %f' % (p_value, ),
                                     'a = %f' % (slope, ),
                                     'b = %f' % (intercept, ),
                                     'n = %i' % (arr_kin_energy_corr.size, )))
                ax.text(0.7, 0.7, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=props)# Place a text box in upper left in axes coords
                local = str('analytic'+'\\' + aer_cam_ano_est + 'scatter' + 'KP' + '.png')
                plt.savefig(local, format='png', dpi=300)
                fig1.show()
                #____________________________
                #input("Press [enter] to continue")
                plt.close(fig1)

                salvaEst(R_val[0, 1], p_value, slope, intercept, arr_kin_energy_corr.size, std_err)
            else:
                salvaEst("", "", "", "", arr_kin_energy_corr.size, "")

            ###################################################################################
            #
            # Data Analysis:                                                                  #
            # Here each FFT is analysed and save
            #
            ###################################################################################

            # Ploting and saving variables from FFT
            print('\t----------->Analysing and Saving FFT<-----------\n')
            print("day K FFT: ", day_kin_numberFFT.size, ' =?= ',
                  "dados K FFT: ", len_init_w, '\n',
                  "day P FFT: ", day_pot_numberFFT.size, ' =?= ',
                  "dados P FFT: ", len_init_t, "\n")

            indK_FFT = np.where(day_kin_numberFFT != None)
            indK_FFTr= np.where(phase_nw[indK_FFT] != 0)
            indP_FFT = np.where(day_pot_numberFFT != None)
            indP_FFTr= np.where(phase_t[indP_FFT] != 0)

            if len(phase_nw[indK_FFT]) == 0 or len(phase_nw[indK_FFT][indK_FFTr]) == 0:
                print('\n /!\ The datas FFT wind is empty! :(')
            else:
                # fig1 = plt.figure(1)
                # plt.title('Phase Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], phase_nw[indK_FFT][indK_FFTr], 'o')
                # #local = str('analytic'+'\\' + aer_cam_ano_est + 'NWphase' + '.png')
                # #plt.savefig(local, format='png', dpi=300)
                # # fig1.show()
                # # input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("nw_pha", phase_nw[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])
                salvaFFT("ew_pha", phase_ew[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Amplitude Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], amp_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("nw_amp", amp_nw[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])
                salvaFFT("ew_amp", amp_ew[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Lambda Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], lamb_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("nw_lamb", lamb_nw[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])
                salvaFFT("ew_lamb", lamb_ew[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Rel. Error Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], rel_err_wn[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("nw_rel", rel_err_wn[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])
                salvaFFT("ew_rel", rel_err_ew[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Ax^2+bx+c Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], a_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("nw_paA", a_nw[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])
                salvaFFT("ew_paA", a_ew[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('ax^2+Bx+c Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], a_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("nw_paB", b_nw[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])
                salvaFFT("ew_paB", b_ew[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('ax^2+bx+C Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT, c_nw, 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("nw_paC", c_nw[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])
                salvaFFT("ew_paC", c_ew[indK_FFT][indK_FFTr], day_kin_numberFFT[indK_FFT][indK_FFTr])

            if len(phase_t[indP_FFT]) == 0 or len(phase_t[indP_FFT][indP_FFTr]) == 0:
                print('\n /!\ The datas FFT temperature is empty! :(')
            else:
                # fig1 = plt.figure(1)
                # plt.title('Amplitude Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], amp_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("te_pha", phase_t[indP_FFT][indP_FFTr], day_pot_numberFFT[indP_FFT][indP_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Amplitude Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], amp_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("te_amp", amp_t[indP_FFT][indP_FFTr], day_pot_numberFFT[indP_FFT][indP_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Lambda Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], lamb_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("te_lamb", lamb_t[indP_FFT][indP_FFTr], day_pot_numberFFT[indP_FFT][indP_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Rel. Error Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], rel_err_wn[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                salvaFFT("te_rel", rel_err_t[indP_FFT][indP_FFTr], day_pot_numberFFT[indP_FFT][indP_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('Ax^2+bx+c Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], a_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("te_paA", a_t[indP_FFT][indP_FFTr], day_pot_numberFFT[indP_FFT][indP_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('ax^2+Bx+c Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT[indK_FFT][indK_FFTr], a_nw[indK_FFT][indK_FFTr], 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("te_paB", b_t[indP_FFT][indP_FFTr], day_pot_numberFFT[indP_FFT][indP_FFTr])

                # fig1 = plt.figure(1)
                # plt.title('ax^2+bx+C Northerly Wind '+aer_cam_ano_est)
                # plt.plot(day_kin_numberFFT, c_nw, 'o')
                # #fig1.show()
                # #input("Press [enter] to continue.")
                # plt.close(fig1)
                salvaFFT("te_paC", c_t[indP_FFT][indP_FFTr], day_pot_numberFFT[indP_FFT][indP_FFTr])

            # Here is made the denoisy, ACF, PACF of energies
            print('\t----------->Analysing ENERGY<-----------\n')

            energias = (arr_kin_energy, arr_pot_energy, tot_energy)
            dias = (day_kin_number, day_pot_number, day_number)
            tipos = ('K', 'P', 'M')
            cores = ('green', 'orange', 'red')
            tamanhos = (len(arr_kin_energy), len(arr_pot_energy), len(tot_energy))

            print(' Tamanhos correlação de K, P, M e: \n\t'
                  , len(arr_kin_energy_corr), len(arr_pot_energy_corr), len(tot_energy))
            print(' Tamanhos de K, P, M e: \n\t'
                  , len(arr_kin_energy), len(arr_pot_energy), len(tot_energy))
            print(' Tamanhos dias K, P, M e: \n\t'
                  , len(day_kin_number), len(day_pot_number), len(day_number), '\n')

            energia, dia, tipo = [], [], []

            for m in range(0, 3):

                energia = energias[m]
                dia = dias[m]
                tipo = tipos[m]
                cor = cores[m]
                tamanho = np.floor(tamanhos[m]*.75)

                if(tamanho < 2):
                    continue

                print(" ----------------<<<<", tipo, " de K, P e M>>>>----------------")

                # Localizing days without colect
                intervalo = [0.5 * x for x in range((366 + 2) * 2 - 1)]  # Days of colect
                indicador_falta = np.zeros(len(intervalo))
                cont = 0
                for i in range(len(intervalo)):
                    if intervalo[i] == dia[cont]:
                        #print(tipo, "i=", i, "intervalo e dia com dado: ", intervalo[i], dia[cont])
                        indicador_falta[i] = None
                        if cont < (len(dia)-1):
                            cont += 1
                            while dia[cont] == dia[cont-1]:
                                cont += 1

                # # Residuos of energy: ACF and PACF
                # plot_acf(energia, lags=tamanho, marker='', title='ACF ' + tipo + ' ' + aer_cam_ano_est, color=cor)
                # plt.xlabel('lag')
                # plt.ylabel('ACF')
                # local = str('analytic' + '\\' + aer_cam_ano_est + 'acf' + tipo + '.png')
                # plt.savefig(local, format='png', dpi=300)
                # #pyplot.show()
                # #____________________________ raw_input("Press [enter] to continue.")
                # plt.close()
                #
                # if tamanho > 12:
                #     plot_pacf(energia, lags=tamanho, marker='', title='PACF ' + tipo + ' ' + aer_cam_ano_est,color=cor)
                #     plt.xlabel('lag')
                #     plt.ylabel('PACF')
                #     local = str('analytic' + '\\' + aer_cam_ano_est + 'pacf' + tipo + '.png')
                #     plt.savefig(local, format='png', dpi=300)
                #     #pyplot.show()
                #     #____________________________ raw_input("Press [enter] to continue.")
                #     plt.close()
                # else:
                #     print(' Nao executou PACF pois tamanho =', tamanho, '\n')

                # Residuos of energy: test for trendy
                tendencia = mk.original_test(energia)
                #print(tendencia, '\n')

                # Residuos of energy: linear regression
                X = dia.reshape(-1, 1)
                Y = energia.reshape(-1, 1)
                regressor = LinearRegression()
                regressor.fit(X, Y)
                beta1 = regressor.coef_#;print("Inclinacao e: ", beta1) # Inclination
                beta0 = regressor.intercept_#;print("Intercepto e: ",beta0)
                Y_hat = regressor.predict(X)

                # Residuos of energy: Destrend
                fig1 = plt.figure(1)
                x_detrended = signal.detrend(energia)
                ax = plt.gca()
                plt.plot(dia, energia, color=cor)
                plt.plot(dia, x_detrended, label="No trends", color='gray')
                plt.xlim(0, 370)
                plt.plot(X, Y_hat, linestyle='--', color='black')
                plt.plot(np.arange(370), np.arange(370)*0, color='black')
                plt.plot(intervalo, indicador_falta, '|', color='yellow')
                plt.legend(loc='upper right')
                plt.title('Energia ' + tipo + ' ' + aer_cam_ano_est)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = '\n'.join(('a      = %f' % (beta1, ),
                                     'b      = %f' % (beta0, ),
                                     'MK-p = %f' % (tendencia[2], ),
                                     'n      = %i' % (tamanhos[m], ),
                                     'n(p)= %f' % (round(tamanhos[m] * 100 / (366 * 2), 3)),
                                     ))
                ax.text(0.2, 0.7, textstr, transform=ax.transAxes, fontsize=8,
                        verticalalignment='bottom', bbox=props) # Place a text box in upper left in axes coords
                plt.xlabel('dia')
                plt.ylabel(r'$(m^2/s^2)$')
                local = str('analytic' + '\\' + aer_cam_ano_est + 'destrend' + tipo + '.png')
                plt.savefig(local, format='png', dpi=300)
                fig1.show()
                #____________________________
                # input("Press [enter] to continue.")
                plt.close(fig1)

                salvaEne(energia, tipo, dia, tendencia, beta0, beta1)

            print(" ---------------->>>> out ----------------")

fim_prog = time.time() - ini_prog
print("Tempo de execucao:", round(fim_prog, 2), "s =", round(fim_prog/60, 2), "min =", round(fim_prog/3600, 2), 'h\n')

