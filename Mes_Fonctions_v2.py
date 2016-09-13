# -*- coding: utf-8 -*-
from PIL.Image import *
from random import gauss
import matplotlib
import math
import os
import time
import numpy as np


# from decimal import Decimal

###############################################################################
#################### FONCTION GRADIENT i ######################################
###############################################################################  
def gradi(vect):
    """ --- Gradient selon x ---
    Prend comme entrée une matrice (une matrice)
    Retourne le gradient selon x de la matrice : abs(mat[i+1] - mat[i])
    
    On fait une iterpolation cad on suppose que vect est linéaire,
    du coup on fait la formule des dérrivés avec dt plus petit
    que 1 (ici 0.1), on a alors f(t) = vect(i) et 
    f(t+dt) = vect(i) + dt*(vect(i+1)-vect(i)/1) """

    NbLigne, NbColonne = vect.shape
    Retour = np.zeros((NbLigne, NbColonne))
    for i in range(0, NbLigne):
        for j in range(0, NbColonne):
            if i == NbLigne - 1:
                Retour[i, j] = abs((vect[i, j] - vect[i - 1, j]))
            else:
                Retour[i, j] = abs((vect[i + 1, j] - vect[i, j]))
    return Retour


###############################################################################
#################### FONCTION GRADIENT j ######################################
###############################################################################    
def gradj(vect):
    """ --- Gradient selon y ---
    Prend comme entrée une matrice (une matrice)
    Retourne le gradient selon y de la matrice : abs(mat[j+1] - mat[j])
    
    On fait une iterpolation cad on suppose que vect est linéaire,
    du coup on fait la formule des dérrivés avec dt plus petit
    que 1 (ici 0.1), on a alors f(t) = vect(j) et 
    f(t+dt) = vect(j) + dt*(vect(j+1)-vect(j)/1) """

    NbLigne, NbColonne = vect.shape
    Retour = np.zeros((NbLigne, NbColonne))
    for i in range(0, NbLigne):
        for j in range(0, NbColonne):
            if j == NbColonne - 1:
                Retour[i, j] = abs((vect[i, j] - vect[i, j - 1]))
            else:
                Retour[i, j] = abs((vect[i, j + 1] - vect[i, j]))
    return Retour


###############################################################################
################### FONCTION CONVOLUTION ######################################
###############################################################################     
def conv(image, kernel, place="centre", forme="matrice"):
    """ --- Convolution d'une image par un kernel ---
    Prend comme entrée:
    -l'image et une PSF (deux matrices) 
    -la chaine de caractère "coin" si la PSF envoyée est dans un coin
            (de base, la PSF envoyé est centrée)
    -la chaine de caractère "vecteur" si on souhaite un vecteur en retour
            (de base, l'image retournée est une matrice)
    Retourne leur produit de convolution
    
    La Convolution est faite par transformée de Fourier
    Les valeurs inférieurs à 0.001 sont filtrés
    La norme de la PSF reçu est testé et un message d'erreur est renvoyé si
    elle est supérieure à 1
    La norme des images avant et après convolution est comparée et un message
    d'erreur est renvoyé si elles sont différentes"""

    ###### ON S'ASSURE QUE LE NOYAU EST BIEN UNITAIRE
    #    try:
    #        assert round(sum(kernel)) == 1
    #    except AssertionError:
    #        print("  !!!!!!!!!! LA PSF N'EST PAS NORMALISEE DANS CONV !!!!!!!!!! ")
    #
    ##### ON EXTRAIT NbLigne ET NbColonne
    NbLigne, NbColonne = image.shape
    parite = 0

    ##### ON FAIT LE PRODUIT DE CONVOLUTION...
    if place == "coin":

        ##### ON RECENTRE LE KERNEL...
        if NbLigne % 2 == 0:
            kernel = np.roll(kernel.reshape(NbLigne * NbColonne), int(NbLigne * NbColonne / 2 - NbLigne / 2)).reshape(
                (NbLigne, NbColonne))
        else:
            kernel = np.roll(kernel.reshape(NbLigne * NbColonne), -int(np.floor(NbLigne * NbColonne / 2))).reshape(
                (NbLigne, NbColonne))
        TF = real(ifft2(fft2(image) * fft2(kernel)))

    ##### ON REPLACE CORRECTEMENT L'IMAGE
    elif place == "centre":
        TF = real(ifft2(fft2(image) * fft2(kernel)))
        TF = np.roll(TF.reshape(NbLigne * NbColonne), -int(floor(NbLigne * NbColonne / 2)))
        TF = TF.reshape((NbLigne, NbColonne))
        if NbColonne % 2 == 0:
            TF = np.roll(TF, 1, axis=1)
            parite = 1
        TF_intermediaire = TF.copy()
        for j in range(int(ceil((NbColonne + 1) / 2)), NbColonne + parite):
            for i in range(0, NbLigne):
                TF[(i + 1) % NbLigne, j % NbColonne] = TF_intermediaire[i, j % NbColonne]

    else:
        print("Erreur, mauvais paramètre d'entrée dans CONV")

    ##### ON FILTRE LES BASSES FREQUENCES
    TF = np.around(TF, decimals=10)

    ##### ON RETOURNE TF SOUS LA FORME DÉSIRÉE
    if forme == "vecteur":
        Retour = TF.reshape(NbLigne * NbColonne)
    else:
        Retour = TF

    ###### ON S'ASSURE DE LA CONSERVATION DE L'ENERGIE
    #    try:
    #        assert round(sum(image)) == round(sum(Retour))
    #    except AssertionError:
    #        print("  !!!!! INTEGRALE(IMAGE) != INTEGRALE(FLOUE) DANS CONV !!!!!! ")
    #
    ##### ON RETOURNE TF
    return Retour


###############################################################################
################### FONCTION CONVOLUTION 2 ####################################
###############################################################################     
def conv_matrices(A, kernel, forme="vecteur", place="coin"):
    """ --- Convolution d'une image par un kernel ---
    Prend comme entrée :
    -la matrice image A et une PSF (deux matrices) 
    -la chaine de caractère "matrice" si on souhaite une matrice en retour
            (de base, l'image retournée est un vecteur)
    -la chaine de caractère "coin" si la PSF envoyée est dans un coin
            (de base, la PSF envoyée est centrée)
    Retourne leur produit de convolution
    
    La Convolution est faite par produit de matrices
    Les valeurs inférieurs à 0.001 sont filtrés
    La norme de la PSF reçu est testé et un message d'erreur est renvoyé si
    elle est supérieure à 1
    La norme des images avant et après convolution est comparée et un message
    d'erreur est renvoyé si elles sont différentes"""

    ###### ON S'ASSURE QUE LE NOYAU EST BIEN UNITAIRE
    #    try:
    #        assert round(sum(kernel)) == 1
    #    except AssertionError:
    #        print("  !!!!!!!!!! LA PSF N'EST PAS NORMALISEE DANS CONV_MATRICES !!!!!!!!!! ")
    #
    ##### ON EXTRAIT "NbLigne" ET "NbColonne"
    NbLigne, NbColonne = kernel.shape

    ##### ...SAUF SI IL EST DEJA CENTRÉ
    if place == "centre":
        K = kernel.reshape(NbLigne * NbColonne)

    ##### ON RECENTRE LE KERNEL...
    elif place == "inv":

        if NbLigne % 2 == 0:
            K = np.roll(kernel[::-1].reshape(NbLigne * NbColonne), int(NbLigne * NbColonne / 2 - NbLigne / 2))
        else:
            K = np.roll(kernel[::-1].reshape(NbLigne * NbColonne), -int(floor(NbLigne * NbColonne / 2)))

            ##### ON RECENTRE LE KERNEL...
    elif place == "coin":

        if NbLigne % 2 == 0:
            K = np.roll(kernel.reshape(NbLigne * NbColonne)[::-1], int(NbLigne * NbColonne / 2 - NbLigne / 2))
        else:
            K = np.roll(kernel.reshape(NbLigne * NbColonne)[::-1], -int(floor(NbLigne * NbColonne / 2)))

            ##### ON FAIT LE PRODUIT DE CONVOLUTION ET ON FILTRE LES BASSES FREQUENCES
    AK = dot(A, K)
    AK = np.around(AK, decimals=10)

    ##### ON RETOURNE AK SOUS LA FORME DÉSIRÉE
    if forme == "matrice":
        Retour = AK.reshape((NbLigne, NbColonne))
    else:
        Retour = AK

    ###### ON S'ASSURE DE LA CONSERVATION DE L'ENERGIE
    #    try:
    #        assert np.around(sum(A[:,0])/sum(Retour), decimals = 2) == 1
    #    except AssertionError:
    #        print("  !!!!! INTEGRALE(IMAGE) != INTEGRALE(FLOUE) DANS CONV_MATRICES !!!!!! ")
    #
    ##### ON RETOURNE AK
    return Retour


###############################################################################
#################### FONCTION CRÉATION DE A ###################################
###############################################################################      
def matrice_A(image, methode="3"):
    """ --- Calcul de la matrice A ---
    Prend comme entrée:
    -l'image (matrice)
    -la chaine de caractère "1", "2" ou "3" selon la méthode souhaitée
            (de base, la méthode utilisé est la dernière)
    Retourne la matrice A (de taille n2 x n2)
    
    Met la matrice image sous forme d'une matrice de taille carré
    Permet le produit de convolution par calcul matriciel avec un
    noyau par le biais de la fonction "conv_finale" """

    ##### ON EXTRAIT NbLigne ET NbColonne
    NbLigne, NbColonne = image.shape

    ##### ON DÉCLARE A ET I
    A = zeros((NbLigne * NbColonne, NbLigne * NbColonne))
    I = image.reshape(NbLigne * NbColonne)

    ##### PREMIÈRE METHODE
    if methode == "1":
        for j in range(0, NbLigne * NbColonne):
            for i in range(0, NbLigne * NbColonne):
                if i - j < 0:
                    A[i, j] = I[i - j + NbLigne * NbColonne]
                else:
                    A[i, j] = I[i - j]
                    #            if j%(round(NbLigne*NbColonne/10)) == 0 :
                    #                print("Creation de A: ",10*round(j/(NbLigne*NbColonne/10)),"%")

                    ##### METHODE INTERMEDIAIRE
    elif methode == "2":
        for i in range(0, NbLigne * NbColonne):
            for j in range(0, NbLigne * NbColonne):
                indice_i = (floor((j + i % NbColonne) / NbColonne) + floor(i / NbColonne)) % NbLigne
                indice_j = (j + i % NbColonne) % NbColonne
                if ((abs(indice_j - i % NbColonne) <= NbColonne / 2) and (
                    abs(indice_i - floor(i / NbColonne)) <= NbLigne / 2)):
                    if i + j < NbLigne * NbColonne:
                        A[i, j] = I[i + j]
                    else:
                        A[i, j] = I[i + j - NbLigne * NbColonne]
                else:
                    A[i, j] = 0
                    #            if i%(round(NbLigne*NbColonne/10)) == 0 :
                    #                print("Creation de A: ",10*round(i/(NbLigne*NbColonne/10)),"%")

                    ##### METHODE FINALE
    elif methode == "3":
        for i in range(0, NbLigne * NbColonne):
            for j in range(0, NbLigne * NbColonne):
                indice_i = (floor((j + i % NbColonne) / NbColonne) + floor(i / NbColonne)) % NbLigne
                indice_j = (j + i % NbColonne) % NbColonne
                if ((abs(indice_j - i % NbColonne) <= NbColonne / 2) and (
                    abs(indice_i - floor(i / NbColonne)) <= NbLigne / 2)):
                    if i + j < NbLigne * NbColonne:
                        A[i, j] = I[i + j]
                    else:
                        A[i, j] = I[i + j - NbLigne * NbColonne]
                elif abs(indice_j - i % NbColonne) > NbColonne / 2:
                    if i % NbColonne <= NbColonne / 2:
                        A[i, j] = I[(i + j + NbColonne) % (NbLigne * NbColonne)]
                    else:
                        A[i, j] = I[(i + j - NbColonne) % (NbLigne * NbColonne)]
                elif abs(indice_i - floor(i / NbColonne)) > NbLigne / 2:
                    if i + j < NbLigne * NbColonne:
                        A[i, j] = I[i + j]
                    else:
                        A[i, j] = I[i + j - NbLigne * NbColonne]
                else:
                    print("Petit probleme")
                    # print("Creation de A: ligne= ",i, "sur ",NbLigne*NbColonne)
                    # if i%(round(NbLigne*NbColonne/10)) == 0 :
                    # print"Creation de A: ",10*round(i/(NbLigne*NbColonne/10)),"%"
    else:
        print("  !!!!!   PAR QUELLE METHODE SOUHAITEZ VOUS CALCULER A ?   !!!!! ")

    return A


###############################################################################
################# FONCTION CRÉATION DE L'IMAGE ################################
###############################################################################  
def mon_image(methode, nom="test_nette"):
    """ --- Importation de l'image ---
    Prend comme entrée:
    -la chaine de caractère "import" si l'on souhaite importer l'image
    ou la chaine "create si on souhaite la créer (image 9x9)
    -la chaine de caractère qui contient le nom de l'image à importer
            (de base, nom = "test_nette")
    Retourne:
    -NbLigne --> Le nombre de ligne de l'image
    -NbColonne --> Le nombre de colonne de l'image
    -image --> l'image sous forme de matrice (NbLigne x NbColonne)
    -latent --> une copie de l'image
    -kernel --> Une PSF qui servira à flouter l'image (gaussienne)
    -PSF --> La PSF de base que l'on tentera de faire converger vers "kernel"
    
    Si l'image est importée : l'image est ouverte, puis les differents noyaux
    sont créés (une gaussienne centrée pour "kernel" et une matrice ones pour 
    "PSF"), puis l'image est flouté par "kernel". 
    Si l'image est crée, c'est une matrice (9x9) contenant 4 points lumineux,
    "kernel" est en forme de L et PSF en forme de gaussienne". """

    ############## SI ON VEUT IMPORTER L'IMAGE ##############
    if methode == "import":

        ##### ON SE PLACE DANS LE RÉPERTOIRE APPROPRIÉ
        os.chdir("/obs/gheurtier/Python")

        ##### ON OUVRE L'IMAGE NETTE ET ON DÉFINIT SA TAILLE
        fichier = "./Images/" + nom
        _image = open(fichier, "r")
        image_liste = list(_image.getdata())
        NbColonne, NbLigne = _image.size

        ##### ON DÉFINIT IMAGE ET FLOUE, PSF ET KERNEL
        image = zeros((NbLigne, NbColonne))
        floue = zeros((NbLigne, NbColonne))
        PSF = ones((NbLigne, NbColonne))
        PSF = PSF / sum(PSF)
        kernel = zeros((NbLigne, NbColonne))

        ##### ON IMPORTE LES VALEURS DE L'IMAGE DANS "image"
        for i in range(0, NbLigne * NbColonne):
            image[floor(i / NbColonne), i % NbColonne] = image_liste[i][0]

        ##### ON CRÉÉ UNE GAUSSIENNE CENTRÉE ET UNITAIRE POUR "kernel"
        for i in range(0, NbLigne):
            for j in range(0, NbColonne):
                kernel[i, j] = 255 * e ** (-((i - NbLigne / 2) * (i - NbLigne / 2) / (NbLigne * NbLigne / 256)) - (
                    (j - NbColonne / 2) * (j - NbColonne / 2) / (NbColonne * NbColonne / 256)))
        kernel = kernel / sum(kernel)

    ############ SI ON VEUT CREER L'IMAGE ###################
    if methode == "create":
        ##### ON DÉFINIT "image" ET "kernel" ET LA TAILLE
        NbLigne = NbColonne = 9
        image = zeros((NbLigne, NbColonne))
        image[2, 2] = image[6, 2] = image[2, 6] = image[6, 6] = 255
        kernel = zeros((NbLigne, NbColonne))
        kernel[0, 0] = 255
        kernel[1, 0] = 200
        kernel[2, 0] = 70
        kernel[2, 1] = 150
        kernel = kernel / sum(kernel)
        kernel = np.roll(kernel, -int(floor(NbLigne / 2 + 1)), axis=0)
        kernel = np.roll(kernel, -int(floor(NbColonne / 2 + 1)), axis=1)

        ##### ON DÉFINIT "PSF"
        PSF = zeros((NbLigne, NbColonne))
        PSF[1, 0] = PSF[0, 1] = PSF[1, 2] = PSF[2, 1] = 100
        PSF[0, 0] = PSF[0, 2] = PSF[2, 0] = PSF[2, 2] = 20
        PSF[1, 1] = 255
        PSF = PSF / sum(PSF)
        PSF = np.roll(PSF, -int(floor(NbLigne / 2 + 2)), axis=0)
        PSF = np.roll(PSF, -int(floor(NbColonne / 2 + 2)), axis=1)

    ######## SI ON VEUT TESTER LA CONVOLUTION ##############
    if methode == "test":
        ##### ON DÉFINIE DES MATRICES SIMPLES À CALCULER
        NbLigne = NbColonne = 3
        image = (arange(9) + 1).reshape((3, 3))
        kernel = array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        PSF = zeros((3, 3))

    ##### ON DÉFINIT "latent"
    latent = image.copy()

    ##### ON RETOURNE TOUTES LES DONNÉES CRÉÉES
    return NbLigne, NbColonne, image, latent, kernel, PSF


###############################################################################
################# FONCTION VÉRIFICATION DES DONNÉES ###########################
###############################################################################
def Verifications(On_Verifie_les_kernels=0, On_Verifie_A=0, On_Verifie_la_conv=0, On_Verifie_la_deuxieme_conv=0,
                  On_Compare_deux_matrices=0):
    """ --- Vérification de la méthode ---
    Prend comme entrée 5 booléens (1 = oui, 0 = non)    
    Retourne:
    -1er --> L'affichage du kernel et de la PSF, et leurs norme
    -2eme --> L'affichage de l'image et de A
          --> (A[:,0]-image), et sa somme
    -3eme --> L'affichage de floue , de conv_matrices et de leur difference
          --> (conv_matrices-conv) et sa somme
    -4eme --> L'affichage de conv(PSF), de conv_matrices(PSF) et de leur difference
          --> (conv_matrices-conv)(PSF) et sa somme
    -5eme --> conv_matrices et conv cote à cote
          --> liste des éléments similaires dans conv_matrices et conv, 
    leur position, leur écart, leur nombre. """

    if On_Verifie_les_kernels == 1:
        ########################################################
        # Calcul "kernel" et "PSF"
        # Calcul leur norme
        # Affiche "kernel" et "PSF"
        print("\n \n \n kernel = \n", kernel, "\n \n PSF = ", PSF, "\n \n normes = ", sum(kernel), "  et  ", sum(PSF))
        matshow(kernel, cmap=cm.gray)
        matshow(PSF, cmap=cm.gray)
        ########################################################

    if On_Verifie_A == 1:
        ########################################################
        # Calcul "A-image" 
        # Calcul la moyenne de leur difference
        # Affiche "image" et "A"
        print("\n \n \n A - image = \n", A[:, 0].reshape((NbLigne, NbColonne)) - image,
              " \n \n Somme des différences = ", sum(abs(A[:, 0].reshape((NbLigne, NbColonne)) - image)), "\n \n \n \n")
        matshow(image, cmap=cm.gray)
        matshow(A, cmap=cm.gray)
        ########################################################

    if On_Verifie_la_conv == 1:
        ########################################################
        # Calcul "conv_matrices(A, kernel) - conv(image, kernel)" 
        # Calcul la moyenne de leur difference
        # Affiche "floue", "conv_matrice" et leur difference
        print("AK_vrai - floue = \n", conv_matrices(A, kernel, "matrice", "coin") - floue,
              "\n \n Somme des différences = ", sum(abs(conv_matrices(A, kernel, "matrice", "coin") - floue)))
        matshow(floue, cmap=cm.gray)
        matshow(conv_matrices(A, kernel, "matrice", "coin"), cmap=cm.gray)
        matshow(floue - conv_matrices(A, kernel, "matrice", "coin"), cmap=cm.gray)
        ########################################################

    if On_Verifie_la_deuxieme_conv == 1:
        ########################################################
        # Calcul "conv_matrices(A, PSF) - conv(image, PSF)" 
        # Calcul la moyenne de leur difference
        # Affiche conv, "conv_matrice" et leur difference
        print("\n \n \n AK - conv(image,PSF) = \n", conv_matrices(A, PSF, "matrice") - conv(image, PSF),
              "\n \n Moyenne des differences = ",
              sum(abs(conv_matrices(A, PSF, "matrice") - conv(image, PSF))) / (NbLigne * NbColonne))
        matshow(conv(image, PSF, "centre"), cmap=cm.gray)
        matshow(conv_matrices(A, PSF, "matrice"), cmap=cm.gray)
        matshow(conv(image, PSF, "centre") - conv_matrices(A, PSF, "matrice"), cmap=cm.gray)
        ########################################################

    if On_Compare_deux_matrices == 1:
        ########################################################
        # On Calcul les valeurs des deux matrices côte à côte
        # On Calcul les elements similaires de D et F, leur position, leur écart, leur nombre
        D = conv(image, kernel)
        F = conv_matrices(A, kernel, "matrice", "coin")
        l = 0
        for k in range(0, 3 * NbColonne):
            print(int(floor(k / NbColonne)), "  ", k % NbColonne, "  D = ", D[int(floor(k / NbColonne)), k % NbColonne],
                  "    F = ", F[int(floor(k / NbColonne)), k % NbColonne])
        for k in range(0, NbLigne * NbColonne):
            for i in range(0, NbLigne):
                for j in range(0, NbColonne):
                    if D[i, j] == F[int(floor(k / NbColonne)), k % NbColonne]:
                        print("Positions = ", i + j, "  ", int(floor(k / NbColonne)) + k % NbColonne, " Ecart =  ",
                              i + j - (int(floor(k / NbColonne)) + k % NbColonne), " N° = ", k,
                              " Nb d'élements similaires = ", l, "\n")
                        l += 1
                        ########################################################
