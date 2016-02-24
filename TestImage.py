from random import random
import matplotlib.pyplot as pl
import math
import numpy as np
import copy
from scipy import signal

import opkpy_v3


# -----------------------------------------------------------------------------------  
ImageImportee = "essaie"  # "test", "chat", etc
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
if ImageImportee == "essaie":
    # On essaye avec une petite image
    NbLigne = NbColonne = 9
    image_vraie = np.zeros((NbLigne, NbColonne))
    image_vraie[2, 2] = image_vraie[6, 2] = 255
    image_vraie[2, 6] = image_vraie[6, 6] = 255
    kernel_vrai = np.zeros((NbLigne, NbColonne))
    kernel_vrai[4, 4] = 255
    kernel_vrai[5, 4] = 200
    kernel_vrai[6, 4] = 70
    kernel_vrai[6, 5] = 150
    #kernel_vrai = kernel_vrai / sum(kernel_vrai)
    kernel = np.zeros((NbLigne, NbColonne))
    kernel[5, 4] = kernel[4, 5] = kernel[5, 6] = kernel[6, 5] = 100
    kernel[4, 4] = kernel[4, 6] = kernel[6, 4] = kernel[6, 6] = 20
    kernel[5, 5] = 255
    #kernel = kernel / sum(kernel)
    floue = signal.fftconvolve(image_vraie, kernel_vrai, mode="same")
# else :
    # On importe une vraie image
#    NbLigne, NbColonne, image_vraie, kernel_vrai, floue, kernel, inutile = MonImage.Ouverture("ImageImportee")
# -----------------------------------------------------------------------------------  


# -----------------------------------------------------------------------------------    
# On declare certains parametres
size = NbLigne * NbColonne
dt = 10
# On fait les conversions en vecteur    
floue_v = floue.reshape(size)
kernel_v = kernel.reshape(size)
# On cree la matrice bruit
noise = np.asarray([np.around(random(), decimals=5) for i in range(size)])
C_noise = np.zeros((size, size))
for i in range(0, size):
    C_noise[i, i] = noise[i] * noise[i]
C_noise_inv = np.linalg.inv(C_noise)
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# La fonction que l'on souhaite optimiser
def phi(entry):
    # Likelihood fonction
    x_ = (floue_v - signal.fftconvolve(entry[0:size], entry[size:2*size], mode="same"))
    likelihood = np.dot(np.dot(x_, C_noise_inv), x_.transpose())
    # Regularization term
    image_ = entry[0:size]  # /sum(sum(im))
    PSF_ = entry[size:2 * size]  # /sum(sum(PSF))
    y_ = sum(image_ * image_)
    regularization_im = math.sqrt(y_)
    z_ = sum(PSF_ * PSF_)
    regularization_PSF = math.sqrt(z_)

    t_ = likelihood + regularization_im + regularization_PSF
    return t_
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# On cree le gradient, un vecteur de taille "NombreDeParametres
def grad(entry):
    Retour = np.zeros(2 * size)
    phi_originel = phi(entry)
    for i in range(0, 2 * size):
        entry_dt = copy.deepcopy(entry)
        entry_dt[i] = entry[i] + dt
        Retour[i] = (phi(entry_dt) - phi_originel) / dt
    return Retour
# -----------------------------------------------------------------------------------        


# ----------------------------------------------------------------------------------- 
# Calcul le gradient et retourne la valeur de la fonction au point x
def fg(x, gx):
    legradient = grad(x)
    gx[0:size] = legradient[0:size]
    gx[size:2 * size] = legradient[size:2 * size]
    return phi(x)
# -----------------------------------------------------------------------------------



######################### FONCTION DE DEFLOUTAGE D'IMAGE #########################

# On definit notre premier point x (ici l'image floue et un kernel quelconque)
x = np.concatenate((kernel_v, kernel_v), axis=0)
g = np.concatenate((kernel_v, kernel_v), axis=0)
f = np.array([fg(x,g)], dtype="double")

## On execute la fonction de minimisation
pl.matshow(x[0:size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
pl.matshow(x[size:2*size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
x_out = opkpy_v3.opk_minimize(x, fg, g, algorithm="nlcg", 
                              linesearch="cubic", nlcg="HagerZhang",
                              limited=0, maxeval=50)      
pl.matshow(x[0:size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
pl.matshow(x[size:2*size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)

#################################################################################
