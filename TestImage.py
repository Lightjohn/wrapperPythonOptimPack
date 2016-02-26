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
    # IMAGE NETTE
    image_vraie = np.zeros((NbLigne, NbColonne))
    image_vraie[2, 2] = image_vraie[6, 2] = 255
    image_vraie[2, 6] = image_vraie[6, 6] = 255
    # KERNEL QUI LA FLOUTE
    kernel_vrai = np.zeros((NbLigne, NbColonne))
    kernel_vrai[4, 4] = 255
    kernel_vrai[5, 4] = 200
    kernel_vrai[6, 4] = 70
    kernel_vrai[6, 5] = 150
    kernel_vrai_norm = kernel_vrai/sum(sum(kernel_vrai))
    # RESULTAT : L'IMAGE FLOUTEE
    floue = signal.fftconvolve(image_vraie, kernel_vrai_norm, mode="same")
    floue = np.around(floue, decimals=10)
    # KERNEL QUELCONQUE, POINT DE DEPART
    kernel = np.zeros((NbLigne, NbColonne))
    kernel[5, 4] = kernel[4, 5] = kernel[5, 6] = kernel[6, 5] = 100
    kernel[4, 4] = kernel[4, 6] = kernel[6, 4] = kernel[6, 6] = 20
    kernel[5, 5] = 255
    # IMAGE QUELCONQUE, POINT DE DEPART
    latente = np.zeros((NbLigne, NbColonne))
    latente[3,4] = latente[6,2] = 250
# else :
    # On importe une vraie image
#    NbLigne, NbColonne, image_vraie, kernel_vrai, floue, kernel, inutile = MonImage.Ouverture("ImageImportee")
# -----------------------------------------------------------------------------------  


# -----------------------------------------------------------------------------------    
# On declare certains parametres
size = NbLigne * NbColonne
dt = 1
# On fait les conversions en vecteur    
latente_v = latente.reshape(size)
floue_v = floue.reshape(size)
kernel_v = kernel.reshape(size)
# On cree la matrice bruit
noise = np.asarray([np.around(random(), decimals=5) for i in range(size)])
C_noise = np.zeros((size, size))
for i in range(0, size):
    C_noise[i, i] = noise[i] * noise[i]
C_noise_inv = np.linalg.inv(C_noise) / 1000
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# La fonction que l'on souhaite optimiser
def phi(entry):
    # Likelihood fonction
    kernel_norm = entry[size:2*size] / sum(entry[size:2*size])
    HO2 = signal.fftconvolve(entry[0:size], kernel_norm, mode="same")
    HO2 = np.around(HO2, decimals=10)
    x_ = (floue_v - HO2)
    likelihood = np.dot(np.dot(x_, C_noise_inv), x_.transpose())
    # Regularization term
    image_ = entry[0:size]  # /sum(sum(im))
    PSF_ = entry[size:2 * size]  # /sum(sum(PSF))
    y_ = sum(image_ * image_)
    regularization_im = math.sqrt(y_)
    z_ = sum(PSF_ * PSF_)
    regularization_PSF = math.sqrt(z_)

    t_ = likelihood + regularization_im + regularization_PSF
    RETOUR = sum(x_ * x_) 
    return RETOUR
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
# On cree le gradient, un vecteur de taille "NombreDeParametres
def grad_filtre(entry):
    Retour = np.zeros(2 * size)
    fx = np.array([-1, 0, 1])/2
    fy = np.transpose(fx)
    im_convol_x = signal.convolve(entry[0:size], fx, mode = 'same')
    im_convol_y = signal.convolve(entry[0:size], fy, mode = 'same')
    grad_im = np.sqrt(im_convol_x**2 + im_convol_y**2) 
    ker_convol_x = signal.convolve(entry[size:2*size], fx, mode = 'same')
    ker_convol_y = signal.convolve(entry[size:2*size], fy, mode = 'same')
    grad_ker = np.sqrt(ker_convol_x**2 + ker_convol_y**2) 
    
    Retour = np.concatenate((grad_im, grad_ker), axis=0)        
    return Retour
# ----------------------------------------------------------------------------------- 
    

# -----------------------------------------------------------------------------------
# On cree le gradient, un vecteur de taille "NombreDeParametres
def grad_dams(entry):
    
    ft_obj = np.fft.fft(entry[0:size])
    kernel_norm = entry[size:2*size] / sum(entry[size:2*size])  
    ft_psf = np.fft.fft(kernel_norm) 
    
    HO = np.real(np.fft.ifft(ft_psf*ft_obj))
    HO = np.around(HO, decimals=10)
    HO2 = signal.fftconvolve(entry[0:size], kernel_norm, mode="same")
    HO2 = np.around(HO2, decimals=10)
    
    HOminusI = (floue_v - HO2)
    ft_HOminusI_norm =   np.fft.fft(HOminusI/sum(HOminusI))

    grad_ker = np.real(np.fft.ifft(np.conjugate(ft_obj)*ft_HOminusI_norm))
    grad_im = np.real(np.fft.ifft(np.conjugate(ft_psf)*ft_HOminusI_norm))
    
    Retour = np.concatenate((grad_im, grad_ker), axis=0)
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
x = np.concatenate((latente_v, kernel_v), axis=0)
g = np.zeros(2*size)
f = np.array([fg(x,g)], dtype="double")

# On execute la fonction de minimisation
pl.matshow(x[0:size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
pl.matshow(x[size:2*size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
x_out = opkpy_v3.opk_minimize(x, fg, g, algorithm="vmlmb", 
                              linesearch="cubic", vmlmb="lbfgs",
                              limited=0, maxeval=None, maxiter = None, verbose=0)  
                              
image_defloute = x_out[0:size].reshape((NbLigne,NbColonne))
kernel_defloute = x_out[size:2*size].reshape((NbLigne,NbColonne))
test_floue = signal.fftconvolve(image_defloute.reshape(size), kernel_defloute.reshape(size)/sum(sum(kernel_defloute)), mode="same").reshape((NbLigne,NbColonne))
  
  
pl.matshow(image_defloute,cmap=pl.cm.gray)
pl.matshow(kernel_defloute,cmap=pl.cm.gray)
pl.matshow(test_floue,cmap=pl.cm.gray)
pl.matshow(test_floue-floue,cmap=pl.cm.gray)
print "floue : ", sum(sum(abs(test_floue-floue)))/size
print "image : ", sum(sum(abs(image_defloute-image_vraie)))/size
print "kernel : ", sum(sum(abs(kernel_defloute-kernel_vrai)))/size
#################################################################################
