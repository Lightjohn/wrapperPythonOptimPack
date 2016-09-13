from random import random
from matplotlib.pyplot import *
import math
import copy
import numpy as np
from scipy import signal
from PIL import Image

import opkpy_v3



# ----------------------------------------------------------------------------
psf_init = "blind"
ImageImportee = "essaie"  # "essaie", test", "thiebaut", etc
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
if ImageImportee == "essaie":
    # On essaye avec une petite image
    a = b = 9
    # IMAGE NETTE
    objet_mat = np.zeros((a, b))
    objet_mat[2, 2] = objet_mat[6, 2] = 255
    objet_mat[2, 6] = objet_mat[6, 6] = 255
    # psf QUI LA FLOUTE
    kernel_mat = np.zeros((a, b))
    kernel_mat[4, 4] = 255
    kernel_mat[5, 4] = 200
    kernel_mat[6, 4] = 70
    kernel_mat[6, 5] = 150
    kernel_mat = kernel_mat/sum(sum(kernel_mat))

else :
    ############### On ouvre l'objet #################
    fichier = "/obs/gheurtier/Defloutage/Images/" + ImageImportee + "_nette.png"
    im = Image.open(fichier, "r")
    image_liste = list(im.getdata())     
    a,b = im.size  
    objet_mat = np.zeros((b,a))
    for i in range(0, a*b):
        objet_mat[np.floor(i/a), i%a] = image_liste[i][0]
      
    ################# On ouvre la psf ################
    fichier = "/obs/gheurtier/Defloutage/Images/" + ImageImportee + "_ker.png"
    ke = Image.open(fichier, "r")       
    kernel_liste = list(ke.getdata())     
    c,d = ke.size  
    kernel_mat = np.zeros((b,a))
    for i in range(0, c*d):
        kernel_mat[np.floor(i/c), i%c] = kernel_liste[i][0]
    kernel_mean =  np.mean(kernel_mat)
    kernel_mat[kernel_mat < 3*kernel_mean] = 0
    kernel_mat = kernel_mat/sum(sum(kernel_mat))

############# On definit les tailles ##############
dt = np.array([10,0], dtype="double")
NbLigne = b
NbColonne = a
size = NbLigne * NbColonne
if (NbLigne != NbColonne):
    print" WARNING: Object must be square"
    
###### On definit le bruit et sa variance #########
noise = np.asarray([np.around(random(), decimals=8) for i in range(size)])
noise_mat = noise.reshape((NbLigne, NbColonne))
C_noise = np.zeros((size, size))
#for i in range(0, size):
#    C_noise[i, i] = noise[i] * noise[i]
#C_noise_inv = np.linalg.inv(C_noise) / 1000
variance = 0
noise_mean = np.mean(noise)
for i in range (0,size):
    variance = variance + (noise[i] - noise_mean) * (noise[i] - noise_mean)
variance /= size

############# On definit les vecteurs #############
objet = objet_mat.reshape(size)
kernel = kernel_mat.reshape(size)
ft_objet = np.fft.fft(objet)
ft_kernel = np.fft.fft(np.roll(kernel,np.round((size+1)/2)))
image = np.real(np.fft.ifft(ft_objet * ft_kernel)) 
image[image < 1e-12] = 0
ft_image = np.fft.fft(image)
image_mat = image.reshape((NbLigne,NbColonne))

################# On definit psf ##################
image_mean = np.mean(image)
psf = np.zeros(size)
if psf_init == "carre":
    psf[49] = psf[41] = psf[51] = psf[59] = 100
    psf[40] = psf[42] = psf[58] = psf[60] = 20
    psf[50] = 255
elif psf_init == "blind":
    psf[0:size] = image[0:size]    
    psf[psf < 2*image_mean] = 0
else:
    psf[0:size] = kernel[0:size]
#psf = psf / sum(psf)
psf_mat = psf.reshape((NbLigne,NbColonne))

################# On affiche ######################
#matshow(objet_mat,cmap=cm.gray)
#matshow(kernel_mat,cmap=cm.gray)
#matshow(image_mat,cmap=cm.gray)
#matshow(psf_mat,cmap=cm.gray)

############## On definit les noms ################
"""
--> objet = l'objet de base, on ne l'utilise pas
--> kernel = le kernel de base, on ne l'utilise pas, sauf si set blind = False.
dans ce cas kernel = psf
--> image = le resultat de la convolution d'objet et image kernel (avec kernel
recentre). On s'en sert comme objet au debut de l'algorithme et pour estimer
la valeur initiale de psf
--> psf = la psf qui varie dans l'ago. Si blind = True, psf est initialise
en filtrant les basses frequences d'image.
"""
# ----------------------------------------------------------------------------
  

  
# ----------------------------------------------------------------------------
def phi(x):

    ft_obj = np.fft.fft(x[0:size])
    psf_norm = x[size:2*size] / sum(x[size:2*size])  
    ft_psf = np.fft.fft(np.roll(psf_norm,np.round((size+1)/2))) 
    
    HO = np.real(np.fft.ifft(ft_psf*ft_obj))
    HO = np.around(HO, decimals=10)
    HOminusI = (HO - image)
    
    return sum(HOminusI * HOminusI)
   
def grad_damien(entry):
    Retour = np.zeros(2 * size)
    ft_obj = np.fft.fft(x[0:size])
    psf_norm = x[size:2*size] / sum(x[size:2*size])  
    ft_psf = np.fft.fft(np.roll(psf_norm,np.round((size+1)/2)))  
    HO = np.real(np.fft.ifft(ft_psf*ft_obj))
    HO = np.around(HO, decimals=10)
    HOminusI = (HO - image)
    
    ft_HOminusI_norm =   np.roll(np.fft.fft(HOminusI/sum(HOminusI)),(size+1)/2)
    Retour[size:2 * size] = np.real(np.fft.ifft(np.conjugate(ft_obj)*ft_HOminusI_norm))
    Retour[0:size] = np.real(np.fft.ifft(np.conjugate(ft_psf)*ft_HOminusI_norm))
    return Retour   
   
def grad_dt(entry):
    dt[0] *= 0.99
    Retour = np.zeros(2 * size)
    phi_originel = phi(entry)
    for i in range(0, 2 * size):
        entry_dt = copy.deepcopy(entry)
        entry_dt[i] = entry[i] + dt[0]
        Retour[i] = (phi(entry_dt) - phi_originel) / dt[0]
    return Retour
    
def fg(x, gx):
    legradient = grad_dt(x)
    gx[0:size] = legradient[0:size]
    gx[size:2 * size] = legradient[size:2 * size]
    return phi(x)    
    
###################### FONCTION DE DEFLOUTAGE D'IMAGE ########################

# On definit notre premier point x (ici l'image floue et un kernel quelconque)
x = np.concatenate((image, psf), axis=0)
g = np.zeros(2*size)
f = np.array([fg(x,g)], dtype="double")
print "f initial = ", f
# On execute la fonction de minimisation
x_out = opkpy_v3.opk_minimize(x, fg, g, algorithm="vmlmb", 
                              linesearch="cubic", vmlmb="lbfgs",
                              limited=0, maxeval=200, maxiter = 200)  
                            
                            
objet_defloute = x_out[0:size].reshape((NbLigne,NbColonne))
kernel_defloute = x_out[size:2*size].reshape((NbLigne,NbColonne))
image_out = np.real(np.fft.ifft(np.fft.fft(x_out[0:size]) * np.fft.fft(np.roll(x_out[size:2*size]/sum(x_out[size:2*size]),np.round((size+1)/2))))).reshape((NbLigne,NbColonne)) 

# objets
matshow(image_mat,cmap=cm.gray)          # point de depart
matshow(objet_mat,cmap=cm.gray)          # objectif
matshow(objet_defloute,cmap=cm.gray)     # resultat
# psf
matshow(psf_mat,cmap=cm.gray)
matshow(kernel_mat,cmap=cm.gray)
matshow(kernel_defloute,cmap=cm.gray)
# image
matshow(image_mat,cmap=cm.gray)
matshow(image_out,cmap=cm.gray)
print "objet : ", sum(sum(abs(objet_defloute-objet_mat)))/size
print "psf : ", sum(sum(abs(kernel_defloute-kernel_mat)))/size
print "image : ", sum(sum(abs(image_out-image_mat)))/size
################################################################################
 
    
    