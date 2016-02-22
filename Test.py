#import Kernel_Estimation_v6 as MonImage
import Mes_Fonctions_v2 as MesFonctions
from random import random
import matplotlib.pyplot as pl
import math                                                                     
import numpy as np
import copy
import opkpy_v3




# -----------------------------------------------------------------------------------
# Quelle image importe-t-on ?
#  --> "essaie" pour une simple image 9x9
#  --> nom de l'image sinon ("test", "chat", etc)
ImageImportee = "essaie"
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# On importe une image
if ImageImportee == "essaie" :
    NbLigne, NbColonne, image_vraie, latent, kernel_vrai, kernel = MesFonctions.mon_image("create")
    floue = MesFonctions.conv(image_vraie, kernel_vrai)
#else :
#    NbLigne, NbColonne, image_vraie, kernel_vrai, floue, kernel, inutile = MonImage.Ouverture("ImageImportee")
# -----------------------------------------------------------------------------------  
 
   
# -----------------------------------------------------------------------------------    
# On declare certains parametres
size = NbLigne*NbColonne
regularization =  "Tikhonov"
dt = 10
# On fait les conversions en vecteur    
floue_v = floue.reshape(size)
kernel_v = kernel.reshape(size)
# On cree la matrice bruit
noise = np.asarray([np.around(random(), decimals=5) for i in range(size)])
C_noise = np.zeros((size, size))
for i in range(0,size):
    C_noise[i,i] = noise[i]*noise[i] 
C_noise_inv = np.linalg.inv(C_noise)
# -----------------------------------------------------------------------------------   


# -----------------------------------------------------------------------------------
# La fonction que l'on souhaite optimiser
def phi(entry):     
# Likelihood fonction
    A = MesFonctions.matrice_A(entry[0:size].reshape((NbLigne,NbColonne)), methode="3")    
    x_ = (floue.reshape(size)-MesFonctions.conv_matrices(A, entry[size:2*size].reshape((NbLigne,NbColonne)), "vecteur","coin"))
    likelihood = np.dot(np.dot(x_,C_noise_inv),x_.transpose()) 
# Regularization term    
    image_ = entry[0:size]   #/sum(sum(im))
    PSF_ = entry[size:2*size]    #/sum(sum(PSF))  
    if(regularization == "spectral"):
        fy_= np.fft(image_)
        y_ = 0
        for i in range(0,size):
            y_ = y_ + i*i*abs(fy_[i])*abs(fy_[i])
        regularization_im = y_           
        fz_= np.fft(PSF_)
        z_ = 0
        for i in range(0,size):
            z_ = z_ + i*i*abs(fz_[i])*abs(fz_[i])
        regularization_PSF = z_
    elif(regularization ==  "Tikhonov"):
         y_ = sum(image_*image_)
         regularization_im = math.sqrt(y_)
         z_ = sum(PSF_*PSF_)
         regularization_PSF = math.sqrt(z_)    
    
    t_ = likelihood + regularization_im + regularization_PSF
    return t_
# -----------------------------------------------------------------------------------
    

# -----------------------------------------------------------------------------------
# On cree le gradient, un vecteur de taille "NombreDeParametres
def grad(entry):
    l = 0
    Retour = np.zeros(2*size)
    phi_originel = phi(entry)
    for i in range(0, 2*size):
        entry_dt = copy.deepcopy(entry)
        entry_dt[i] = entry[i] + dt 
        Retour[i] = (phi(entry_dt) - phi_originel)/dt    
        l = l+1   
    return Retour       
# -----------------------------------------------------------------------------------        
        

# ----------------------------------------------------------------------------------- 
# Calcul le gradient et retourne la valeur de la fonction au point x
def fg(x,gx):
    legradient = grad(x)
    gx[0:size] = legradient[0:size]
    gx[size:2*size] = legradient[size:2*size]
    return phi(x)
# ----------------------------------------------------------------------------------- 



########################## FONCTION DE DEFLOUTAGE D'IMAGE #########################

## On definit notre premier point x (ici l'image floue et un kernel quelconque)
#x = np.concatenate((floue_v, kernel_v), axis=0)
#g = np.concatenate((floue_v, kernel_v), axis=0)
#f = np.array([fg(x,g)], dtype="double")
#
## On execute la fonction de minimisation
#pl.matshow(x[0:size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
#pl.matshow(x[size:2*size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
#opkpy.opk_minimize(x, fg, g, algorithm="nlcg", linesearch="Armijo")
#pl.matshow(x[0:size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)
#pl.matshow(x[size:2*size].reshape((NbLigne,NbColonne)),cmap=pl.cm.gray)

###################################################################################


############################ FONCTION DE ROSENBROCK ############################## 
def Rosenbrock (x):
    return (x[0]-1)*(x[0]-1) + 10*(x[0]*x[0]-x[1]*x[1])*(x[0]*x[0]-x[1]*x[1])

def fg_Rosen (x,gx): 
    gx[0] = 2*(x[0]-1)+40*x[0]*(x[0]*x[0]-x[1]*x[1])
    gx[1] = -40*x[1]*(x[0]*x[0]-x[1]*x[1])
    return Rosenbrock(x)
    
x = np.array([-1.2,1.0], dtype="float32")
g = np.array([12,4], dtype="float32")
f = fg_Rosen(x,g)

print("INPUT:"+str(x))
opkpy.opk_minimize(x, fg_Rosen, g, algorithm="nlcg",linesearch="cubic",nlcg ="HestenesStiefel", bl=0, bu=1e6, limited=1)
print("OUTPUT:"+str(x))
###################################################################################  


