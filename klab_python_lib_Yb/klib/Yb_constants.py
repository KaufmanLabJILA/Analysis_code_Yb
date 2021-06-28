#Yb constants for use in analysis notebooks
import numpy as np
import scipy.constants as sc
import klib.transition as t
pi = np.pi

#in SI: (wavelength, natural linewidth, saturation intensity, doppler temp)
t1P1 = t.transition(399e-9, 2*pi*(29e6), 600, 699e-6) #https://www.osapublishing.org/josab/fulltext.cfm?uri=josab-31-10-2302&id=301045
t3P1 = t.transition(555.8e-9, 2*pi*(182.4e3), 1.39, 4.4e-6) #Scazza thesis