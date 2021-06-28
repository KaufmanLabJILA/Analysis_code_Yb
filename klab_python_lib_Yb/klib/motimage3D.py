from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
pi=np.pi
from scipy.optimize import curve_fit
import scipy.constants as sc
import klib.mako as mako
import klib.Yb_constants as ybc

#class for analyzing MOT Mako images and calculating atom numbers
class motimage3D:

	def __init__(self, texp, filepath, filepathbkg, delta=2*pi*(40e6), Ph=2.2e-3, wh=2.5e-3, Pv=0.8e-3, wv=4.2e-3,
		f=100e-3, lensD=50.8e-3, x_obj=220e-3, c=[0,-1,0,-1],
		output = True):

		self.texp = texp
		self.t = ybc.t1P1
		self.l = self.t.l
		self.gamma0 = self.t.gamma0
		self.Isat = self.t.Isat
		self.delta = delta
		self.Ph = Ph
		self.wh = wh
		self.Pv = Pv
		self.wv = wv
		self.NA = lensD/(2*x_obj)
		self.cal = mako.mako3Dcal # pixel total to number of photons
		self.filepathbkg = filepathbkg
		self.im_bkg1 = np.array(Image.open(filepathbkg), dtype=float)
		self.im = np.array(Image.open(filepath), dtype=float)
		self.c = c
		self.output = output


	# calculate the total peak intensity from all beams
	def getIPeak(self):
		return 4 * (2*self.Ph)/(pi*(self.wh**2)) + 2 * (2*self.Pv)/(pi*(self.wv**2))

	def getRsc(self, s, delta):
		return (self.gamma0/2) * s / ( 1 + s + 4*((delta/self.gamma0)**2) )

	def getImageTotal(self):
		im_sub = self.im - self.im_bkg1
		im_sub_c = im_sub[self.c[0]:self.c[1], self.c[2]:self.c[3]]

		return np.sum(im_sub_c), im_sub_c

	# calculate the atom number
	def atomNumber(self):
		I_peak = self.getIPeak()
		s = I_peak/self.Isat # saturation parameter
		Rsc = self.getRsc(s, self.delta) # single atom scattering rate

		c_frac = 0.5*( 1 - np.sqrt(1 - (self.NA**2)) )
		pixel_total, im_sub_c = self.getImageTotal()
		tot_phot_camera = self.cal*pixel_total
		tot_phot = tot_phot_camera/c_frac

		atom_num = tot_phot/(Rsc*self.texp)

		if (self.output):
			print("exposure time: {:.2e} sec".format(self.texp))
			print("saturation parameter: {:.2f}".format(s))
			print("single atom scattering rate: {:.2e}".format(Rsc))
			print("scattered photons per atom within exposure time: {:.2e}".format(Rsc*self.texp))
			print("number of atoms: {:.2e}".format(atom_num))

			plt.imshow(np.array(im_sub_c), vmin=0, vmax=np.max(im_sub_c))
			plt.colorbar()

		return atom_num
