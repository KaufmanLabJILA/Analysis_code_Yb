from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
pi=np.pi
from scipy.optimize import curve_fit
import scipy.constants as sc
import klib.mako as mako
import klib.Yb_constants as ybc

#class for analyzing MOT Mako images and calculating atom numbers
class motimage2D:

	def __init__(self, texp, filepath, filepathbkg, delta=2*pi*(40e6), P=37e-3, w=5.8e-3,
		f=100e-3, lensD=25.4e-3, x_obj=230e-3, bs=0.3, c=[0,-1,0,-1], PSlower=16e-3, wSlower=2.5e-3, deltaSlower=2*pi*(110e6),
		output = True):

		self.texp = texp
		self.t = ybc.t1P1
		self.l = self.t.l
		self.gamma0 = self.t.gamma0
		self.Isat = self.t.Isat
		self.delta = delta
		self.P = P
		self.PSlower = PSlower
		self.wSlower = wSlower
		self.deltaSlower = deltaSlower
		self.w = w
		self.NA = lensD/(2*x_obj)
		self.bs = bs # beam splitter fraction going to camera
		self.cal = mako.mako2Dcal # pixel total to number of photons
		self.filepathbkg = filepathbkg
		self.im_bkg1 = np.array(Image.open(filepathbkg[0]), dtype=float)
		if (len(filepathbkg) == 2):
			self.im_bkg2 = np.array(Image.open(filepathbkg[1]), dtype=float)
		self.im = np.array(Image.open(filepath), dtype=float)
		self.c = c
		self.output = output


	# calculate the total peak intensity from all beams
	def getIPeak(self):
		return 4 * (2*self.P)/(pi*(self.w**2))

	def getIPeakSlower(self):
		return (2*self.PSlower)/(pi*(self.wSlower**2))

	def getRsc(self, s, delta):
		return (self.gamma0/2) * s / ( 1 + s + 4*((delta/self.gamma0)**2) )

	def getImageTotal(self):
		if (len(self.filepathbkg) == 2):
			im_sub = self.im - self.im_bkg1 - self.im_bkg2
		else:
			im_sub = self.im - self.im_bkg1

		#im_sub[im_sub<0] = 0

		im_sub_c = im_sub[self.c[0]:self.c[1], self.c[2]:self.c[3]]

		return np.sum(im_sub_c), im_sub_c

	# calculate the atom number
	def atomNumber(self):
		I_peak = self.getIPeak()
		I_peak_slower = self.getIPeakSlower()
		s = I_peak/self.Isat # saturation parameter
		s_slower = I_peak_slower/self.Isat
		Rsc = self.getRsc(s, self.delta) # single atom scattering rate
		Rsc_slower = self.getRsc(s_slower, self.deltaSlower)
		#print(Rsc, Rsc_slower)

		c_frac = self.bs*0.5*( 1 - np.sqrt(1 - (self.NA**2)) )
		pixel_total, im_sub_c = self.getImageTotal()
		tot_phot_camera = self.cal*pixel_total
		tot_phot = tot_phot_camera/c_frac

		atom_num = tot_phot/((Rsc+Rsc_slower)*self.texp)

		if (self.output):
			print("saturation parameter: {:.2f}".format(s))
			print("saturation parameter slower: {:.2f}".format(s_slower))
			print("single atom scattering rate: {:.2e}".format(Rsc))
			print("scattered photons per atom within exposure time: {:.2e}".format(Rsc*self.texp))
			print("number of atoms: {:.2e}".format(atom_num))

			plt.imshow(np.array(im_sub_c), vmin=0, vmax=np.max(im_sub_c))
			plt.colorbar()

		return atom_num
