import numpy as np
import scipy.constants as sc

# mot beam, has waist position r0, waist w0, wavevector k, transition natural linewidth gamma, detuning from resonance gamma (<0 for red detuned)
# saturation parameter s0 = Is/I0 at the waist, polarization pol = +/-1 for right-handed/left-handed beam

class motBeam:

	def __init__(self, k, w0, r0, gamma, delta, s0, pol):
		self.k = k
		self.km = np.sqrt(np.sum(self.k**2))
		self.r0 = r0
		self.w0 = w0
		self.gamma = gamma
		self.delta = delta
		self.s0 = s0
		self.pol = pol
	
	# beam waist as z
	def w(self, z):
		return self.w0*np.sqrt(1+ (z*((2*np.pi)/self.km)/(np.pi*(self.w0**2)))**2 )

	# beam intensity at r, in units of saturation intensity
	def Sat(self, r):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		z = drm*np.cos(theta)
		rho = np.abs(drm*np.sin(theta))
		return self.s0*((self.w(z)/self.w0)**2) * np.exp( (-2*(rho**2))/((self.w(z))**2) )
	
	# calculates the beams average force on an array of atoms at positions r with velocities v for a transition with effective dipole muEff
	# in a quadrapole field with gradient Bp in all directions
	def FBeam(self, r, v, muEff, Bp):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		kdotv = self.k[0]*v[0]+self.k[1]*v[1]+self.k[2]*v[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		Fkir = ((0.5*(1-np.cos(theta)))**2)/(1 + self.Sat(dr) + (2*(self.delta-kdotv+self.pol*muEff*Bp*drm/sc.hbar)/self.gamma)**2)
		Fkil = ((0.5*(1+np.cos(theta)))**2)/(1 + self.Sat(dr) + (2*(self.delta-kdotv-self.pol*muEff*Bp*drm/sc.hbar)/self.gamma)**2)
		Fkitot = (sc.hbar*self.km*self.gamma*self.Sat(dr)/2)*(Fkir + Fkil)
		Fkixyz = np.outer(self.k/self.km, Fkitot)
		return Fkixyz
	

# shell beam is a p=0, l=2 Laguerre-Gaussian mode for now. Not sure what we're going to get in the experiment 
class motBeamShell(motBeam):
	def Sat(self, r):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		z = drm*np.cos(theta)
		rho = np.abs(drm*np.sin(theta))
		return self.s0*np.sqrt(1/np.pi) * (self.w(z)/self.w0) * ((rho*np.sqrt(2)/self.w(z))**2) * np.exp( (-(rho**2))/((self.w(z))**2) ) #* (1-(rho**2)/((self.w(z))**2))
	
class motBeamSquare(motBeam):
	def Sat(self, r):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		z = drm*np.cos(theta)
		rho = np.abs(drm*np.sin(theta))
		return self.s0*( np.heaviside(rho+w0,0.5)*np.heaviside(-rho+w0,0.5) )

# plot the MOT beams
def plotBeams(ax, kis, plotSize, beamLength, colors=['#9FCC3B'], beamWidth=1, **kw):
		if (len(colors) < len(kis)):
			colors=np.repeat(colors,len(kis))
		for i in range(len(kis)):
			ki = kis[i]
			arrow3d(ax, length=beamLength, width=beamWidth, theta_x=np.arccos(ki.k[2]/ki.km), color=colors[i],
					theta_z=np.arctan2(ki.k[0],-ki.k[1]), offset=(-plotSize*ki.k[0]/ki.km+ki.r0[0]*(1e3),
																  -plotSize*ki.k[1]/ki.km+ki.r0[1]*(1e3),
																  -plotSize*ki.k[2]/ki.km+ki.r0[2]*(1e3)), **kw)

#arrow shape for MOT beams
def arrow3d(ax, length=1, width=0.05, head=0.2, headwidth=1,
				theta_x=0, theta_z=0, offset=(0,0,0), **kw):
	w = width
	h = head
	hw = headwidth
#     theta_x = np.deg2rad(theta_x)
#     theta_z = np.deg2rad(theta_z)

	a = [[0,0],[w,0],[w,(1-h)*length],[hw*w,(1-h)*length],[0,length]]
	a = np.array(a)

	r, theta = np.meshgrid(a[:,0], np.linspace(0,2*np.pi,30))
	z = np.tile(a[:,1],r.shape[0]).reshape(r.shape)
	x = r*np.sin(theta)
	y = r*np.cos(theta)

	rot_x = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x) ],
					  [0,np.sin(theta_x) ,np.cos(theta_x) ]])
	rot_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0 ],
					  [np.sin(theta_z) ,np.cos(theta_z),0 ],[0,0,1]])

	b1 = np.dot(rot_x, np.c_[x.flatten(),y.flatten(),z.flatten()].T)
	b2 = np.dot(rot_z, b1)
	b2 = b2.T+np.array(offset)
	x = b2[:,0].reshape(r.shape); 
	y = b2[:,1].reshape(r.shape); 
	z = b2[:,2].reshape(r.shape); 
	ax.plot_surface(x,y,z, **kw)
