import numpy as np
import scipy.constants as sc

# takes atom position and velocity r and v, and a list of mot beams kis, and calculates the force on each atom
def FMOT(m, muEff, Gz, r, v, kis):
	Natoms = len(r[0])
	FMOT = np.zeros((3,Natoms))
	Fg = np.array([np.zeros(Natoms), np.zeros(Natoms), -np.ones(Natoms)*9.8*m])
	stot = np.zeros(Natoms)
	for ki in kis:
		stot = stot + ki.Sat(r)
	for ki in kis:
		FMOT = FMOT + ki.FBeam(r,v,muEff,Gz,stot)
	FMOT = FMOT + Fg
	return FMOT

# computes atom trajectories using RK4: atom positions and velocities are updated over N time steps, each step 
# is the sum of a weighted average of 4 new positions/velocities computed over the step
# all atom positions (and velocities) are in the one numpy array => calculation time not much worse for more atoms
# adding more time steps on the other hand gives linear scaling with this slow python loop below. can look at scipy RK function
# if need a faster calculation or want much smaller time steps
def RK4MC(kis, m, muEff, Gz, r0, v0, dt, N, Natoms):
    rn, vn = np.zeros((N,3,Natoms)), np.zeros((N,3,Natoms))
    rn[0], vn[0] = r0, v0
    for n in range(N-1):
        k1, l1 = dt*FMOT(m, muEff, Gz, rn[n], vn[n], kis)/m,  dt*vn[n]
        k2, l2 = dt*FMOT(m, muEff, Gz, rn[n] + l1/2, vn[n] + k1/2, kis)/m, dt*(vn[n] + k1/2)
        k3, l3 = dt*FMOT(m, muEff, Gz, rn[n] + l2/2, vn[n] + k2/2, kis)/m, dt*(vn[n] + k2/2)
        k4, l4 = dt*FMOT(m, muEff, Gz, rn[n] + l3, vn[n] + k3, kis)/m, dt*(vn[n] + k3)
        rn[n+1] = rn[n] + (1/6)*(l1 + 2*l2 + 2*l3 + l4)
        vn[n+1] = vn[n] + (1/6)*(k1 + 2*k2 + 2*k3 + k4) #+ np.array([np.random.normal(0,vd,Natoms),
                                                         #         np.random.normal(0,vd,Natoms),
                                                          #       np.random.normal(0,vd,Natoms)])/np.sqrt(3)
    cFrac = captureFrac(Natoms,rn)
    print('capture fraction = {:.3f}'.format(cFrac))
    return rn, vn, cFrac

# computes fraction of atoms in the trap after the N time steps are up
def captureFrac(Natoms,rn,trapSize=4e-3):
    Ncapture = 0
    for i in range(Natoms):
        if(np.abs(rn[-1,0,i])<trapSize and np.abs(rn[-1,1,i])<trapSize and np.abs(rn[-1,2,i])<trapSize):
            Ncapture = Ncapture+1
    return Ncapture/Natoms


# mot beam, has waist position r0, waist w0, wavevector k, transition natural linewidth gamma, detuning from resonance gamma (<0 for red detuned)
# saturation parameter s0 = Is/I0 at the waist, polarization pol = +/-1 for right-handed/left-handed beam

class motBeam:

	def __init__(self, k, w0, r0, gamma, delta, s0, pol, l=3, w0x=1, w0y=1, gammaB=1):
		self.k = k
		self.km = np.sqrt(np.sum(self.k**2))
		self.r0 = r0
		self.w0 = w0
		self.gamma = gamma
		self.delta = delta
		self.s0 = s0
		self.pol = pol
		self.l = l
		self.w0x = w0x
		self.w0y = w0y
		self.gammaB = gammaB
	
	# beam waist at z
	def w(self, z):
		return self.w0*np.sqrt(1+ (z*((2*np.pi)/self.km)/(np.pi*(self.w0**2)))**2 )

	# elliptical beam waists at z
	def wEll(self, z):
		wx = self.w0x*np.sqrt(1+ (z*((2*np.pi)/self.km)/(np.pi*(self.w0x**2)))**2 )
		wy = self.w0y*np.sqrt(1+ (z*((2*np.pi)/self.km)/(np.pi*(self.w0y**2)))**2 )
		return [wx, wy]

	# beam intensity at r, in units of saturation intensity
	def Sat(self, r):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		z = drm*np.cos(theta)
		rho = np.abs(drm*np.sin(theta))
		return self.s0*((self.w0/self.w(z))**2) * np.exp( (-2*(rho**2))/((self.w(z))**2) )
	
	def Bquadrapole(self, r, Gz):
		return np.array([-0.5*Gz*r[0], -0.5*Gz*r[1], Gz*r[2]])
	
	# calculates the beam's average force on an array of atoms at positions r with velocities v for a transition with effective dipole muEff
	# in a quadrapole field with gradient Gz along z
	def FBeam(self, r, v, muEff, Gz, stot):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		B = np.array([-0.5*Gz*r[0], -0.5*Gz*r[1], Gz*r[2]])
		Bm = np.sqrt(np.sum(B**2, axis=0))
		kdotB = self.k[0]*B[0]+self.k[1]*B[1]+self.k[2]*B[2]
		kdotv = self.k[0]*v[0]+self.k[1]*v[1]+self.k[2]*v[2]
		theta = np.arccos(kdotB/(Bm*self.km))
		Fkil = ((0.5*(1-np.cos(theta)))**2)/(1 + self.Sat(r) + (2*(self.delta-kdotv+self.pol*muEff*Bm/sc.hbar)/self.gamma)**2)
		Fkipi = 0#(0.5*(1-(np.cos(theta))**2))/(1 + stot + (2*(self.delta-kdotv)/self.gamma)**2)
		Fkir = ((0.5*(1+np.cos(theta)))**2)/(1 + self.Sat(r) + (2*(self.delta-kdotv-self.pol*muEff*Bm/sc.hbar)/self.gamma)**2)
		Fkitot = (sc.hbar*self.km*self.gamma*self.Sat(r)/2)*(Fkir + Fkipi + Fkil)
		Fkixyz = np.outer(self.k/self.km, Fkitot)
		return Fkixyz
	

# shell beam is a p,l Laguerre-Gaussian mode w/ p=0 for now. Not sure what we're going to get in the experiment. 
# l is passed to motBeam, default is l=3
class motBeamShell(motBeam):
	def Sat(self, r):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		z = drm*np.cos(theta)
		rho = np.abs(drm*np.sin(theta))
		return self.s0 * ((self.l/2)**(-self.l/2)) * np.exp(self.l/2) * ((self.w0/self.w(z))**2) * ((rho*np.sqrt(2)/self.w(z))**self.l) * np.exp( (-2*(rho**2))/((self.w(z))**2) ) #* (1-(rho**2)/((self.w(z))**2))
	
class motBeamElliptical(motBeam):
	def Sat(self, r):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		z = drm*np.cos(theta)
		dx, dy = dr[0], dr[1]
		wx, wy = self.wEll(z)
		return self.s0*((self.w0x*self.w0y)/(wx*wy)) * np.exp( (-2*(dx**2))/(wx**2) ) * np.exp( (-2*(dy**2))/(wy**2) )

class motBeamFreqDith(motBeam):
	def FBeam(self, r, v, muEff, Gz, stot):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		B = np.array([-0.5*Gz*r[0], -0.5*Gz*r[1], Gz*r[2]])
		Bm = np.sqrt(np.sum(B**2, axis=0))
		kdotB = self.k[0]*B[0]+self.k[1]*B[1]+self.k[2]*B[2]
		kdotv = self.k[0]*v[0]+self.k[1]*v[1]+self.k[2]*v[2]
		theta = np.arccos(kdotB/(Bm*self.km))
		Fkil = ((0.5*(1-np.cos(theta)))**2)/(1 + self.Sat(r) + (2*(self.delta-kdotv+self.pol*muEff*Bm/sc.hbar)/self.gammaB)**2)
		Fkipi = 0#(0.5*(1-(np.cos(theta))**2))/(1 + stot + (2*(self.delta-kdotv)/self.gamma)**2)
		Fkir = ((0.5*(1+np.cos(theta)))**2)/(1 + self.Sat(r) + (2*(self.delta-kdotv-self.pol*muEff*Bm/sc.hbar)/self.gammaB)**2)
		Fkitot = (sc.hbar*self.km*self.gamma*self.Sat(r)/2)*(Fkir + Fkipi + Fkil)
		Fkixyz = np.outer(self.k/self.km, Fkitot)
		return Fkixyz

class motBeamSquare(motBeam):
	def Sat(self, r):
		dr = np.array([r[0]-self.r0[0], r[1]-self.r0[1], r[2]-self.r0[2]])
		drm = np.sqrt(np.sum(dr**2, axis=0)) + 1e-15
		kdotdr = self.k[0]*dr[0]+self.k[1]*dr[1]+self.k[2]*dr[2]
		theta = np.arccos(kdotdr/(drm*self.km))
		z = drm*np.cos(theta)
		rho = np.abs(drm*np.sin(theta))
		return self.s0*( np.heaviside(-rho+self.w0,0.5) )

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
