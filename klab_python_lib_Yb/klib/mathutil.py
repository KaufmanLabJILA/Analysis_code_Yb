from .imports import *

# from .expfile import *
# from .analysis import *
# from .mathutil import *
# from .plotutil import *
# from .imagutil import *
# from .mako import *
# from .adam import *
# Math tools for klablib

def aom_phase_chirp(x, s1, s2, yinf):
    return yinf*(1 + s1/x + s2*x )

def beam_waist(z,z0,zr,w0,lam):
#     zr=np.pi*w0**2/lam
    return w0*np.sqrt(1+((z-z0)/zr)**2)

def const(x,a):
    return a

def cos(t, f, A, phi, y0):
    return abs(A/2)*np.cos(2*np.pi*f*t+phi) + y0

def cosRam(t, f, yup, phi, ydown):
    return abs((yup-ydown)/2)*np.cos(2*np.pi*f*t+phi) + (yup+ydown)/2

def cosFit(keyVals, dat, n = 0, ic = False):
    """Cosine fit. Parameter order: [f. A, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)

    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)

        A = (ymax-ymin)/2
        f = .1/(xdat[1]-xdat[0])
        phi = np.pi
        y0 = np.mean(ydat)

        guess = [f, A, phi, y0]
    else:
        guess = ic

#     print(guess)
    params, uncert = curve_fit(cos, xdat, ydat, p0=guess, maxfev=999999)

#     cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     plt.plot(xdat, cos(xdat, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr

def cosFitF(keyVals, dat, f, n = 0, ic = False, bd = False):
    """Cosine fit. Parameter order: [f, A, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)

    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)

        A = np.abs((ymax-ymin)/2)
#         f = 1/(xdat[1]-xdat[0])
        phi = 0
        y0 = np.mean(ydat)

        guess = [A, phi, y0]
    else:
        guess = ic

    if not bd:
        bound_lower = [0, -1.5*np.pi, -100]
        bound_upper = [1000, 1.5*np.pi, 100]
    else:
        bound_lower = bd[0]
        bound_upper = bd[1]

#     print(guess,bound_lower,bound_upper)
    params, uncert = curve_fit(lambda x, amp, phase, y: cos(x, f, amp, phase, y), xdat, ydat, p0=guess, bounds = (bound_lower, bound_upper), maxfev = 9999)

#     cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     plt.plot(xdat, cos(xdat, f, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr

def cosFitFA(keyVals, dat, f, A, n = 0, ic = False, bd = False):
    """Cosine fit. Parameter order: [f, A, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)

    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)
#         f = 1/(xdat[1]-xdat[0])
        phi = 0
        y0 = np.mean(ydat)

        guess = [phi, y0]
    else:
        guess = ic

    if not bd:
        bound_lower = [-1.5*np.pi, -100]
        bound_upper = [1.5*np.pi, 100]
    else:
        bound_lower = bd[0]
        bound_upper = bd[1]

#     print(guess,bound_lower,bound_upper)
    params, uncert = curve_fit(lambda x, phase, y: cos(x, f, A, phase, y), xdat, ydat, p0=guess, bounds = (bound_lower, bound_upper), maxfev = 9999)

#     cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     plt.plot(xdat, cos(xdat, f, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr

def dampedCos(t, A, tau, f, phi, y0):
    return A*np.exp(-t/tau)/2 * (np.cos(2*np.pi*f*t+phi)) + y0

def dampedCosFit(keyVals, dat, n = 0, ic = False, plotOut = True):
    """Damped cosine fit. Parameter order: [A, tau, f, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)

    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)

        A = ymax-ymin
        tau = np.mean(xdat)
        f = 1/(xdat[1]-xdat[0])
        phi = 0
        y0 = np.mean(ydat)

        guess = [A, tau, f, phi, y0]
    else:
        guess = ic

#     print(guess)
    params, uncert = curve_fit(dampedCos, xdat, ydat, p0=guess, maxfev=999999)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if plotOut:
        plt.plot(xdat, dampedCos(xdat, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr

def decay(t, gamma, a, y0):
    """Exponential decay [gamma, amp, offset]"""
    return y0 + a*np.exp(-gamma*t)

def decayt(t, tau, a, y0):
    """Exponential decay [tau, amp, offset]"""
    return y0 + a*np.exp(-t/tau)

def decayt_gaussian(t, tau, a, y0):
    return y0 + a*np.exp(-(t/tau)**2)

def decayt_gaussian0(t, tau, a):
    return a*np.exp(-(t/tau)**2)

def decayt0(t, tau, a):
    """Exponential decay [tau, amp]"""
    return a*np.exp(-t/tau)

def decayt1(t, tau, a):
    """Exponential decay [tau, amp]"""
    return a*np.exp(-t/tau) + 1

def depol(x,traman,tau):
    """Fit depolarization signal for pumped-away signal. Params: [traman,tau]"""
    return (1/2*np.exp(-x/tau))*(1-np.exp(-2*x/traman))

def depolknowntau(x,traman):
    """Fit depolarization signal for pumped-away signal. Params: [traman]"""
    return (1/2*np.exp(-(x+30)/5611))*(1-np.exp(-2*(x+30)/traman))

def depoloppknowntau(x,traman):
    """Fit depolarization signal for pumped-to signal. Params: [traman]"""
    return np.exp(-(x+30)/5611)-(1/2*np.exp(-(x+30)/5611))*(1-np.exp(-2*(x+30)/traman))

def depolimpureknowntau(x,traman,p):
    """Fit depolarization for pumped-away signal, allowing imperfect initial spin purity. Params: [traman,prob]"""
    return (1/2*np.exp(-(x+30)*(1/6070+2/traman)))*(-1+2*p+np.exp(2*(x+30)/traman))

def depolheuristicloss(x,t1):
    """Fit depolarization for pumped-to signal, with heuristic atom loss eqn to match PRX paper. Params: [t1]"""
    a = 5.428e-4
    b = -10.096e-9
    return (1/2*(np.exp(-a*x-b*x*x)-np.exp(-a*x-b*x*x-2*x/t1)))

def depoloppheuristicloss(x,t1):
    """Fit depolarization for pumped-to signal, with heuristic atom loss eqn to match PRX paper. Params: [t1]"""
    a = 1.562e-4
    b = 2.599e-9
    return (1/2*(np.exp(-a*x-b*x*x)+np.exp(-a*x-b*x*x-2*x/t1)))

def depolimpureheuristicloss(x,t1,p):
    """Fit depolarization for pumped-to signal, with heuristic atom loss eqn to match PRX paper. Params: [t1,p]"""
    a = 5.428e-4
    b = -10.096e-9
    return (1/2*(np.exp(-a*x-b*x*x)-(1-2*p)*np.exp(-a*x-b*x*x-2*x/t1)))

def double_gaussian(x, *p):
    return p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )

def emccd_bkg(x, alpha, x0, beta, A):
    return A*gamma.pdf(x, loc=x0, a=alpha, scale=1./beta)*gammaf(alpha)/( (beta**alpha) * (((alpha-1)/beta)**(alpha-1)) * np.exp(-(alpha-1)) )

def emccd_hist(x, alpha, x0, scale, A, a, x1, sig):
    return emccd_bkg(x, alpha, x0, scale, A) + gaussian(x, a, x1, sig, 0)

def emccd_hist_skew(x, alpha, x0, scale, A, a, x1, sig, beta):
    return emccd_bkg(x, alpha, x0, scale, A) + gaussian_skew(x, a, x1, sig, beta, 0)

def erfc(x, amp, x0, sigma):
    if x < x0:
        return amp * np.sqrt(np.pi*(sigma**2)/2) * (1 + erf((x-x0)/np.sqrt(2)/np.abs(sigma)))
    if x >= x0:
        return amp * np.sqrt(np.pi*(sigma**2)/2) * (1 - erf((x-x0)/np.sqrt(2)/np.abs(sigma)))

def expfit(t, A, tau):
    return A*np.exp(-t/tau)

def expatomloss(x,a,b,Amp):
    """Fit atom loss from the trap, using heuristic function. Param: [a, b, Amp]"""
    return (Amp*np.exp(-a*x-b*x*x))

def fivelor(x, a0, a1, a2, a3, a4, kc, ks, kss, x0, dx, dxx, y0):
    return y0 + lor(x, a0, kss, x0-dxx)+ lor(x, a1, ks, x0-dx)+lor(x, a2, kc, x0) + lor(x, a3, ks, x0+dx) +lor(x, a4, kss, x0+dxx)

def gaussian(x, a, x0, sig, y0):
    return a * np.exp( -((x - x0)**2) / (2*(sig**2)) ) + y0

def gaussian_skew(x, a, x0, sig, alpha, y0):
    return a*np.sqrt(2*np.pi) * (1/np.sqrt(2*np.pi) * np.exp( -((x - x0)**2) / (2*(sig**2)) )*(1+erf(alpha*(x-x0)/sig/np.sqrt(2)))) + y0

def gaussianBeam(x, x0, a, waist, y0):
    """Math: 1D Gaussian function, optics definition. params [x0, amp, waist, y0]"""
    return a*np.exp(-(2*(x-x0)**2)/(waist**2))+y0

def gaussianBeam0(x, x0, a, waist):
    """Math: 1D Gaussian function with zero offset, optics definition. params [x0, amp, waist]"""
    return a*np.exp(-(2*(x-x0)**2)/(waist**2))

def gaussianBeam07MHz(x, x0, waist):
    return 7.1*np.exp(-(2*(x-x0)**2)/(waist**2))

def gaussianNormal(x, x0, a, sig, y0):
    """Math: 1D Gaussian function, statistics definition. params [x0, amp, sig, y0]"""
    return a*np.exp(-((x-x0)**2)/(2*sig**2))+y0

def gausCos(t, A, sig, f, phi, y0):
    return (A/2)*np.exp(-(t/sig)**2/2) * (np.cos(2*np.pi*f*t+phi)) + y0

def gausCosFit(keyVals, dat, n = 0, ic = False, plotOut = True):
    """Damped cosine fit. Parameter order: [A, tau, f, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)

    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)

        A = ymax-ymin
        tau = np.mean(xdat)
        f = 1/(xdat[1]-xdat[0])
        phi = 0
        y0 = np.mean(ydat)
        sig=tau

        guess = [A, sig, f, phi, y0]
    else:
        guess = ic

#     print(guess)
    params, uncert = curve_fit(gausCos, xdat, ydat, p0=guess, maxfev=999999)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if plotOut:
        plt.plot(xdat, gausCos(xdat, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr

def gausfitNormal(keyVals, dat, y_offset=False, negative=False, n=0, guess = []):
    """1D gaussian fit, with or without Y-offset, and with positive or negative amplitude. Can select nth color from default matplotlib color cycle. returns gausparams [x0, a, sigma, y0], perr"""
    xdat=keyVals
    ydat=arr(dat)
    if negative:
        i=ydat.argmin()
        a = -abs(ydat[i]-max(ydat))
        ihalf=np.argmin(np.abs(ydat-np.min(ydat)-(np.max(ydat)-np.min(ydat))/2)) #find position of half-maximum

    else:
        i=ydat.argmax()
        a = ydat[i]
        ihalf=np.argmin(np.abs(ydat-(np.max(ydat)-np.min(ydat))/2)) #find position of half-maximum

    x0 = xdat[i]
    y0 = ydat[0]
    #sig = np.abs(xdat[i]-xdat[ihalf])
    sig = (keyVals[-1]-keyVals[0])/5
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     plt.plot(xdat, ydat,".", color=cycle[n])

    if y_offset:
        if len(guess) == 0:
            guess = [x0, a, sig, y0]
#         print(guess)
        gauss_params, gauss_uncert = curve_fit(gaussianNormal, xdat, ydat, p0=guess, maxfev=10000, bounds = ([keyVals.min(), 0, 0, -np.inf], [keyVals.max(), np.inf, np.inf, np.inf]))
#         plt.plot(xdat, gaussian(xdat, *gauss_params), "-", color=cycle[n])
    else:
        if len(guess) == 0:
            guess = [x0, a, sig]
#         print(guess)
        gauss_params, gauss_uncert = curve_fit(lambda x, x0, a, sig: gaussianNormal(x, x0, a, sig, 0), xdat, ydat, p0=guess, maxfev=100000, bounds = ([keyVals.min(), 0, 0], [keyVals.max(), np.inf, np.inf]))
#         print(gauss_params)
#         plt.plot(xdat, gaussian(xdat, *gauss_params, 0), "-", color=cycle[n])

    perr = np.sqrt(np.diag(gauss_uncert))

    # plt.plot(xdat, lorentz(xdat, *lorentz_params), "r")
#     plt.xlabel('Modulation freq (MHz)')
#     plt.ylabel('ROI sum (arb)')
#     plt.show()

#     df=pd.DataFrame([gauss_params])
#     df.columns=['x0','a','sig']
#     df

    return gauss_params, perr

def gausfit(keyVals, dat, y_offset=False, negative=False, n=0, guess = []):
    """1D gaussian fit, with or without Y-offset, and with positive or negative amplitude. Can select nth color from default matplotlib color cycle. returns gausparams [x0, a, waist, y0], perr"""
    xdat=keyVals
    ydat=arr(dat)
    if negative:
        i=ydat.argmin()
        a = -abs(ydat[i]-max(ydat))
        ihalf=np.argmin(np.abs(ydat-np.min(ydat)-(np.max(ydat)-np.min(ydat))/2)) #find position of half-maximum

    else:
        i=ydat.argmax()
        a = ydat[i]
        ihalf=np.argmin(np.abs(ydat-(np.max(ydat)-np.min(ydat))/2)) #find position of half-maximum

    x0 = xdat[i]
    y0 = ydat[0]
    #sig = np.abs(xdat[i]-xdat[ihalf])
    sig = (keyVals[-1]-keyVals[0])/5
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     plt.plot(xdat, ydat,".", color=cycle[n])

    if y_offset:
        if len(guess) == 0:
            guess = [x0, a, sig, y0]
#         print(guess)
        gauss_params, gauss_uncert = curve_fit(gaussian, xdat, ydat, p0=guess, maxfev=10000, bounds = ([keyVals.min(), 0, 0, -np.inf], [keyVals.max(), np.inf, np.inf, np.inf]))
#         plt.plot(xdat, gaussian(xdat, *gauss_params), "-", color=cycle[n])
    else:
        if len(guess) == 0:
            guess = [x0, a, sig]
#         print(guess)
        gauss_params, gauss_uncert = curve_fit(lambda x, x0, a, sig: gaussian(x, x0, a, sig, 0), xdat, ydat, p0=guess, maxfev=100000, bounds = ([keyVals.min(), 0, 0], [keyVals.max(), np.inf, np.inf]))
#         print(gauss_params)
#         plt.plot(xdat, gaussian(xdat, *gauss_params, 0), "-", color=cycle[n])

    perr = np.sqrt(np.diag(gauss_uncert))

    # plt.plot(xdat, lorentz(xdat, *lorentz_params), "r")
#     plt.xlabel('Modulation freq (MHz)')
#     plt.ylabel('ROI sum (arb)')
#     plt.show()

#     df=pd.DataFrame([gauss_params])
#     df.columns=['x0','a','sig']
#     df

    return gauss_params, perr


def gauss2d(xy, amp, x0, y0, theta, sig_x, sig_y):
    """Math: 2D Gaussian"""
    x, y =xy

    a = np.cos(theta)**2/(2*sig_x**2) + np.sin(theta)**2/(2*sig_y**2);
    b = -np.sin(2*theta)/(4*sig_x**2) + np.sin(2*theta)/(4*sig_y**2);
    c = np.sin(theta)**2/(2*sig_x**2) + np.cos(theta)**2/(2*sig_y**2);

    return amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0)**2 * (y - y0)**2 + c * (y - y0)**2))

def gaussFit2d(datc):
    """2D Gaussian fit to image matrix. params [amp, x0, y0, theta, w_x, w_y]"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    i = datf.argmax()
    ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximum
    sig_x_guess = np.abs(x[i]-x[ihalf])+1
    sig_y_guess = np.abs(y[i]-y[ihalf])+1
    print(sig_x_guess,sig_y_guess)
    guess = [datf[i], x[i], y[i], 0, sig_x_guess, sig_y_guess]
    pred_params, uncert_cov = curve_fit(gauss2d, xy, datf, p0=guess, maxfev=100000)

    zpred = gauss2d(xy, *pred_params)
    #print('Predicted params:', pred_params)
    print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params

def gaussianBeam1D(xy, I0, x0, y0, w0, a0):
    """Math: 2D Gaussian"""
    x, y =xy
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    return I0 * np.exp(-2 * r**2 / w0**2) + a0

def gaussianBeamFit(datc,verbose=False):
    """2D Gaussian fit to image matrix. params [I0, x0, y0, w0, offset]"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    i = datf.argmax()
    ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximumsync
    w0_guess = np.abs(x[i]-x[ihalf])+1
    offset = np.mean(datc)
    if verbose:
        print(w0_guess)
    guess = [datf[i], x[i], y[i], w0_guess, offset]
    pred_params, uncert_cov = curve_fit(gaussianBeam, xy, datf, p0=guess, maxfev=100000)

    zpred = gaussianBeam(xy, *pred_params)
    if verbose:
        print('Predicted params:', pred_params)
        print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params

def gaussianBeam2D(xy, amp, x0, y0, theta, wx, wy, z0):
    """Math: 2D Gaussian w/ factor of two for optics formalism. params [amp, x0, y0, theta, wx, wy]"""
    x, y =xy

    a = 2*np.cos(theta)**2/(wx**2) + 2*np.sin(theta)**2/(wy**2);
    b = -np.sin(2*theta)/(wx**2) + np.sin(2*theta)/(wy**2);
    c = 2*np.sin(theta)**2/(wx**2) + 2*np.cos(theta)**2/(wy**2);

    return z0+amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))

def gaussianBeamFit2D(datc, auto = True, mguess = []):
    """2D Gaussian fit to image matrix. params [amp, x0, y0, theta, w_x, w_y, z0]. Returns fitdata, params, perr"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    if auto:
        i = datf.argmax()
        ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximum
        wx_guess = np.abs(x[i]-x[ihalf])+1
        wy_guess = np.abs(y[i]-y[ihalf])+1
#         print(wx_guess,wy_guess)
        guess = [datf[i], x[i], y[i], 0, wx_guess, wy_guess, 0]
    else:
        guess = mguess
    pred_params, uncert_cov = curve_fit(gaussianBeam2D, xy, datf, p0=guess, maxfev=1000, bounds = ([0,0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.pi/2,np.inf,np.inf,np.inf]))

    perr = np.sqrt(np.diag(uncert_cov))
    zpred = gaussianBeam2D(xy, *pred_params)
    #print('Predicted params:', pred_params)
#     print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params, perr

def hockey_fall(x_arr, x0, rate, offset):
    y_arr =[]
    for x in x_arr:
        if x < x0:
            y_arr.append(offset)
        else:
            y_arr.append(offset - rate*(x-x0))
    return y_arr

def hockey_rise(x_arr, x0, rate, offset):
    y_arr =[]
    for x in x_arr:
        if x > x0:
            y_arr.append(offset)
        else:
            y_arr.append(rate*(x-x0)+offset)
    return y_arr


def line(x, a, b):
    return a*x+b

def linex(x, x0, a, b, c, d):
    return a*np.exp(b*(x-x0)) + c*(x-x0) +d

def v_fit(x, x0, a, b):
    return a*abs(x-x0)+b

def lorentz(x, x0, a, sig, y0):
    """Math: 1D Lorentz function"""
    return a*(sig/2)/((x-x0)**2+(sig/2)**2)+y0

def lor(x, a, k, x0):
    return a/(1+((x-x0)/(k/2))**2)

def Omega(n,m,Omega0,eta):
    s = max(n,m)-min(n,m)
    Omega = Omega0*np.exp(-eta**2/2)*eta**s*np.sqrt(math.factorial(max(n,m))/math.factorial(min(n,m)))*scipy.special.genlaguerre(min(n,m),s)(eta**2)
    return Omega

def quadgaus(x, a0, a1, a2, a3, sc, ss, s3, x0, dx1, dx2, x3, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+gaussianNormal(x, x0-dx1, a0, ss, 0)+gaussianNormal(x, x0, a1, sc, 0) + gaussianNormal(x, x0+dx2, a2, ss, 0) + gaussianNormal(x, x3, a3, s3, 0)

def quintgaus(x, a0, a1, a2, a3, a4, sc, ss, s3, x0, dx1, dx2, dx3, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+gaussianNormal(x, x0-dx1, a0, ss, 0)+gaussianNormal(x, x0, a1, sc, 0) + gaussianNormal(x, x0+dx2, a2, ss, 0) + gaussianNormal(x, x0-dx3, a3, s3, 0)+ gaussianNormal(x, x0+dx30, a4, s3, 0)

def radial_profile(data, center):
    """Returns radial average of matrix about user-defined center."""
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def ramanRabi_pol(x, theta, T, A, x0):
    """Fit Raman Rabi X qubit rate, assuming perfect linear polarization but imperfect orthogonality b/w k and B"""
    return np.abs(A*np.sin((x-x0)/T)*np.cos(theta)*np.sqrt(np.cos((x-x0)/T)**2+(np.sin((x-x0)/T)*np.sin(theta))**2))

def sin_rect(x, T, A, x0):
    """Fit Raman Rabi X qubit rate, assuming perfect polarization and perfect orthogonality b/w k and B"""
    return np.abs(A*np.sin(2*3.14159265*(x-x0)/T))

def sinc2(x, x0, a0, y0, k0):
    return a0*np.sinc((x-x0)/k0)**2 + y0

def square(x, x0, a, y0):
    return a*(x-x0)**2 + y0

def ramsey(d, d0, a0, y0, Omega0, T, tau):
    Delta = (d-d0)
    Omega = np.sqrt(Omega0**2 + Delta**2)
    return a0* 4 * (Omega0/Omega)**2 * np.sin(2*np.pi*Omega*tau/2)**2 * ( np.cos(2*np.pi*Delta*T/2)*np.cos(2*np.pi*Omega*tau/2) - Delta/Omega * np.sin(2*np.pi*Delta*T/2)*np.sin(2*np.pi*Omega*tau/2 ) )**2 + y0


def rabi_pi(d, d0, a0, y0, Omega):
    return a0*Omega**2/(Omega**2+(d-d0)**2)*np.sin(np.sqrt((d-d0)**2+Omega**2)*2*3.14159265 / Omega / 4)**2 + y0

def Thermal_dephase(t, nbar, Omega0, A, y0, eta=0.33):
    P = 0
    for n in range(10):
        Pn = nbar**n/(1+nbar)**(n+1)
        P += A*Pn/2*(1-np.cos(Omega(n,n,Omega0,eta)*t)) + y0
    return P

def triplor(x, a0, a1, a2, kc, ks, x0, dx, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+lor(x, a0, ks, x0-dx)+lor(x, a1, kc, x0) + lor(x, a2, ks, x0+dx)

def triplor2(x, a0, a1, a2, kc, ks, x0, dx1, dx2, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+lor(x, a0, ks, x0-dx1)+lor(x, a1, kc, x0) + lor(x, a2, ks, x0+dx2)

def tripgaus(x, a0, a1, a2, sc, ss, x0, dx, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+gaussianNormal(x, x0-dx, a0, ss, 0)+gaussianNormal(x, x0, a1, sc, 0) + gaussianNormal(x, x0+dx, a2, ss, 0)

def tripgaus_letting_the_peaks_roam_free(x, a0, a1, a2, sc, ss, x0, x1, x2, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+gaussianNormal(x, x0, a0, ss, 0)+gaussianNormal(x, x1, a1, sc, 0) + gaussianNormal(x, x2, a2, ss, 0)

def twogaussian(x, x0, x1, a0, a1, sig0, sig1, y0):
    return a0*np.exp(-((x-x0)**2)/(2*sig0**2)) + a1*np.exp(-((x-x1)**2)/(2*sig1**2)) + y0

def twolor(x, a0, a1, k0, k1, x0, x1, y0):
    return y0 + lor(x, a0, k0, x0) + lor(x, a1, k1, x1)

def twoloronev(x, a0, a1, ac, ampl, lwc, gwc, ks, x0, dx, y0):
    return y0 + lor(x, a0, ks, x0-dx) + lor(x, a1, ks, x0+dx) + ac*Voigt1D(x_0=x0, amplitude_L=ampl, fwhm_L=lwc, fwhm_G=gwc)(x)


def twosinc2(x, x0, x1, a0, a1, y0, k0, k1):
    return a0*np.sinc((x-x0)/k0)**2 + a1*np.sinc((x-x1)/k1)**2 + y0

def waistFit(kvals,dat,lam):
    i=dat.argmin()
    guess=[kvals[i],1,dat[i]]
#     print(guess)
    pred_params, uncert = curve_fit(lambda z, z0, zr, w0: beam_waist(z, z0, zr, w0, lam), kvals, dat, p0=guess)
    perr = np.sqrt(np.diag(uncert))
    zpred = beam_waist(kvals, *pred_params, lam)
#     print('Predicted params (z0, zr, w0):', pred_params)
#     print('Residual, RMS(obs - pred):', np.sqrt(np.mean((dat - zpred)**2)))
    return zpred, pred_params, perr
