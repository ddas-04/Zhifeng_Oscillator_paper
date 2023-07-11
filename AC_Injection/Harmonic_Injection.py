import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import os
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


plt.rcParams.update({'font.size':24})
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams.update({'font.weight':'bold'})
plt.rcParams["font.family"] = "Times New Roman"
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


# #### Function for LLGS Equation

def LLGS(m,Heff,beta_STT,beta_SOT,alpha,H_STT,H_SOT):
	global mu0, gamma

	precission=-gamma*np.cross(m,Heff)
	damping=-alpha*gamma*np.cross(m,np.cross(m,Heff))
	Field_like_STT=gamma*(alpha-beta_STT)*(np.cross(m,H_STT))
	Damp_like_STT=-gamma*((alpha*beta_STT)+1)*np.cross(m,np.cross(m,H_STT))
	Field_like_SOT=gamma*(alpha-beta_SOT)*np.cross(m,H_SOT)
	Damp_like_SOT=-gamma*((alpha*beta_SOT)+1)*np.cross(m,np.cross(m,H_SOT))

	total_torque=precission+damping+Field_like_STT+Damp_like_STT+Field_like_SOT+Damp_like_SOT

	dmdt=(mu0/(1+alpha**2))*total_torque

	return dmdt


# #### Function for Effective field calculation

def H_eff_calc(m,Han,Nx,Ny,Nz,Hext):
	global uniax_dir
	H_uni=Han*np.dot(m,uniax_dir)*uniax_dir                        # in A/m
	H_demag=-(Ms/mu0)*np.array([Nx*m[0], Ny*m[1], Nz*m[2]])    # in A/m

	H_eff=H_uni+H_demag+H_ext

	return H_eff

def vec_mag(A):
    return np.sqrt((A*A).sum())


# #### Function for calculating the cos$\theta$, where $\theta$ is the angle between PL and FL

def cos_theta(m):
    global PL_vec
    return np.dot(m,PL_vec)/(vec_mag(m)*vec_mag(PL_vec))


# #### Function to calculate $\eta$

def eta(m):
    global P    
    return P/(1+(P**2)*cos_theta(m))


# #### Function to calculate resistance of the MTJ

def R_MTJ(m,V_MTJ):
    global Rp, Rap, Vh    
    R=Rp+((Rap-Rp)*(1-cos_theta(m))/(2*(1+V_MTJ/Vh)**2))
    return R

def V_MTJ(ti):
	global VDD_STT, inj_freq,v_ac
	return VDD_STT+(v_ac*np.sin(2*np.pi*inj_freq*ti))
	
	
##### Function to calculate |$\overrightarrow{H}_{STT}$|

def H_STT_mag(m,ti):
	global hbar, t_FL, Ms
	I_STT=(V_MTJ(ti))/R_MTJ(m,V_MTJ(ti))
	J_STT=I_STT/A_MTJ
	mag=(eta(m)*hbar*J_STT)/(2*q*t_FL*Ms)

	return mag


# #### Function to calculate |$\overrightarrow{H}_{SOT}$|

def H_SOT_mag(V_HM):
	global hbar, t_FL, Ms, R_HM, A_HM, t_HM, lambda_sf, theta_SH
	I_c_HM=V_HM/R_HM
	J_c_HM=I_c_HM/A_HM
	J_SOT_mag=theta_SH*J_c_HM*(1-(1/np.cosh(t_HM/lambda_sf)))
	Hsotmag=(hbar*J_SOT_mag)/(2*q*t_FL*Ms)

	return Hsotmag    


# #### Universal Constants

gamma=1.76e11;           # in [(rad)/(s.T)]
mu0=4*np.pi*1e-7 ;       # in T.m/A
q=1.6e-19;               # in Coulomb
hbar=1.054e-34;          # in (J-s)
K_B=1.38064852e-23       #in J/K


# #### Parameters

A_MTJ=50*90*1e-18      # in m^2
t_FL=3e-9              # in nm
RA=1.5*(1e-6)**2       # in Ohm-m^2
Ku_Bulk=2.245e5        # in J/m^3
Ki=1.286e-3            # in J/m^3
Ms=1.58                # in T
L_HM=100e-9            # in m
W_HM=100e-9            # in m
t_HM=5e-9              # in m
rho_HM=200e-8          # in Ohm-m
lambda_sf=5e-9         # in m
theta_SH=-0.1           # dimensionless
P=0.4                  # dimensionless
TMR=150                # in %
alpha=0.025            # dimensionless
Vh=0.5
PL_vec=np.array([0,-1,0])
Ny=0.03984407224293963 # dimensionless
Nx=0.07340636630038447 # dimensionless
Nz=0.8867495614566816  # dimensionless
uniax_dir=np.array([0,0,1])

# #### Calculated parameters

Rp=RA/A_MTJ                             # in Ohm
Rap=((TMR/100.0)+1)*Rp                  # in Ohm

Han=(2*(Ku_Bulk+(Ki/t_FL)))/(Ms)        # in A/m

A_HM=W_HM*t_HM
R_HM=rho_HM*L_HM/A_HM

# #### Assumed Parameters

VDD_SOT=4.5*0.27                             # in V
I_C_SOT=VDD_SOT/R_HM


VDD_STT=4.5*0.27                              # in V

inj_freq=2*3.75187594*1e9
v_ac=0.1

# #### External Magnetic field

H_ext_mag=1.0*529*1e3/(4*np.pi)                                 # in A/m
H_ext=H_ext_mag*np.array([0,1,0])


beta_STT=0.29
beta_SOT=2.9

#### Time parameters
t_start=0
t_step=1e-12
t_end=50e-9
N=int(t_end/t_step)+1
t_save=np.zeros(N)
t_save[0]=t_start


#### Initial Magnetization
mz=1.0
m_init=np.array([0, np.sqrt(1-mz**2), mz])

m_save=np.zeros((N,3))
m_save[0,:]=m_init


V_inj=np.zeros(N)
V_inj[0]=V_MTJ(t_save[0])

h=t_step
t=t_start
i=0
m=m_init
while t<t_end:
	t=t+t_step
	print('-------------------------------------')
	print('Time = ' + str(t*1e9)+' ns')
	t_save[i+1]=t
	V_inj[i+1]=V_MTJ(t)
	
	Heff=H_eff_calc(m,Han,Nx,Ny,Nz,H_ext)
	H_STT=H_STT_mag(m,t)*PL_vec
	H_SOT=H_SOT_mag(VDD_SOT)*np.array([0,-1,0])

	k1=LLGS(m,Heff,beta_STT,beta_SOT,alpha,H_STT,H_SOT)
	mk2=m+h*k1/2.0
	Heff=H_eff_calc(mk2,Han,Nx,Ny,Nz,H_ext)
	H_STT=H_STT_mag(mk2,t)*PL_vec
	H_SOT=H_SOT_mag(VDD_SOT)*np.array([0,-1,0])

	k2=LLGS(mk2,Heff,beta_STT,beta_SOT,alpha,H_STT,H_SOT)
	mk3=m+h*k2/2.0
	Heff=H_eff_calc(mk3,Han,Nx,Ny,Nz,H_ext)
	H_STT=H_STT_mag(mk3,t)*PL_vec
	H_SOT=H_SOT_mag(VDD_SOT)*np.array([0,-1,0])

	k3=LLGS(mk3,Heff,beta_STT,beta_SOT,alpha,H_STT,H_SOT)
	mk4=m+h*k3
	Heff=H_eff_calc(mk4,Han,Nx,Ny,Nz,H_ext)
	H_STT=H_STT_mag(mk4,t)*PL_vec
	H_SOT=H_SOT_mag(VDD_SOT)*np.array([0,-1,0])

	k4=LLGS(mk4,Heff,beta_STT,beta_SOT,alpha,H_STT,H_SOT)

	m_new=m + ((h/6.0)*(k1+2*k2+2*k3+k4))
	m=m_new
	'''
	H_demag=-(Ms/mu0)*np.array([Nx*m[0], Ny*m[1], Nz*m[2]]) 
	H_STT=H_STT_mag(m,VDD_STT)*PL_vec
	H_SOT=H_SOT_mag(VDD_SOT)*np.array([0,-1,0])
	H_uni=Han*m[2]*np.array([0, 0, 1])
	print('H_uni = ' + str(H_uni))
	print('H_Demag = ' + str(H_demag))
	print('H_ext = ' + str(H_ext))
	print('H_SOT = ' + str(H_SOT))
	print('H_STT = ' + str(H_STT))


	'''
	os.system('clear')
	i=i+1


	m_save[i,:]=m_new

os.system('clear')
t_save=t_save*1e9


plt.figure(figsize=(12,6))
plt.plot(t_save,m_save[:,0], linewidth=2.5, label='mx')
plt.plot(t_save,m_save[:,1], linewidth=2.5, label='my')
plt.plot(t_save,m_save[:,2], linewidth=2.5, label='mz')
plt.legend()
plt.xlabel('Time(ns)')
plt.ylabel('m')
#plt.savefig('Complete_m_profile_simulation_result.png', bbox_inches='tight', pad_inches=0.0)
plt.grid()

Sample_len=4000

plt.figure(figsize=(12,6))
plt.plot(t_save[N-Sample_len:N-1],m_save[N-Sample_len:N-1,0], linewidth=2.5)
plt.xlabel('Time(ns)')
plt.ylabel(r"$m_x$")
#plt.savefig('mx_oscillation_result.png', bbox_inches='tight', pad_inches=0.0)
plt.grid()


plt.figure(figsize=(12,6))
plt.plot(t_save[N-Sample_len:N-1],m_save[N-Sample_len:N-1,1], linewidth=2.5)
plt.xlabel('Time(ns)')
plt.ylabel(r"$m_y$")
#plt.savefig('my_oscillation_result.png', bbox_inches='tight', pad_inches=0.0)
plt.grid()

r = 1
phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]

x = r*np.sin(phi)*np.cos(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*np.cos(phi)




fig=plt.figure(figsize=(8,8))
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=2.0)
ax.plot3D(m_save[N-Sample_len:N-1,0], m_save[N-Sample_len:N-1,1], m_save[N-Sample_len:N-1,2], 'red')
ax.arrow3D(0,0,0,
           0,0,1.2*r,
           mutation_scale=10,
           ec ='black',
           fc='black')
ax.text(0,0,1.3*r, r"$z$", size=26, zorder=1,  color='k')
ax.arrow3D(0,0,0,
           0,1.2*r,0,
           mutation_scale=10,
           ec ='black',
           fc='black')
ax.text(0,1.3*r,0, r"$y$", size=26, zorder=1,  color='k')
#ax.text(0,-2.2*r,0, r"$m_\mathrm{x}$", size=36, zorder=1,  color='k')
ax.arrow3D(0,0,0,
           1.2*r,0,0,
           mutation_scale=10,
           ec ='black',
           fc='black')
ax.text(1.3*r,0,0, r"$x$", size=26, zorder=1,  color='k')
#plt.show()

signal_time_frame=(t_save[N-Sample_len:N-1])*1e-9   # converted from ns to s
N_signal=len(signal_time_frame)

duration=signal_time_frame[-1]-signal_time_frame[0]
sample_rate=N_signal/duration

signal=m_save[N-Sample_len:N-1,1]

yf=rfft(signal)
xf=rfftfreq(N_signal,1/sample_rate)

peaks, _ = find_peaks(np.abs(yf), height=100)
print('Peak Value = ' + str(np.abs(yf[peaks])))
print('Frequencies = ' +str(xf[peaks]/1e9)+ ' GHz')

plt.figure(figsize=(8,6))
plt.plot(xf/1e9, np.abs(yf),'*-',markersize=10)
plt.xlim([0.0,15])
plt.xlabel('Frequency(GHz)')
#plt.savefig('Stable_Oscillation_FFT.png', bbox_inches='tight', pad_inches=0.0)

'''
fig, ax = plt.subplots(2, 1,figsize=(16,9), gridspec_kw={'height_ratios': [1, 3]})
ax[0].plot(t_save[N-Sample_len:N-1],V_inj[N-Sample_len:N-1]-VDD_STT, linewidth=2.5)

#ax[0].set_xlabel('Time(ns)')
ax[0].set_ylabel('v_ac')
#ax[0].set_xlim([0, Total_time])
#ax[0].set_ylim([-1.2, 1.2])
#plt.grid()
#plt.xlim([0, 5])

ax[1].plot(t_save[N-Sample_len:N-1],m_save[N-Sample_len:N-1,1], linewidth=2.5)
ax[1].set_xlabel('Time(ns)')
ax[1].set_ylabel(r"$m_y$")
'''

plt.figure(figsize=(16,8))
plt.plot(t_save[N-Sample_len:N-1],V_inj[N-Sample_len:N-1]-VDD_STT, linewidth=2.5, label='signal')
plt.plot(t_save[N-Sample_len:N-1],m_save[N-Sample_len:N-1,1], linewidth=2.5, label='m_y')
plt.legend()
plt.grid()
plt.xlabel('Time(ns)')
plt.show()

plt.show()

