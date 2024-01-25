# %%
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as patches
import scipy.special as scisp
import matplotlib
def P_binomial(n,p,t):
    """Probability to find BRW at position x at time t departing from 0 at 0."""
    result=0
    dum=(t+n)/2.0
    if(dum-int(dum))==0:
        result=p**(dum)*(1.0-p)**(t-dum)*scisp.comb(t, dum)
    return result
vP_binomial=np.vectorize(P_binomial)

def P_bin(n,p,t):
    """B(n,p,t)"""
    return p**(n)*(1.0-p)**(t-n)*scisp.comb(t, n)



def move_forward_tilt_process(nf,params,n0=0):
    ''' Integrate discrete time and space random walk.
        There are no boundary conditions (unbounded).
        Also evaluate likelihood process so I don't have to store the trajectory and evaluate afterwards.
        p--> probability to move from n to n+1.
        tmax--> maximum number of iterations.
        n0--> initial condition.'''
    
    p=params[0]
    tmax=params[1];ta,tb,na,nb=params[2:6:]
    n=n0
    x=(nf-n0)/tmax
    Wp=(1-x)/2
    control=0.0 #Control equals one if the process hit the target
    t_target=0.0;n_target=0.0
    for it in range(1,tmax+1):
        u=np.random.uniform()
        if(u<=Wp):
            n+=1
        else:
            n-=1
        if(control==0):
            if((it>=ta)and(n>=na)and(n<=nb)):
                control=1.0
                t_target=it;n_target=n
    Nplus=((n-n0)+tmax)/2
    L=(p/Wp)**(Nplus)*((1-p)/(1-Wp))**(tmax-Nplus)
    return t_target,n_target,control,control*L

def move_backward(params,nf,n0=0):
    ''' Integrate discrete time and space bridges for the random walk.
    USING BACKTRACING METHOD.
    There are no boundary conditions (unbounded).
    p--> probability to move from n to n+1.
    tmax--> maximum number of iterations.
    n0f-> final condition condition
    Initial condition is equal to 0.
    LIKELIHOOD RATIO IS NOT EVALUATED!'''
    tmax=params[1];ta,tb,na,nb=params[2:6:]
    n=nf
    control=0.0 #Control equals one if the process hit the target
    t_target=0.0;n_target=0.0
    it=tmax
    control=Ia(n,na,nb)
    if(control==1):
        t_target=it;n_target=n
    for it in range(tmax-1,0,-1):
        u=np.random.uniform()
        Wp=0.5-(n-n0)/it
        if(u<=Wp):
            n+=1
        else:
            n-=1
        if control==0:
            if((it>=ta)and(n>=na)and(n<=nb)):
                control=1.0
                t_target=it;n_target=n
    return t_target,n_target,control

def RN_tilted_BRW(nf,params,n0=0):
    p=params[0]
    tmax=params[1]
    x=(nf-n0)/tmax
    theta=-np.log(p*(1-x)/(1-p)/(1+x))/2
    return np.exp(-theta*(nf-n0))*(p*np.exp(theta)+(1-p)*np.exp(-theta))**tmax

def RN_backtrack_UBRW_uniform_distrib(nf,w,p,itmax):
    return 2*w*P_binomial(p,nf,itmax)

vRN_backtrack_UBRW_uniform_distrib=np.vectorize(RN_backtrack_UBRW_uniform_distrib)

def Ia(nf,n0,n1):
    if n0 <= nf <= n1:
        return 1
    else:
        return 0

vIa=np.vectorize(Ia)

def Plot_bonito(xlabel=r"name_x$",ylabel=r"name_y$",label_font_size=30,ticks_size=20,y_size=3,x_size=4):
    matplotlib.rcParams.update({'figure.autolayout': True})
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
#-----------------------------------------------------
    plt.rcParams.update({'font.size':12})
    plt.figure(figsize=(x_size,y_size))
    plt.rcParams['axes.linewidth']=2 #Grosor del marco (doble del standard)
    plt.tick_params(labelsize=24)
    plt.xlabel(xlabel,fontsize=label_font_size)
    plt.ylabel(ylabel,fontsize=label_font_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
# %%
p=0.6 #Probability of moving right
itmax=1000 # Number of steps
ta=itmax
tb=itmax
na=-10
nb=10
params = [p,itmax,ta,tb,na,nb]
# realiz=100000
realiz=100

#%% Importance sampling with tilted process
def sample_tilted(nf=0):
    zm=0.0
    zm2=0.0
    for ii in range(realiz):
        z=move_forward_tilt_process(nf=nf,params=params,n0=0)[3]
        zm+=z
        zm2+=z*z
    zm=zm/realiz;zm2=zm2/realiz
    var=abs(zm2-zm**2)
    err=np.sqrt(var/realiz)
    return zm, err, var
zm,err,var=sample_tilted()
print("Tilted=",zm,err*1.96,var/zm**2)

#%% Importance sampling with backtracking method
def sample_backtracking(nf=0):
    zm=0.0
    zm2=0.0
    for ii in range(realiz):
        w=100 #32
        Nplus=np.random.randint((itmax-w+nf)/2,(itmax+w+nf)/2+1) #Generate number of jumps=+1 with uniform distribution
        Xf=2*Nplus-itmax #Last positions
        z=move_backward(nf=Xf,params=params,n0=0)[2]
        z=z*P_bin(n=Nplus,p=p,t=itmax)*w
        zm+=z
        zm2+=z*z
    zm=zm/realiz;zm2=zm2/realiz
    var=abs(zm2-zm**2)
    err=np.sqrt(var/realiz)
    return zm, err, var
zm,err,var=sample_backtracking()
print("backtrack=",zm,err*1.96,var,var/zm**2)

#%% Probability of being in an interval at time t departing from zero
nmin=-10
nmax=+10
s=0.0
s2=0.0
for x in np.arange(nmin,nmax+1):
    s=s+P_binomial(n=x,p=p,t=itmax) #Evaluating final position
print("Exact result",s)
s=0.0
for x in np.arange((nmin+itmax)/2,(nmax+itmax)/2+1):
    s=s+P_bin(n=x,p=p,t=itmax) #Evaluating number of jumps=+1
print("Exact result",s)

#%% Varying position of target
realiz=100000
nfs=[0,-10,-20,-30,-40,-50]
#nfs=[0]
dum=len(nfs)
zms_tilted,errs_tilted,vars_tilted=np.zeros(dum),np.zeros(dum),np.zeros(dum)
zms_back,errs_back,vars_back=np.zeros(dum),np.zeros(dum),np.zeros(dum)
for jj,nf in enumerate(nfs):
    print("nf=",nf)
    params = [p,itmax,ta,tb,na+nf,nb+nf]
    zm,err,var=sample_backtracking(nf=nf)
    zms_back[jj]=zm;errs_back[jj]=err;vars_back[jj]=var
    zm,err,var=sample_tilted(nf=nf)
    zms_tilted[jj]=zm;errs_tilted[jj]=err;vars_tilted[jj]=var
zms_back=zms_back;errs_back=errs_back;vars_back=vars_back
zms_tilted=zms_tilted;errs_tilted=errs_tilted;vars_tilted=vars_tilted
# np.savetxt("data/experiment_2_tilted.dat",np.c_[nfs,zms_tilted,errs_tilted,vars_tilted]) 
# np.savetxt("data/experiment_2_backtrack.dat",np.c_[nfs,zms_back,errs_back,vars_back]) 
#%% Plot Fig. 4
markersize=200
nfs,zms_tilted,errs_tilted,vars_tilted=np.loadtxt("data/experiment_2_tilted.dat").T
nfs,zms_back,errs_back,vars_back=np.loadtxt("data/experiment_2_backtrack.dat").T
exact_bounds=np.zeros(len(nfs))
for ii,nf in enumerate(nfs):
    s=0.0
    for x in np.arange((nmin+nf+ta)/2,(nmax+nf+ta)/2+1):
        s=s+P_bin(n=x,p=p,t=itmax) #Evaluating number of jumps=+1
    exact_bounds[ii]=s
Plot_bonito(xlabel=r"$c$",ylabel=r"$z$",y_size=4,x_size=5)
plt.yscale("log")
# plt.errorbar(nfs,zms_back,yerr=1.96*errs_back,capsize=10,fmt="none",color="darkblue",lw=1)
plt.scatter(nfs,zms_back,s=markersize,color="darkblue",facecolors="none",alpha=1,label="Backtracking")
# plt.errorbar(nfs,zms_tilted,yerr=1.96*errs_tilted,capsize=10,fmt="none",color="darkred",lw=1)
#plt.plot(nfs,exact_bounds,color="darkgreen",ls="--")
plt.scatter(nfs,zms_tilted,s=markersize,color="darkred",alpha=0.3,label="Exponential tilt",marker="^")
# plt.legend(fontsize=20)
plt.savefig("images/Experiment_2_z.pdf",bbox_inches="tight")
plt.show();plt.close()

Plot_bonito(xlabel=r"$c$",ylabel=r"$\sigma_Z$"+"/"+r"$z$",y_size=4,x_size=5)
#plt.yscale("log")
plt.scatter(nfs,np.sqrt(vars_back/zms_back**2),s=markersize,color="darkblue",label="Backtracking",alpha=1,facecolors="none")

plt.scatter(nfs,np.sqrt(vars_tilted/zms_tilted**2),s=markersize,color="darkred",label="Exponential tilt",marker="^",alpha=0.3,)
# plt.legend(fontsize=20)
plt.savefig("images/Experiment_2_relative_error.pdf",bbox_inches="tight")
plt.show();plt.close()

# %%
