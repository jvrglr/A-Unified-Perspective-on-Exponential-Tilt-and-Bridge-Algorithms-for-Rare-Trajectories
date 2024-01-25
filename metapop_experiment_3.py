# %%

from scipy.optimize import curve_fit
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as patches
import functions as f
from numpy import exp as exp
import matplotlib


def r(params,n):
    '''Probability to move to the right.
    If theta=0 then this is the original process.
    If theta<0 then this is the tilted process favouring the -L->L transition'''
    L,v,theta=params[0],params[1],params[3]
    return 1.0/(1+exp(v*n*(n-L)*(L+n)-2*theta))

def g(params,n):
    L,v,theta=params[0],params[1],params[3]
    f=v*n*(n-L)*(L+n)
    return (exp(f-theta)+exp(theta))/(1.0+exp(f))

def prod(params,t1,t2,ns):
    p=1.0
    for t in range(t1+1,t2+1):
        p=p*g(params,ns[t])
    return p

def move_forward_original_or_tilted_process(params,n0=0):
    ''' Integrate discrete time and space random walk.
    Probability transition depend on state.
    If theta=0 then original process, otherwise tilted process.
    There are no boundary conditions (unbounded).
    Also evaluate likelihood process.
    n0--> initial condition.'''
    tmax,theta=params[2],params[3]
    t=0
    n=n0
    ts=np.arange(tmax+1)
    ns=np.zeros(tmax+1);ns[0]=n
    Lt=1.0
    for t in ts[1::]:
        u=np.random.uniform()
        if u<r(params,n):
            n+=1
        else:
            n-=1
        ns[t]=n
        Lt=Lt*g(params,n)
    Lt=Lt*exp(-theta*(n-n0))#np.exp(-theta*(n-n0))
    return ns,Lt

def move_backwards_backtrack_process(params,nf,P,N):
    ''' Integrate discrete time and space backtrack process.
    Probability transition depend on state.
    There are boundary conditions (reflective)-->this is an approximation of the unbounded process.
    Also evaluate likelihood process.
    nf--> final condition.
    P--> solution of master eq.
    N--> artificial boundary introduced to solve Master Eq.'''
    tmax=params[2]
    t=tmax
    n=nf
    ts=np.arange(tmax+1)
    ns=np.zeros(tmax+1);ns[-1]=n
    Lt=1
    for t in ts[-2::-1]:      
        u=np.random.uniform()
        index=n+N
        # print(index,n)
        ratioP=P[index+1][t]/P[index][t+1]
        p_plus=ratioP*(1-r(params,n+1))
        if u<p_plus:
            n=n+1
        else:
            n-=1
        ns[t]=n

    Lt=P[nf+N][itmax]
    return ns,Lt


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
L=15 #Position of meta-stable state
v=0.001 #constant in exponent
itmax=400 # Number of steps
theta=0 #no bias
params = [L,v,itmax,theta]



# %%
#Trajectories for the original process Fig 5(a)
print("Original process with  X_0=0")

realiz=100
theta=0 #< tilt parameter
params = [L,v,itmax,theta]
Plot_bonito(xlabel=r"$t$",ylabel=r"$n$")
ts=np.arange(itmax+1)
for ii in range(realiz):
    ns,dum=move_forward_original_or_tilted_process(params,n0=0)
    plt.plot(ts,ns,alpha=0.1)
plt.yticks([-15,0,15],[r"$-\ell$",0,r"$\ell$"])
plt.plot([0,itmax],[L,L],lw=3,ls="--",color="darkred")
plt.plot([0,itmax],[-L,-L],lw=3,ls="--",color="darkred")
# plt.savefig("images/trajectories_exp3.pdf",bbox_inches="tight")
plt.show();plt.close()

#%% Trajectories with tilted process Fig 5(b)
realiz=100
theta=1 #< tilt parameter
params = [L,v,itmax,theta]
print("Exponential tilted process with  X_0=-L")
Plot_bonito(xlabel=r"$t$",ylabel=r"$n$")
ts=np.arange(itmax+1)
for ii in range(realiz):
    ns,Lt=move_forward_original_or_tilted_process(params,n0=-L)
    plt.plot(ts,ns,alpha=0.1)

plt.yticks([-15,0,15],[r"$-\ell$",0,r"$\ell$"])
plt.plot([0,itmax],[L,L],lw=3,ls="--",color="darkred")
plt.plot([0,itmax],[-L,-L],lw=3,ls="--",color="darkred")
# plt.savefig("images/trajectories_exp3_exponential_tilt.pdf",bbox_inches="tight")
plt.show();plt.close()


# %%
#Solve master equation
itmax=400
theta=0 #< tilt parameter
params = [L,v,itmax,theta]
N=40 #Artificial barrier for numerical approach
P=np.zeros((2*N+1,itmax+1))
x0=-L
n0=x0+N
P[n0][0]=1
for t in range(1,itmax+1):
    for n in range(1,2*N):
        x=n-N
        P[n][t]=P[n-1][t-1]*r(params,x-1)+(1.0-r(params,x+1))*P[n+1][t-1]
    x,n=-N,0
    #P[n][t]=P[n][t-1]*r(params,x)+(1.0-r(params,x+1))*P[n+1][t-1]
    P[n][t]=P[n+1][t-1]
    x,n=N,2*N
    P[n][t]=P[n-1][t-1]
np.save("data/Prob/P.npy", P, allow_pickle=True, fix_imports=True)
#%% Trajectories with Backtracking process Fig 5(c)
print("backtracking process with  X_f=L")
realiz=100
P=np.load("data/Prob/P.npy")
N=40
ts=np.arange(itmax+1)
Plot_bonito(xlabel=r"$t$",ylabel=r"$n$")
for ii in range(realiz):
    ns,Lt=move_backwards_backtrack_process(params,nf=L,P=P,N=N)    
    plt.plot(ts,ns,alpha=0.1)

plt.yticks([-15,0,15],[r"$-\ell$",0,r"$\ell$"])
plt.plot([0,itmax],[L,L],lw=3,ls="--",color="darkred")
plt.plot([0,itmax],[-L,-L],lw=3,ls="--",color="darkred")
# plt.savefig("images/trajectories_exp3_backtrack.pdf",bbox_inches="tight")
plt.show();plt.close()
# %%
#Measure transition time with tilted process
realiz=100
theta=1
Etau=0.0
for ii in range(realiz):
    try:
        ns,_=move_forward_original_or_tilted_process(params,n0=-L)
        t1=np.where(ns==-L)[0][-1]
        t2=np.where(ns==L)[0][0]
        tau=t2-t1
        Lt=prod(params,t1,t2,ns)*exp(-2*theta*L)
    except:
        tau=0
        Lt=1
Etau+=tau*Lt
Etau=Etau/realiz
print(Lt,Etau,tau)

# %% Figs 6(b) and 7
#Compute probability that transition is before t=100 with exponential tilted

#numerical
realiz=10000
itmax=100 # Number of steps
params = [L,v,itmax,theta]
thetas=np.linspace(0,1,20)
Zs,errs,counts=np.zeros(len(thetas)),np.zeros(len(thetas)),np.zeros(len(thetas))
for jj,theta in enumerate(thetas):
    print("---------------------")
    print("Theta=",theta)
    params = [L,v,itmax,theta]
    Z,Z2=0.0,0.0
    count=0
    for ii in range(realiz):
        ns,Lt=move_forward_original_or_tilted_process(params,n0=-L)
        if (ns[-1]>=(L-2)):
            IA=1
            count+=1
        else:
            IA=0
        dum=IA*Lt
        Z2+=dum*dum
        Z+=dum
    Z,Z2=Z/realiz,Z2/realiz
    err=np.sqrt(np.abs(Z*Z-Z2)/realiz)
    Zs[jj],errs[jj],counts[jj]=Z,err,count
    print("# of samples:",count)
    print("prob.jump was before t=100, comparing exact and exponential tilt")
    print("Estimation=",Z,"Exact=",s)
    print("Difference=",np.abs(Z-s),"Abs. Error=",err)
    print("real relative=",np.abs(s/Z),"rel. Error=",err/Z)
Zs=np.array(Zs);errs=np.array(errs);counts=np.array(counts)
# np.savetxt("data/experiment_3_tilted_r10000.dat",np.c_[thetas,Zs,errs]) 
#%%
#Exact
N=40
P=np.load("data/Prob/P.npy")
s=0.0
for x in range(L-2,N):
    n=x+N
    s+=P[n][100]
#np.savetxt("data/experiment_3_tilted_v1.dat",np.c_[thetas,Zs,errs]) 
#thetas,Zs,errs=np.loadtxt("data/experiment_3_tilted_r1000000.dat").T
thetas,Zs,errs=np.loadtxt("data/experiment_3_tilted_r1000000.dat").T
Plot_bonito(xlabel=r"$\theta$",ylabel=r"$z$",y_size=4,x_size=5)
plt.yticks([0,0.00001,0.00002],[ 0,r"$1\cdot10^{-5}$",r"$2\cdot10^{-5}$"])
#plt.yscale("log")
#plt.ylim([0.000000000000001,0.00005])
plt.errorbar(thetas,Zs,yerr=1.96*errs,capsize=10,fmt="none",color="darkblue",lw=1)
plt.scatter(thetas,Zs,s=100,color="darkblue",alpha=0.5,label="Backtracking")
plt.plot(thetas,thetas*0+s,color="darkgreen",ls="--",lw=3)

# plt.savefig("images/Experiment_3_z_v1_exponential_tilted_r10E5.pdf",bbox_inches="tight")
plt.savefig("images/Experiment_3_z_v1_exponential_tilted_r1000000.pdf",bbox_inches="tight")

plt.show();plt.close()

# Plot_bonito(xlabel=r"$\theta$",ylabel="Counts",y_size=6,x_size=8)

# #plt.yscale("log")
# plt.scatter(thetas,counts/realiz,s=100,color="darkblue",alpha=0.5,label="Backtracking")

#plt.savefig("images/Experiment_3_counts_v2.pdf",bbox_inches="tight")
#plt.show();plt.close()



# %% Fig 6(a)
#Compute probability that transition is before t=100 with Backtracking
#Exact
N=40
P=np.load("data/Prob/P.npy")
s=0.0
for x in range(L-2,N):
    n=x+N
    s+=P[n][100]
print(s)

#numerical
realiz=10000
theta=0
itmaxs=np.linspace(100,200,11,dtype=int)
D=3
Zs,errs,counts=np.zeros(len(itmaxs)),np.zeros(len(itmaxs)),np.zeros(len(itmaxs))
for jj,itmax in enumerate(itmaxs):
    print("---------------------")
    print("tmax=",itmax)
    params = [L,v,itmax,theta]
    Z,Z2=0.0,0.0
    count=0
    for ii in range(realiz):
        u=np.random.randint(-D,D+1)
        u=u*2
        ns,Lt=move_backwards_backtrack_process(params,nf=L+u,P=P,N=N)    
        if (ns[100]>=(L-2)):
            IA=1
            count+=1
        else:
            IA=0
        dum=IA*Lt*(2*D+1)
        Z2+=dum*dum
        Z+=dum
    Z,Z2=Z/realiz,Z2/realiz
    err=np.sqrt(np.abs(Z*Z-Z2)/realiz)
    Zs[jj],errs[jj],counts[jj]=Z,err,count
    print("# of samples:",count)
    print("prob.jump was before t=100, comparing exact and exponential tilt")
    print("Estimation=",Z,"Exact=",s)
    print("Difference=",np.abs(Z-s),"Abs. Error=",err)
    print("real relative=",np.abs(s/Z),"rel. Error=",err/Z)
Zs=np.array(Zs);errs=np.array(errs);counts=np.array(counts)
#np.savetxt("data/experiment_3_backtracking.dat",np.c_[itmaxs,Zs,errs]) 
#%%
itmaxs,Zs,errs=np.loadtxt("data/experiment_3_backtracking.dat").T
Plot_bonito(xlabel=r"$t^{\dag}$",ylabel=r"$z$",y_size=4,x_size=5)
#plt.yscale("log")
plt.errorbar(itmaxs,Zs,yerr=1.96*errs,capsize=10,fmt="none",color="darkblue",lw=1)
plt.scatter(itmaxs,Zs,s=100,color="darkblue",alpha=0.5,label="Backtracking")
plt.plot(itmaxs,itmaxs*0+s,color="darkgreen",ls="--",lw=3)
plt.yticks([0.0000045,0.0000040,0.0000035,0.0000030],[r"$4.5\cdot10^{-6}$",r"$4\cdot10^{-6}$",r"$3.5\cdot10^{-6}$",r"$3\cdot10^{-6}$"])
plt.savefig("images/Experiment_3_z_backtracking.pdf",bbox_inches="tight")
plt.show();plt.close()

