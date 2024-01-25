# %%

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scisp

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as patches
import matplotlib

def move_forward(params,n0=0):
    ''' Integrate discrete time and space random walk.
        There are no boundary conditions (unbounded).
        p--> probability to move from n to n+1.
        tmax--> maximum number of iterations.
        n0--> initial condition'''
    p=params[0];tmax=params[1]
    ns=[n0]
    n=n0

    for it in range(1,tmax+1):
        u=np.random.uniform()
        if(u<=p):
            n+=1
        else:
            n-=1
        ns.append(n)
    ts=list(range(tmax+1))
    ts=np.array(ts);ns=np.array(ns)
    return ts,ns

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
        p--> probability to move from n to n+1.
        tmax--> maximum number of iterations.
        n0--> initial condition'''
    tmax=params[1]
    n=n0
    x=(nf-n0)/tmax
    Wp=(1-x)/2
    ns=np.zeros(tmax+1)
    ns[0]=n
    for it in range(1,tmax+1):
        u=np.random.uniform()
        if(u<=Wp):
            n+=1
        else:
            n-=1
        ns[it]=n
    return np.arange(tmax+1),ns

def move_backward(params,nf=0,n0=0):
    ''' Integrate discrete time and space bridges for the random walk.
    USING BACKTRACING METHOD.
    There are no boundary conditions (unbounded).
    p--> probability to move from n to n+1.
    tmax--> maximum number of iterations.
    n0f-> final condition condition
    Initial condition is equal to 0.'''
    p=params[0];tmax=params[1]
    n=nf
    ns=np.arange(tmax+1)
    ns[tmax]=n
    ns[0]=n0
    for it in range(tmax-1,0,-1):
        u=np.random.uniform()
        Wp=0.5-(n-n0)/it
        if(u<=Wp):
            n+=1
        else:
            n-=1
        ns[it]=n
    return np.arange(tmax+1),ns

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
params = [p,itmax]
# realiz=100000

#%%
#Forward paths Fig. 1(a)
realiz=100
Plot_bonito(xlabel=r"$t$",ylabel=r"$n$",y_size=4,x_size=5)
for i in range(100):
    ts,ns = move_forward(params)
    plt.plot(ts,ns,alpha=0.2,label="UBRW")
    plt.plot(ts, ts*(2*p-1),color="darkred",ls="--",lw=5)
    mn=ts*(2*p-1)
    sig=np.sqrt(4*p*(1-p)*itmax)
    plt.plot(ts, mn+np.sqrt(4*p*(1-p)*ts),color="darkblue",ls="--",lw=3)
    plt.plot(ts, mn-np.sqrt(4*p*(1-p)*ts),color="darkblue",ls="--",lw=3)
    #ts,ns = f.move_backward(params)
    #plt.plot(ts,ns,label="Backtracing")
#plt.legend()
plt.xlim([-10,1065])
plt.plot([1000,1000],[-10,10],lw=5,color="red")
# plt.text(1015,-12,r"$A$",fontsize=30,color="red")
# plt.savefig("images/UBRW.png",bbox_inches="tight",dpi=300)
plt.show()

#%% Trajectories tilted process Fig. 1(b)
realiz=100
Plot_bonito(xlabel=r"$t$",ylabel=r"$n$",y_size=4,x_size=5)
for ii in range(realiz):
    ts,ns = move_forward_tilt_process(nf=0,params=params)
    #xfs.append(f.move_forward_cloning(params)[1][-1])
    plt.plot(ts,ns,alpha=0.1)
q=1/2
ts=np.arange(itmax+1)
mn=ts*(2*q-1)
sig=np.sqrt(4*q*(1-q)*itmax)
plt.xlim([-10,1065])
plt.plot(ts, mn,color="darkred",ls="--",lw=5)
plt.plot(ts, mn+np.sqrt(4*q*(1-q)*ts),color="darkblue",ls="--",lw=3)
plt.plot(ts, mn-np.sqrt(4*q*(1-q)*ts),color="darkblue",ls="--",lw=3)
plt.plot([1000,1000],[-10,10],lw=5,color="red")
# plt.text(1015,-12,r"$A$",fontsize=30,color="red")
# plt.savefig("images/UBRW_tilted_process.png",bbox_inches="tight",dpi=300)
plt.show()

#%% Trajectories backward process Fig. 1(c)
Plot_bonito(xlabel=r"$t$",ylabel=r"$n$",y_size=4,x_size=5)
realiz=100
def f1(N,T,t):
    return N*t/T
def f2(N,T,t):
    return N*t*(N*(t-1)-t+T)/(T*(T-1))

for ii in range(realiz):
    ts,ns = move_backward(nf=0,params=params)
    #xfs.append(f.move_forward_cloning(params)[1][-1])
    plt.plot(ts,ns,alpha=0.1,zorder=0)
plt.xlim([-10,1065])
ts=np.linspace(0,1000,100)
Nplus=500
mn=f1(Nplus,1000,ts)
xm=mn*2-ts
plt.plot(ts, xm,color="darkred",ls="--",lw=5,zorder=1)
sig=np.sqrt(4*f2(Nplus,1000,ts)-2*ts*mn)
plt.plot(ts, xm+sig,color="darkblue",ls="--",lw=3)
plt.plot(ts, xm-sig,color="darkblue",ls="--",lw=3)

plt.plot([1000,1000],[-10,10],lw=5,color="red")
# plt.text(1015,-12,r"$A$",fontsize=30,color="red")
# plt.savefig("images/UBRW_backtracked_process.png",bbox_inches="tight",dpi=300)
plt.show()

#%% Probability of being in an interval at time t departing from zero
nmin=-10
nmax=+10
s=0.0
s2=0.0
for x in np.arange(nmin,nmax+1):
    s=s+P_binomial(n=x,p=p,t=itmax) #Evaluating final position
    s2=s2+P_binomial(n=x,p=p,t=itmax)**2 #Evaluating final position
print("Exact result",s)
s=0.0
for x in np.arange((nmin+itmax)/2,(nmax+itmax)/2+1):
    s=s+P_bin(n=x,p=p,t=itmax) #Evaluating number of jumps=+1
print("Exact result",s)

#%% Importance sampling on last point tilted process
q=0.5 #New transition probability for jumps=+1
Nplus=np.random.binomial(itmax, q,size=realiz) #Generate number of jumps=+1 with tilted distribution
Xfs=2*Nplus-itmax
M=np.mean(vIa(Xfs,nmin,nmax)*(p/q)**Nplus*((1-p)/(1-q))**(itmax-Nplus))
print("Result exponential Tilted=",M)

#%% Importance sampling on last point tilted process Fig.2
n0=0
Ps=[];errs=[];Vars=[];P2s=[]
ws=np.array([-20,-15,-10,-5,0,5,10,15,20])
for nf in ws:
    x=(nf-n0)/itmax
    q=(1-x)/2
    Nplus=np.random.binomial(itmax, q,size=realiz) #Generate number of jumps=+1 with tilted distribution
    Xfs=2*Nplus-itmax #Last positions
    dum=vIa(Xfs,nmin,nmax)*(p/q)**Nplus*((1-p)/(1-q))**(itmax-Nplus)
    P=np.mean(dum)
    P2=np.mean(dum*dum)
    Var=abs(P2-P**2)
    err=np.sqrt(Var/realiz)
    Ps.append(P);errs.append(err);Vars.append(Var);P2s.append(P2)
Ps=np.array(Ps);errs=np.array(errs);Vars=np.array(Vars);P2s=np.array(P2s)
#np.savetxt("data/prob_hit_A_UBRW_backtrack_variying_uniform_distrib.dat",np.c_[wfs,Ps,errs,Vars]) 
print("Result exponential Tilted=",P)

Plot_bonito(xlabel=r"$n_{k^\dag}$",ylabel=r"$\sigma_Z$"+"/"+r"$z$",y_size=4,x_size=5)

dums=np.arange(-20,20+1)
xs=(dums-n0)/itmax
qs=(1-xs)/2
z2=[]
for q in qs:
    dum=[(p/q)**((itmax+n))*((1-p)/(1-q))**((itmax-n))*P_bin((itmax+n)/2,q,itmax) for n in range(nmin,nmax+1,2)]
    z2.append(sum(np.array(dum)))
z2=np.array(z2)
plt.plot(dums,np.sqrt(z2/(7.543795923522006e-10)**2-1),lw=3,color="darkred",ls="--")
plt.scatter(ws,np.sqrt(Vars/(Ps**2)),s=100)
plt.savefig("images/Relative_error_UBRW_tilted.pdf",bbox_inches="tight")
plt.show();plt.close()

Plot_bonito(xlabel=r"$n_{k^\dag}$",ylabel=r"$z$",y_size=4,x_size=5)
plt.errorbar(ws,Ps,yerr=2*errs,capsize=10,fmt="none",color="darkblue",lw=1)
plt.scatter(ws,Ps)
plt.yticks(np.array([7.4,7.6,7.8])*10**(-10),[r"$7.4\cdot10^{-10}$",r"$7.6\cdot10^{-10}$",r"$7.8\cdot10^{-10}$"])
plt.plot(ws,np.ones(len(ws))*7.543795923522006e-10,color="darkred",ls="--",lw=3)
# plt.savefig("images/Pa_UBRW_tilted.pdf",bbox_inches="tight")
plt.show();plt.close()

#%% Importance sampling on last point Backtrack process Fig. 3

#WARNING! The final distribution for the Backtrack process have to respect that in odd (even) times the process can only be in odd (even) states (given that n0 is even).

realiz=100000
Ps=[];errs=[];Vars=[];P2s=[]
ws=np.array([2**i for i in range(2,9)])
for ii in ws:
    w=ii/2
    Nplus=np.random.randint(itmax/2-w,itmax/2+w+1,size=realiz) #Generate number of jumps=+1 with uniform distribution
    Xfs=2*Nplus-itmax #Last positions
    dum=vIa(Xfs,nmin,nmax)*P_bin(n=Nplus,p=p,t=itmax)*2*w
    P=np.mean(dum)
    P2=np.mean(dum*dum)
    Var=abs(P2-P**2)
    err=np.sqrt(Var/realiz)
    Ps.append(P);errs.append(err);Vars.append(Var);P2s.append(P2)
Ps=np.array(Ps);errs=np.array(errs);Vars=np.array(Vars);P2s=np.array(P2s)
#np.savetxt("data/prob_hit_A_UBRW_backtrack_variying_uniform_distrib.dat",np.c_[wfs,Ps,errs,Vars]) 

print("Result Backtracking=",P)

Plot_bonito(xlabel=r"$D$",ylabel=r"$\sigma_Z$"+"/"+r"$z$",y_size=4,x_size=5)
plt.xscale("log")
plt.xlim([1,10**3])
plt.scatter(ws,np.sqrt(Vars/(Ps**2)),s=100)
dums=np.linspace(0,ws[-1],1000)
plt.plot(dums,np.sqrt(dums*s2/s**2-1),lw=3,ls="--",color="darkred")
plt.savefig("images/Relative_error_UBRW_Backtrack.pdf",bbox_inches="tight")
plt.show();plt.close()

Plot_bonito(xlabel=r"$D$",ylabel=r"$z$",y_size=4,x_size=5)
plt.yticks(np.array([2,4,6,8])*10**(-10),[r"$2\cdot10^{-10}$",r"$4\cdot10^{-10}$",r"$6\cdot10^{-10}$",r"$8\cdot10^{-10}$"])
plt.xscale("log")
plt.xlim([1,10**3])
plt.errorbar(ws,Ps,yerr=2*errs,capsize=10,fmt="none",color="darkblue",lw=1)
plt.scatter(ws,Ps)
plt.plot(ws,np.ones(len(ws))*7.543795923522006e-10,color="darkred",ls="--",lw=3)
# plt.savefig("images/Pa_UBRW_Backtrack.pdf",bbox_inches="tight")
plt.show();plt.close()
