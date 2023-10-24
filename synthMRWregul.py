import numpy as np
import torch


def synthMRWregul(N,c1,c2,L,epsilon=0.2,win=1):
    
    '''
    sig=synthMRWregul(N,c1,c2,epsilon,L,win)
    
    Synthesis of a multifractal  motion with a singularity spevtrum
    zeta(q)=(c1+c2)q-c2 q^2/2. 
    
    Input :
        N : size of the signalto generate
        c1 : parameter of long-range dependance
        c2 : parameter of intermittency
        epsilon : size of the small-scale regularization (typically epsilon=2)
              Default value is 0.2)
         L : size of the integral scale. L must verify L<N/16. 
              Default value is N/16)
         win : type of the large scale regularization function
              if win==1 : gaussian function
              if win==2 : bump function
              Defaultvalue is 1.
              
    Output :
        sig : the synthetized fBm
        
    Example  :
        H=1/3;
        N=2^25;
        epsilon=2;
        L=2^14;
        sig=synthfbmRegul(N,H,L,epsilon)
        plot(sig)
        
    %%
    % S. Roux, ENS Lyon, 03/09/2017
    % Python version C. Granero Belinchon, IMT Atlantique 09/2023
    '''
    
    # Regularised norm
    #RegulNorm=inline('sqrt(x.^2+epsilon^2)','x','epsilon'); % norme regularisé
    RegulNorm = lambda x, epsilon : np.sqrt(x**2+epsilon**2)
    dx=1/N
    L=L/N
    epsilon=epsilon*dx
    alpha=3/2-c1
    
    x=np.zeros(N,)
    n2=np.zeros(N,)

    # Definition of x-axis
    x[0:int(N/2)+1]=np.arange(0,N/2+1,1)*dx
    x[int(N/2)+1:N]=np.arange(-N/2+1,0,1)*dx
    
    # Scaling
    y=x/RegulNorm(x,epsilon)**alpha
    # Large scale function
    
    if win==1: # Gaussian with std=L
        if L > 1/8:
            print('L must be greater than N/8 for right scaling')
        
        G=1/np.sqrt(2*np.pi*L**2)*np.exp(-x**2/2/L**2)
    if win==2: # exp(x^/(L^2-x^2))
        if L > 1/2:
            print('L must be greater than N/2 for right scaling')
        
        G=x*0
        ii=np.where(abs(x)<L)
        G[ii]=np.exp(-(x[ii])**2/(L**2-x[ii]**2))
        G=G/sum(G)
    
    #------ Synthesis of the covariance of the correlated noise c2*log(|x|)
    if c2 >0:
        L=L/dx
        n2[0:int(N/2)+1]=np.minimum(np.arange(0,N/2+1,1),L)
        n2[int(N/2)+1:N]=np.maximum(np.arange(-N/2+1,0,1),-L)
    
        mycorr=c2*np.log(L/RegulNorm(abs(n2),1))
        L2=np.real(np.fft.fft(mycorr))
        x2=np.real(np.fft.ifft(np.fft.fft(np.random.randn(N,))*np.sqrt(L2)))
        
        Xr =np.exp(x2)
        
    else:
        Xr=np.ones(N,)
    
    
    # -- synthesis of MRW
    bb=np.random.randn(N,)
    sig =np.real(np.fft.ifft(np.fft.fft(bb*np.sqrt(dx)*Xr)*np.fft.fft(y*G)))
    sig=sig/np.std(sig)
    
    return sig

#%% Data generation for Classication problem

# We fix two parameters N and win
N=2**20
win=1 # large scale window 1 : gaussian, 2 : bump 

# With N=2**20 we are generating 32 signals of N**15 and L must be <2**16

# We define the other variables that will change between generations
c1s=np.array((0.2,0.4,0.6,0.8))
c2s=np.array((0.02,0.04,0.06,0.08))
Ls=np.arange(1000,5001,1000)
epsilons=np.arange(0.5,5.1,1)

mrw=np.zeros((len(c1s),len(c2s),len(Ls),len(epsilons),N))

for ic1 in range(len(c1s)):
    for ic2 in range(len(c2s)):
        for iL in range(len(Ls)):
            for iepsilon in range(len(epsilons)):
                print(ic1/len(c1s),ic2/len(c2s),iL/len(Ls),iepsilon/len(epsilons))
                c1=c1s[ic1]
                c2=c2s[ic2]
                L=Ls[iL]
                epsilon=epsilons[iepsilon]
                mrw[ic1,ic2,iL,iepsilon,:]=synthMRWregul(N,c1,c2,L,epsilon,win)

np.savez('Pre_MRW.npz',mrw=mrw,c1s=c1s,c2s=c2s,Ls=Ls,epsilons=epsilons,N=N,win=win)
  
#%% Cette fonction vous permet d'etudier la variance, skewness et flatness des increments du processus

def analyseIncrsTorchcuda(signal,scales, device='cpu'):

    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''  
    
    Nreal=signal.size()[0]
    Struc=torch.zeros((Nreal,3,len(scales)), dtype=torch.float32, device=device)
        
    for ir in range(Nreal):
        
        # We normalize the image by centering and standarizing it
        nanstdtmp=torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp=(signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

        for isc in range(len(scales)):
                
            scale=int(scales[isc])
                
            incrs=tmp[scale:]-tmp[:-scale]
            incrs=incrs[~torch.isnan(incrs)]
            Struc[ir,0,isc]=torch.log(torch.nanmean(incrs.flatten()**2))
            nanstdincrs=torch.sqrt(torch.nanmean(torch.abs(incrs-torch.nanmean(incrs))**2))
            incrsnorm=(incrs-torch.nanmean(incrs))/nanstdincrs
            Struc[ir,1,isc]=torch.nanmean(incrsnorm.flatten()**3)
            Struc[ir,2,isc]=torch.nanmean(incrsnorm.flatten()**4)/3
        
    return Struc
              