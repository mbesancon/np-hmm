'''
Created on 8 Oct 2013

@author: James McInerney
'''

#Modular definition of sensors, in which each Sensor object must
#be able to specify what it adds to the expectation of the hidden state (E-step)
#and know how to optimise its own parameters given partitioned data (M-step in variational inference)

#from numpy import *
import numpy as np
from scipy.special.basic import digamma
from model.util import inv0

class Sensor(object):
    def __init__(self,K,hyperparams):
        self._K = K #truncation parameter
        self._hyperparams = hyperparams
        
    def loglik(self,X):
        """
        given data set X, provide the likelihood of each (N) data point.
        returns: (N,K) matrix, with unnormalised log liklihood 
        for each component and each data point 
        """        
        raise Exception('not implemented')
    
    def m(self,X,exp_z):
        #given expected value of z, calculate the
        #variational parameters of each component (w.r.t. this sensor)
        raise Exception('not implemented')
    
    def save(self,filepath):
        raise Exception('not implemented')
    
    def load(self,filepath):
        raise Exception('not implemented')
        
        
class MVGaussianSensor(Sensor):  
    
    def __init__(self,K,XDim,hyp=None):
        if hyp is None:
            #variance of mean (smaller: broader the means)
            hyp = {'beta0':(1e-20), 
                   'v0':XDim+2, #degrees of freedom in inverse wishart
                   'm0':np.zeros(XDim), #prior mean
                   'W0':(1e0)*np.eye(XDim), #prior cov (bigger: smaller covariance)
                }

        self._W = []
        for k in range(K): self._W.append(hyp['W0']) #init value of expW0
        Sensor.__init__(self,K,hyp)
        
    def randInit(self,K):
        0
        
    def loglik(self,X):
        """return log lik of each data point x latent component"""
        K = self._K
        (N,XDim)=np.shape(X)
        #calculate some features of the data:
        exp_diff_mu = self._eDiffMu(X,XDim,self._NK,self._betak,self._m,self._W,self._xd,self._vk,N,K) #eqn 10.64 Bishop
        exp_invc = self._eInvc(self._W,self._vk,XDim,K) #eqn 10.65 Bishop
        ln_lik = 0.5*exp_invc - 0.5*exp_diff_mu
        return ln_lik
    
    def m(self,X,exp_z,randInit=0):
        """optimise variational parameters"""
        (N,XDim) = np.shape(X)
        (N1,K) = np.shape(exp_z)
        #access hyperparameters
        v0 = self._hyperparams['v0']
        beta0 = self._hyperparams['beta0']
        m0 = self._hyperparams['m0']
        if 'alpha_sig0' in self._hyperparams and 'alpha_sig1' in self._hyperparams:
            #use hyperhyper parameters:
            print('using hyper-hyperparameters')
            #alpha sig0/sig1 are hyperhyper parameters for precision matrix            
            alpha_sig0,alpha_sig1 = self._hyperparams['alpha_sig0'],self._hyperparams['alpha_sig1']
            alpha_sigA = alpha_sig0 #- 0.5*(XDim*K*v0)
            tracek = np.array([np.trace(self._W[k]) for k in range(K)])
            alpha_sigB = ( (1/alpha_sig1) + 0.5*tracek.sum() )**(-1.)
            self._expW0 =  alpha_sigA*alpha_sigB*np.eye(XDim)
            assert alpha_sigA>0 and alpha_sigB>0
        else: 
            self._expW0 = self._hyperparams['W0']
        NK = exp_z.sum(axis=0)
        vk = v0 + NK + 1
        xd = self._mXd(exp_z,X)
        S = self._mS(exp_z,X,xd,NK)
        betak = beta0 + NK
        self._m = self._mM(K,XDim,beta0,m0,NK,xd,betak)
        self._W = self._mW(K,self._expW0,xd,NK,m0,XDim,beta0,S) 
        self._xd = xd
        self._S = S
        self._NK = NK
        self._vk = vk
        self._betak = betak
                
        #print 'mvg',str(self)
        #sys.exit(0)
        #[no return value]

    def save(self,filepath):
        np.save(filepath+'_K',self._K)
        np.save(filepath+'_NK',self._NK)
        np.save(filepath+'_betak',self._betak)
        np.save(filepath+'_m',self._m)
        np.save(filepath+'_W',self._W)
        np.save(filepath+'_xd',self._xd)
        np.save(filepath+'_vk',self._vk)
        print('saved MVGSensor',filepath)
        
    def load(self,filepath):
        self._K = np.load(filepath+'_K.npy')
        self._NK = np.load(filepath+'_NK.npy')
        self._betak = np.load(filepath+'_betak.npy')
        self._m = np.load(filepath+'_m.npy')
        self._W = np.load(filepath+'_W.npy')
        self._xd = np.load(filepath+'_xd.npy')
        self._vk = np.load(filepath+'_vk.npy')
        print('loaded MVGSensor',filepath)

    
    def __str__(self):
        return 'means:\n%s\ncovs:\n%s\n'%(str(self._m),str(self.expC()))
    
        
    def _eInvc(self,W,vk,XDim,K):
        invc = [None for _ in range(K)]
        for k in range(K):
            dW = np.linalg.det(W[k])
            #print 'dW',dW
            if dW>1e-30: 
                ld = np.log(dW)
            else: 
                ld = 0.0
            invc[k] = sum([digamma((vk[k]+1-i) / 2.) for i in range(XDim)]) + XDim*np.log(2) + ld
        return np.array(invc)

    def _eDiffMu(self,X,XDim,NK,betak,m,W,xd,vk,N,K):
        Mu = np.zeros((N,K))
        A = XDim / betak #shape: (k,)
        for k in range(K):
            B0 = (X - m[k,:]).T
            B1 = np.dot(W[k], B0)
            #l = dot(B0.T, B1)
            l = (B0*B1).sum(axis=0)
            assert np.shape(l)==(N,),np.shape(l)
            Mu[:,k] = A[k] + vk[k]*l #shape: (n,k)
        return Mu

    def _eDiffMuSlow(self,X,XDim,NK,betak,m,W,xd,vk,N,K):
        Mu = np.zeros((N,K))
        for n in range(N):
            for k in range(K):
                A = XDim / betak[k] #shape: (k,)
                B0 = np.reshape((X[n,:] - m[k,:]),(XDim,1))
                B1 = np.dot(W[k], B0)
                l = np.dot(B0.T, B1)
                assert np.shape(l)==(1,1),np.shape(l)
                Mu[n,k] = A + vk[k]*l #shape: (n,k)
        return Mu
    
    def _mXd(self,Z,X):
        """weighted means by component responsibilites"""
        (N,XDim) = np.shape(X)
        (N1,K) = np.shape(Z)
        NK = Z.sum(axis=0)
        assert N==N1
        xd = np.zeros((K,XDim))
        for k in range(K):
            xd[k,:] = (np.reshape(Z[:,k],(N,1))*X).sum(axis=0)
        #safe divide:
        for k in range(K):
            if NK[k]>0: xd[k,:] = xd[k,:]/NK[k]
        return xd

    def _mXdSlow(self,Z,X):
        """weighted means (by component responsibilites)"""
        (N,XDim) = np.shape(X)
        (N1,K) = np.shape(Z)
        NK = Z.sum(axis=0)
        assert N==N1
        xd = np.zeros((K,XDim))
        for n in range(N):
            for k in range(K):
                xd[k,:] += Z[n,k]*X[n,:]
        #safe divide:
        for k in range(K):
            if NK[k]>0: xd[k,:] = xd[k,:]/NK[k]
        
        return xd
    
    def _mS(self,Z,X,xd,NK):
        (N,K)=np.shape(Z)
        (N1,XDim)=np.shape(X)
        assert N==N1
        S = [np.zeros((XDim,XDim)) for _ in range(K)]
        for k in range(K):
            B0 = np.reshape(X - xd[k,:], (N,XDim))
            for d0 in range(XDim):
                for d1 in range(XDim):
                    L = B0[:,d0]*B0[:,d1]
                    S[k][d0,d1] += (Z[:,k]*L).sum()
        #safe divide:
        for k in range(K):
            if NK[k]>0: S[k] = S[k]/NK[k]
        return S

    def _mSSlow(self,Z,X,xd,NK):
        (N,K)=np.shape(Z)
        (N1,XDim)=np.shape(X)
        assert N==N1        
        S = [np.zeros((XDim,XDim)) for _ in range(K)]
        for n in range(N):
            for k in range(K):
                B0 = np.reshape(X[n,:]-xd[k,:], (XDim,1))
                L = np.dot(B0,B0.T)
                assert np.shape(L)==np.shape(S[k]),np.shape(L)
                S[k] += Z[n,k]*L
        #safe divide:
        for k in range(K):
            if NK[k]>0: S[k] = S[k]/NK[k]
        return S
    
    def expC(self):
        """calculate expected covariance matrix for each component"""
        return np.array([inv0(Wk*vk) for (Wk,vk) in zip(self._W,self._vk)])
    
    def _mW(self,K,W0,xd,NK,m0,XDim,beta0,S):
        Winv = [None for _ in range(K)]
        for k in range(K): 
            Winv[k]  = NK[k]*S[k] + inv0(W0)
            Q0 = np.reshape(xd[k,:] - m0, (XDim,1))
            q = np.dot(Q0,Q0.T)
            Winv[k] += (beta0*NK[k] / (beta0 + NK[k]) ) * q
            assert np.shape(q)==(XDim,XDim)
        W = []
        for k in range(K):
            try:
                W.append(inv0(Winv[k]))
            except np.linalg.LinAlgError:
                #print 'Winv[%i]'%k, Winv[k]
                raise np.linalg.LinAlgError()
        return W
    
    def _mM(self,K,XDim,beta0,m0,NK,xd,betak):
        m = np.zeros((K,XDim))
        for k in range(K): m[k,:] = (beta0*m0 + NK[k]*xd[k,:]) / betak[k]
        return m  


class DiscreteSensor(Sensor):  
    def __init__(self,K,hyp=None):
        if hyp is None:
            hyp = {'c':1.0}
        Sensor.__init__(self,K,hyp)
    def loglik(self,X):
        """
        given data set X, provide the likelihood of each (N) data point
        returns: (N,K) matrix, with unnormalised log liklihood for each 
        component and each data point 
        """        
        return np.dot(X,self._exp_ln_ctk)
    def m(self,X,exp_z):
        """
        given expected value of z, calculate the variational 
        parameters of each component (w.r.t. this sensor)
        """
        alpha_c = self._hyperparams['c'] 
        tau_ctk = alpha_c + np.dot(X.T, exp_z)
        #assumes symmetric prior (easy to change to non-symmetric prior)
        self._tau_ctk = tau_ctk        
        self._exp_ctk = tau_ctk / tau_ctk.sum(axis=0)
        self._exp_ln_ctk = digamma(tau_ctk) - digamma(tau_ctk.sum(axis=0))

if __name__ == "__main__":
    from testing.run import test1
    test1()