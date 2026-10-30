"""Original mean-centered top-five class similarity functions."""
import numpy as np

def norm(x):
 a=np.array(x,dtype=float,copy=True);a-=a.mean(axis=-1,keepdims=True)
 return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def class5(x,z,labs,classes):
 sim=norm(x)@norm(z).T
 return np.stack([np.sort(sim[:,labs==c],axis=1)[:,-min(5,int((labs==c).sum())):].mean(axis=1) for c in classes],axis=1)
