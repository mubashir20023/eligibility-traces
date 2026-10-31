"""Tabular check (26 Oct 2026): quit option at every chain state. Result: every method learns to quit (return ~0.5); exploration failure, uninformative."""
import numpy as np, sys
def make(m,n):
    return 1+m*n+1
def run(m, sigma, eta, alpha, seed, n=6, lam=0.8, gamma=0.99, eps=0.1, episodes=1500, c=0.5):
    rng=np.random.default_rng(seed); S=make(m,n); b=S-1; A=2
    Q=np.zeros((S,A)); Z=np.zeros((S,A,S,A)); cnt=np.zeros((S,A)); rets=[]
    for ep in range(episodes):
        ch=rng.integers(m); s=0; pos=-1; e=np.zeros((S,A)); y=np.zeros((S,A)); R=0; done=False
        while not done:
            g=int(np.argmax(Q[s]+1e-9*rng.random(A))); a=int(rng.integers(A)) if rng.random()<eps else g
            if s==0: s2=1+ch*n; pos=0; r=0.0
            elif s==b: r=1.0+sigma*rng.standard_normal(); done=True; s2=s
            elif a==1: r=c; done=True; s2=s
            elif pos<n-1: pos+=1; s2=s+1; r=0.0
            else: s2=b; r=0.0
            R+=r
            delta=r+(0 if done else gamma*Q[s2].max())-Q[s,a]
            x=np.zeros((S,A)); x[s,a]=1; e=gamma*lam*e+x; cnt[s,a]+=1; Z[s,a]+=(e-Z[s,a])/cnt[s,a]
            y=(1-eta)*Z[s,a]+eta*(gamma*lam*y+x); Q+=alpha*delta*y
            if a!=g: e[:]=0; y[:]=0
            s=s2
        rets.append(R)
    return np.mean(rets)
for sigma in [0.0,1.0,2.0]:
    for m in [1,8,32]:
        out=[]
        for name,eta in [("Q(l)",1.0),("ET.3",0.3),("ET0",0.0)]:
            best=max((np.mean([run(m,sigma,eta,a,s) for s in range(8)]),a) for a in [0.1,0.3,0.5])
            out.append(f"{name}: {best[0]:.3f} (a={best[1]})")
        print(f"sigma={sigma} m={m:<3d} "+" | ".join(out), flush=True)
