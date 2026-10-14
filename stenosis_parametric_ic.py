"""Four bonded particles in a fixed stenosis: parametric initial positions.
Fluid transient input: (x,y,t,x_a0,y_a0,x_b0,y_b0).
Trajectory input: (t,x_a0,y_a0,x_b0,y_b0); initial velocities are zero.
The steady carrier remains independent of particle initial positions.
Run --self-test, --quick-test, or see README.md for training and evaluation.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
try:
    import torch
    from torch import nn
except ImportError:
    torch = nn = None


@dataclass
class Config:
    L: float = 1.0
    H: float = 0.4
    xc: float = 0.4
    R: float = 0.1
    Re: float = 50.0
    U: float = 1.0
    T: float = 0.7
    St: float = 0.05
    beta: float = 5e-5
    rp: float = 0.015
    eps: float = 0.025
    q0: tuple = ((0.13,0.14),(0.19,0.14),(0.19,0.26),(0.13,0.26))
    bond_kn: float = 400.0  # k / reduced mass, as in the supplied code
    bond_cn: float = 20.0   # damping / reduced mass
    pair_kn: float = 1e5
    pair_cn: float = 100.0
    wall_kn: float = 2e5
    wall_cn: float = 150.0
    width: int = 48
    depth: int = 4
    batch: int = 512
    boundary: int = 160
    particle_points: int = 192
    steady_adam: int = 6000
    steady_lbfgs: int = 1500
    coupled_adam: int = 5000
    coupled_lbfgs: int = 800
    pde_chunk: int = 192
    primitive_weight: float = 1.0
    continuity_weight: float = 4.0
    steady_lr: float = 0.0005
    coupled_lr: float = 0.0003
    repair_adam: int = 1500
    repair_lbfgs: int = 1200
    refinement_rounds: int = 2
    log_every: int = 250
    seed: int = 1234
    device: str = 'auto'
    outdir: str = 'results_parametric'
    variable_beads: tuple = (0, 1)  # zero-based particle indices
    ic_bounds: tuple = ((0.10, 0.18), (0.09, 0.20), (0.15, 0.24), (0.09, 0.20))
    initial_clearance: float = 0.002
    rest_lengths: str = 'case'  # case equilibrium (original model) or nominal prestress
    validation_cases: int = 8
    screening_points: int = 3072
    screening_times: int = 7

    def validate(self):
        positive=('L','H','Re','U','T','St','rp','eps','width','depth','batch',
                  'boundary','particle_points','log_every','pde_chunk',
                  'primitive_weight','continuity_weight','steady_lr','coupled_lr')
        for k in positive:
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0:
                raise ValueError(k+' must be finite and positive')
        for k in ('beta','bond_kn','bond_cn','pair_kn','pair_cn','wall_kn','wall_cn'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<0:
                raise ValueError(k+' must be finite and nonnegative')
        for k in ('steady_adam','steady_lbfgs','coupled_adam','coupled_lbfgs',
                  'refinement_rounds','repair_adam','repair_lbfgs'):
            if getattr(self,k)<0: raise ValueError(k+' cannot be negative')
        if not (0<self.R<self.H/2 and self.R<self.xc<self.L-self.R):
            raise ValueError('Invalid stenosis geometry')
        q=np.asarray(self.q0,dtype=float)
        if q.shape!=(4,2) or not np.isfinite(q).all(): raise ValueError('Exactly four initial centers required')
        if np.any((q[:,0]<=0)|(q[:,0]>=self.L)) or gaps_np(q,self).min()<=self.rp:
            raise ValueError('Initial bead intersects a wall or end plane')
        if 2*self.rp>=self.H-2*self.R: raise ValueError('Beads do not fit through throat')
        if any(np.linalg.norm(q[i]-q[j])<=2*self.rp for i,j in EDGES):
            raise ValueError('Initial beads overlap')
        ids = tuple(self.variable_beads)
        if len(ids) != 2 or len(set(ids)) != 2 or any(i not in range(4) for i in ids):
            raise ValueError('variable_beads must contain two distinct indices in 0..3')
        box = np.asarray(self.ic_bounds, dtype=float)
        if box.shape != (4,2) or not np.isfinite(box).all() or np.any(box[:,0] >= box[:,1]):
            raise ValueError('ic_bounds must contain four finite increasing intervals')
        if not math.isfinite(self.initial_clearance) or self.initial_clearance < 0:
            raise ValueError('initial_clearance must be nonnegative')
        if self.rest_lengths not in ('case', 'nominal'):
            raise ValueError('rest_lengths must be case or nominal')
        if self.validation_cases < 1 or self.screening_points < 1 or self.screening_times < 2:
            raise ValueError('positive validation counts required; screening_times >= 2')
        require_valid_initial(nominal_theta(self), self)
        # Verify a nonempty admissible training domain without changing the RNG used for training.
        sample_initial(np.random.default_rng(self.seed+777), 8, self)
        # Symmetric steady carrier is valid for symmetric geometry/inlet only.
        if self.device not in ('auto','cpu','cuda'): raise ValueError('device: auto/cpu/cuda')


EDGES = tuple(itertools.combinations(range(4),2))

def bounds_np(x,c):
    b=np.sqrt(np.maximum(c.R**2-(np.asarray(x)-c.xc)**2,0))
    return b,c.H-b

def gaps_np(q,c):
    return np.stack((q[...,1],c.H-q[...,1],
        np.linalg.norm(q-[c.xc,0],axis=-1)-c.R,
        np.linalg.norm(q-[c.xc,c.H],axis=-1)-c.R),-1)

def nominal_theta(c):
    return np.asarray(c.q0, dtype=float)[list(c.variable_beads)].reshape(4)


def initial_positions_np(theta, c):
    theta = np.asarray(theta, dtype=float)
    if theta.shape[-1:] != (4,):
        raise ValueError('initial-position input must have four coordinates')
    q = np.broadcast_to(np.asarray(c.q0), (*theta.shape[:-1],4,2)).copy()
    for k,i in enumerate(c.variable_beads): q[...,i,:] = theta[...,2*k:2*k+2]
    return q


def valid_initial(theta, c):
    theta=np.asarray(theta,dtype=float); q=initial_positions_np(theta,c)
    box=np.asarray(c.ic_bounds)
    ok=np.isfinite(theta).all(-1)&((theta>=box[:,0])&(theta<=box[:,1])).all(-1)
    ok &= ((q[...,0] > c.rp+c.initial_clearance)&
           (q[...,0] < c.L-c.rp-c.initial_clearance)).all(-1)
    ok &= (gaps_np(q,c) > c.rp+c.initial_clearance).all(axis=(-1,-2))
    for i,j in EDGES:
        ok &= np.linalg.norm(q[...,i,:]-q[...,j,:],axis=-1) > 2*c.rp+c.initial_clearance
    return ok


def require_valid_initial(theta,c):
    if not np.all(valid_initial(theta,c)):
        raise ValueError('Initial positions are outside ic_bounds, overlap, or violate wall/end clearance')


def sample_initial(rng,n,c):
    box=np.asarray(c.ic_bounds); blocks=[];count=0
    for _ in range(1000):
        a=rng.uniform(box[:,0],box[:,1],size=(max(32,2*(n-count)),4))
        a=a[valid_initial(a,c)];blocks.append(a);count+=len(a)
        if count>=n: return np.concatenate(blocks)[:n]
    raise ValueError('Cannot sample admissible initial positions; revise bounds/clearance')


def initial_positions(theta,c):
    return torch.stack([theta[:,2*list(c.variable_beads).index(i):2*list(c.variable_beads).index(i)+2]
        if i in c.variable_beads else theta.new_tensor(c.q0[i]).expand(len(theta),2)
        for i in range(4)],1)


def normalized_theta(theta,c):
    box=theta.new_tensor(c.ic_bounds)
    return 2*(theta-box[:,0])/(box[:,1]-box[:,0])-1


def condition_like(t,c,theta=None):
    if theta is None: theta=t.new_tensor(nominal_theta(c))
    else: theta=torch.as_tensor(theta,dtype=t.dtype,device=t.device)
    if theta.ndim==1: theta=theta[None]
    if theta.shape==(1,4): theta=theta.expand(len(t),4)
    if theta.shape!=(len(t),4): raise ValueError('Expected one condition or one four-coordinate condition per row')
    return theta


def mechanics_np(q,v,c,q0=None):
    """Independent NumPy spring, dashpot, disk/plane and soft-sphere forces."""
    q,v=np.asarray(q).reshape(4,2),np.asarray(v).reshape(4,2)
    a=np.zeros((4,2));q0=np.asarray(c.q0 if q0 is None else q0);active=(q[:,0]>0)&(q[:,0]<c.L)
    for i,j in EDGES:
        dr=q[i]-q[j];d=np.linalg.norm(dr)
        if d<1e-12: raise FloatingPointError('Coincident bead centers')
        n=dr/d;s=np.dot(v[i]-v[j],n);l0=np.linalg.norm(q0[i]-q0[j])
        fmu=-(c.bond_kn*(d-l0)+c.bond_cn*s)*n
        a[i]+=0.5*fmu;a[j]-=0.5*fmu  # equal inertial shares; mu/m=1/2
        if active[i] and active[j] and d<2*c.rp:
            ac=0.5*max(c.pair_kn*(2*c.rp-d)-c.pair_cn*s,0)*n
            a[i]+=ac;a[j]-=ac
    for i in range(4):
        if not active[i]: continue
        normals=[np.array([0.,1.]),np.array([0.,-1.])]
        for cy in (0.,c.H):
            dr=q[i]-[c.xc,cy];normals.append(dr/max(np.linalg.norm(dr),1e-12))
        for g,n in zip(gaps_np(q[i],c),normals):
            if g<c.rp:a[i]+=max(c.wall_kn*(c.rp-g)-c.wall_cn*np.dot(v[i],n),0)*n
    return a


class MLP(nn.Module if nn else object):
    def __init__(self,nin,nout,width,depth):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in
            zip([nin]+[width]*depth,[width]*depth+[nout])])
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.normal_(self.layers[-1].weight,std=.002)

    def forward(self,x):
        for i,l in enumerate(self.layers):
            x=l(x)
            if i<len(self.layers)-1:x=torch.tanh(x)
        return x

    def jac(self,x,j):
        """All first input derivatives in one pass; still trainable by backprop."""
        for i,l in enumerate(self.layers):
            x=l(x);j=torch.einsum('oi,bik->bok',l.weight,j)
            if i<len(self.layers)-1:
                x=torch.tanh(x);j=(1-x.square())[:,:,None]*j
        return x,j

    def time_derivatives(self,t,T):
        x=torch.cat((2*t[:,:1]/T-1,t[:,1:]),1)
        d=torch.cat((torch.full_like(t[:,:1],2/T),torch.zeros_like(t[:,1:])),1);dd=torch.zeros_like(x)
        for i,l in enumerate(self.layers):
            x=l(x);d=d@l.weight.T;dd=dd@l.weight.T
            if i<len(self.layers)-1:
                x=torch.tanh(x);g=1-x.square()
                dd=g*dd-2*x*g*d.square();d=g*d
        return x,d,dd


class Fluid(nn.Module if nn else object):
    def __init__(self,c):
        super().__init__();self.c=c
        self.base=MLP(2,6,c.width,c.depth)
        self.delta=MLP(7,6,c.width,c.depth)
        nn.init.zeros_(self.delta.layers[-1].weight);nn.init.zeros_(self.delta.layers[-1].bias)

    def fields(self,z,steady=False,jac=False):
        c=self.c;x,y=z[:,:1],z[:,1:2];xi=x/c.L;s=2*y/c.H-1
        f=xi*(1-s.square());vf=f*(1-xi);ps=max(c.U**2,8*c.U*c.L/(c.Re*c.H**2))
        p=8*c.U*(c.L-x)/(c.Re*c.H**2)
        u=c.U*(1-s.square());shear=-4*c.U*s/(c.Re*c.H)
        value=torch.cat((u,0*u,p,-p,shear,-p),1)
        zero=x*0;one=zero+1
        factors=torch.cat((c.U*f,c.U*vf*s,ps*one,ps*one,ps*s,ps*one),1)
        features=torch.cat((2*xi-1,2*s.square()-1),1)
        if jac:
            dim=z.shape[1];dvalue=z.new_zeros((len(z),6,dim))
            dvalue[:,0,1]=-4*c.U*s[:,0]/c.H
            dvalue[:,2,0]=-8*c.U/(c.Re*c.H**2)
            dvalue[:,3,0]=dvalue[:,5,0]=8*c.U/(c.Re*c.H**2)
            dvalue[:,4,1]=-8*c.U/(c.Re*c.H**2)
            jf=z.new_zeros((len(z),2,dim));jf[:,0,0]=2/c.L;jf[:,1,1]=8*s[:,0]/c.H
            raw,jraw=self.base.jac(features,jf)
            df=z.new_zeros((len(z),6,dim))
            fx=(1-s.square())/c.L;fy=-4*xi*s/c.H
            df[:,0,0]=c.U*fx[:,0];df[:,0,1]=c.U*fy[:,0]
            df[:,1,0]=(c.U*s*(1-s.square())*(1-2*xi)/c.L)[:,0]
            df[:,1,1]=(c.U*(s*fy*(1-xi)+2*vf/c.H))[:,0]
            df[:,4,1]=2*ps/c.H
            dvalue=dvalue+factors[:,:,None]*jraw+df*raw[:,:,None]
        else:raw=self.base(features)
        value=value+factors*raw
        if not steady:
            t=z[:,2:3];gate=-torch.expm1(-t/c.St)
            theta=condition_like(t,c,z[:,3:] if z.shape[1]==7 else None)
            features=torch.cat((2*xi-1,s,2*t/c.T-1,normalized_theta(theta,c)),1)
            factors=torch.cat((c.U*f*gate,c.U*vf*gate,ps*one,ps*one,ps*one,ps*one),1)
            loading=c.beta/max(c.beta,1e-4)
            if jac:
                dim=z.shape[1];jf=z.new_zeros((len(z),7,dim))
                jf[:,0,0]=2/c.L;jf[:,1,1]=2/c.H;jf[:,2,2]=2/c.T
                if dim==7:
                    box=z.new_tensor(c.ic_bounds)
                    for k in range(4): jf[:,3+k,3+k]=2/(box[k,1]-box[k,0])
                raw,jraw=self.delta.jac(features,jf);df=z.new_zeros((len(z),6,dim))
                df[:,0,0]=(c.U*fx*gate)[:,0];df[:,0,1]=(c.U*fy*gate)[:,0]
                df[:,1,0]=(c.U*(1-s.square())*(1-2*xi)/c.L*gate)[:,0]
                df[:,1,1]=(c.U*fy*(1-xi)*gate)[:,0]
                dg=torch.exp(-t/c.St)/c.St
                df[:,0,2]=(c.U*f*dg)[:,0];df[:,1,2]=(c.U*vf*dg)[:,0]
                dvalue=dvalue+loading*(factors[:,:,None]*jraw+df*raw[:,:,None])
            else:raw=self.delta(features)
            value=value+loading*factors*raw
        return (value,dvalue) if jac else value


class Parcels(nn.Module if nn else object):
    def __init__(self,c):
        super().__init__();self.c=c;self.net=MLP(5,8,c.width,3)
        self.register_buffer('q0',torch.tensor(c.q0,dtype=torch.float64))
        self.register_buffer('u0',torch.zeros(4,2,dtype=torch.float64))

    def motion(self,t,theta=None):
        if t.shape[1]==5:
            if theta is not None: raise ValueError('Provide conditions once')
            theta=t[:,1:];t=t[:,:1]
        c=self.c;theta=condition_like(t,c,theta);q0=initial_positions(theta,c)
        inputs=torch.cat((t,normalized_theta(theta,c)),1)
        raw,dr,ddr=self.net.time_derivatives(inputs,c.T)
        b=self.u0[None]+c.U*raw.reshape(-1,4,2)
        db=c.U*dr.reshape(-1,4,2);ddb=c.U*ddr.reshape(-1,4,2)
        h=(t+c.St*torch.expm1(-t/c.St))[:,:,None]
        dh=(-torch.expm1(-t/c.St))[:,:,None];ddh=(torch.exp(-t/c.St)/c.St)[:,:,None]
        return q0+h*b,dh*b+h*db,ddh*b+2*dh*db+h*ddb


def mechanics(q,v,c,q0=None):
    """Batched four-bead forces. Connector itself is massless and has no wall surface."""
    active=((q[:,:,0]>0)&(q[:,:,0]<c.L)).to(q.dtype)
    acc=q*0;q0=q.new_tensor(c.q0)[None].expand_as(q) if q0 is None else q0
    for i,j in EDGES:
        dr=q[:,i]-q[:,j];d=dr.square().sum(-1).clamp_min(1e-24).sqrt();n=dr/d[:,None]
        s=((v[:,i]-v[:,j])*n).sum(-1);l0=torch.linalg.vector_norm(q0[:,i]-q0[:,j],dim=-1)
        f=-(c.bond_kn*(d-l0)+c.bond_cn*s)[:,None]*n/2
        overlap=2*c.rp-d
        pair=.5*(c.pair_kn*overlap-c.pair_cn*s).clamp_min(0)*(overlap>0)*active[:,i]*active[:,j]
        f=f+pair[:,None]*n
        # No in-place edits on values needed by autograd.
        basis=q.new_zeros(4);basis[i]=1;basis[j]=-1
        acc=acc+f[:,None,:]*basis[None,:,None]
    y=q[:,:,1];up=torch.stack((y*0,y*0+1),-1)
    gaps=[y,c.H-y];normals=[up,-up]
    for cy in (0.,c.H):
        dr=q-q.new_tensor([c.xc,cy]);d=dr.square().sum(-1).clamp_min(1e-24).sqrt()
        gaps.append(d-c.R);normals.append(dr/d[:,:,None])
    gaps=torch.stack(gaps,-1);normals=torch.stack(normals,-2)
    o=c.rp-gaps;vn=(v[:,:,None,:]*normals).sum(-1)
    mag=(c.wall_kn*o-c.wall_cn*vn).clamp_min(0)*(o>0)
    acc=acc+(mag[:,:,:,None]*normals).sum(2)*active[:,:,None]
    return acc,active,gaps


class System(nn.Module if nn else object):
    def __init__(self,c):
        super().__init__();self.c=c;self.fluid=Fluid(c);self.parcels=Parcels(c)
        nodes,w=np.polynomial.legendre.leggauss(48)
        edges=[0,c.xc-c.R,c.xc,c.xc+c.R,c.L]
        xs=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in zip(edges[:-1],edges[1:])])
        ws=np.concatenate([(b-a)/2*w for a,b in zip(edges[:-1],edges[1:])])
        lo,hi=bounds_np(xs,c)
        for name,a in (('gx',xs),('gw',ws),('glo',lo),('ghi',hi)):
            self.register_buffer(name,torch.tensor(a,dtype=torch.float64))

    def drag_state(self,t,theta=None):
        if t.shape[1]==5:
            if theta is not None: raise ValueError('Provide conditions once')
            theta=t[:,1:];t=t[:,:1]
        c=self.c;theta=condition_like(t,c,theta)
        q,v,a=self.parcels.motion(t,theta)
        active=((q[:,:,0]>0)&(q[:,:,0]<c.L)).to(q.dtype)
        xy=torch.stack((q[:,:,0].clamp(0,c.L),q[:,:,1]),-1)
        z=torch.cat((xy.reshape(-1,2),t.repeat_interleave(4,0),theta.repeat_interleave(4,0)),1)
        uv=self.fluid.fields(z)[:,:2].reshape(-1,4,2)
        return q,v,a,active,(uv-v)/c.St*active[:,:,None]

    def normalizer(self,q):
        c=self.c;xp,yp=q[:,:,0:1],q[:,:,1:2]
        pdf=torch.exp(-.5*((self.gx[None,None,:]-xp)/c.eps).square())/(math.sqrt(2*math.pi)*c.eps)
        cdf=lambda z:.5*torch.erfc(-z/math.sqrt(2))
        vertical=cdf((self.ghi[None,None,:]-yp)/c.eps)-cdf((self.glo[None,None,:]-yp)/c.eps)
        return (pdf*vertical*self.gw[None,None,:]).sum(-1)

    def source(self,z):
        if self.c.beta==0:return z[:,:2]*0
        q,_,_,active,drag=self.drag_state(z[:,2:3],z[:,3:] if z.shape[1]==7 else None);norm=self.normalizer(q)
        if bool(((norm<1e-10)&active.bool()).any()):
            raise FloatingPointError('Unresolved source: predicted parcel left the fluid walls')
        pdf=torch.exp(-.5*((z[:,:2,None].transpose(1,2)-q)/self.c.eps).square().sum(-1))
        kernel=pdf/(2*math.pi*self.c.eps**2*norm.clamp_min(1e-10))
        mass=self.c.beta*(self.c.L*self.c.H-math.pi*self.c.R**2)/4
        return -mass*(drag*kernel[:,:,None]).sum(1)


def fluid_residual(s,z,steady=False,primitive=False):
    c=s.c
    if primitive:z=z.detach().requires_grad_(True)
    f,j=s.fluid.fields(z,steady,jac=True);u,v,p,xx,xy,yy=f.unbind(1)
    ux,uy=j[:,0,0],j[:,0,1];vx,vy=j[:,1,0],j[:,1,1]
    ut,vt=(u*0,v*0) if steady else (j[:,0,2],j[:,1,2])
    src=z[:,:2]*0 if steady else s.source(z);k=c.H/c.U**2
    rr=[(ux+vy)*c.H/c.U,(ut+u*ux+v*uy-j[:,3,0]-j[:,4,1]-src[:,0])*k,
        (vt+u*vx+v*vy-j[:,4,0]-j[:,5,1]-src[:,1])*k,
        (xx+p-2*ux/c.Re)/c.U**2,(xy-(uy+vx)/c.Re)/c.U**2,(yy+p-2*vy/c.Re)/c.U**2]
    if primitive:
        g=lambda a:torch.autograd.grad(a,z,torch.ones_like(a),create_graph=True)[0]
        lapu=g(ux)[:,0]+g(uy)[:,1];lapv=g(vx)[:,0]+g(vy)[:,1]
        rr.extend(((ut+u*ux+v*uy+j[:,2,0]-lapu/c.Re-src[:,0])*k,
                   (vt+u*vx+v*vy+j[:,2,1]-lapv/c.Re-src[:,1])*k))
    return torch.stack(rr,1)


class Samples:
    def __init__(self,s,seed,theta=None):
        self.fixed_theta=theta
        self.s=s;self.c=s.c;self.rng=np.random.default_rng(seed)
        self.device=next(s.parameters()).device
    def tensor(self,a):
        if not torch.is_tensor(a): a=np.asarray(a)
        return torch.as_tensor(a,device=self.device,dtype=torch.float64)
    def trajectory(self,t):
        t=self.tensor(t) if not torch.is_tensor(t) else t
        theta=(sample_initial(self.rng,len(t),self.c) if self.fixed_theta is None
               else np.broadcast_to(self.fixed_theta,(len(t),4)))
        return torch.cat((t,self.tensor(np.array(theta,copy=True))),1)
    def times(self,n,horizon):
        t=self.rng.uniform(0,horizon,(n,1));t[:n//5]*=min(1,3*self.c.St/horizon)
        return self.trajectory(t)
    def interior(self,n,steady,horizon):
        c=self.c;xy=[]
        while sum(len(a) for a in xy)<n:
            a=self.rng.uniform([0,0],[c.L,c.H],(n,2));lo,hi=bounds_np(a[:,0],c)
            xy.append(a[(a[:,1]>lo)&(a[:,1]<hi)])
        z=self.tensor(np.concatenate(xy)[:n])
        return z if steady else torch.cat((z,self.times(n,horizon)),1)
    def batch(self,steady,horizon,n=None,extra=None,flux_sections=15):
        c=self.c;n=n or c.batch;z=self.interior(n,steady,horizon)
        if extra is not None:z=torch.cat((z,extra),0)
        if not steady:
            t=self.times(96,horizon)
            with torch.no_grad():q=self.s.parcels.motion(t)[0].cpu().numpy()
            pos=q[np.arange(len(q)),self.rng.integers(0,4,len(q))]+self.rng.normal(0,c.eps,q[:,0].shape)
            lo,hi=bounds_np(pos[:,0],c);ok=(pos[:,0]>0)&(pos[:,0]<c.L)&(pos[:,1]>lo)&(pos[:,1]<hi)
            z=torch.cat((z,torch.cat((self.tensor(pos[ok]),t[self.tensor(ok).bool()]),1)))
        theta=(np.arange(c.boundary)+.5)/c.boundary*math.pi
        # Open-interval arc nodes avoid differentiating directly at sharp shoulders.
        arcs=np.concatenate((np.c_[c.xc+c.R*np.cos(theta),c.R*np.sin(theta)],
                             np.c_[c.xc+c.R*np.cos(theta),c.H-c.R*np.sin(theta)]))
        end=np.c_[np.full(c.boundary,c.L),self.rng.uniform(0,c.H,c.boundary)]
        tail=lambda a:self.tensor(a) if steady else torch.cat((self.tensor(a),self.times(len(a),horizon)),1)
        xn=np.unique(np.r_[np.linspace(.001,c.L-.001,flux_sections),
                            c.xc,c.xc-c.R,c.xc+c.R,c.xc-.5*c.R,c.xc+.5*c.R])
        nodes,w=np.polynomial.legendre.leggauss(40);lo,hi=bounds_np(xn,c)
        y=lo[:,None]+(hi-lo)[:,None]*(nodes+1)/2
        flux=self.tensor(np.c_[np.repeat(xn,40),y.ravel()])
        if not steady:flux=torch.cat((flux,self.times(len(xn),horizon).repeat_interleave(40,0)),1)
        return {'pde':z,'wall':tail(arcs),'end':tail(end),'flux':flux,
            'fw':self.tensor((hi-lo)[:,None]*w/2),'t':self.times(c.particle_points,horizon)}


def objective(s,d,stage):
    c=s.c;steady=stage=='steady';zero=next(s.parameters()).sum()*0;parts={}
    if stage!='particle':
        # This was missing from the supplied objective. It must reach backprop,
        # not just validation: train pressure and second velocity derivatives.
        parts['pde']=zero;parts['primitive']=zero
        for block in d['pde'].split(c.pde_chunk):
            r=fluid_residual(s,block,steady,primitive=True)
            fraction=len(block)/len(d['pde'])
            mixed_weights=r.new_tensor([c.continuity_weight,1,1,1,1,1])
            parts['pde']=parts['pde']+fraction*(r[:,:6].square().mean(0)*mixed_weights).sum()
            parts['primitive']=parts['primitive']+fraction*r[:,6:8].square().mean(0).sum()
        parts['wall']=(s.fluid.fields(d['wall'],steady)[:,:2]/c.U).square().sum(1).mean()
        f,j=s.fluid.fields(d['end'],steady,jac=True)
        parts['outlet']=(f[:,3]/c.U**2).square().mean()+(f[:,1]/c.U).square().mean()
        parts['outlet']+=((-f[:,2]+2*j[:,0,0]/c.Re)/c.U**2).square().mean()
        flow=(s.fluid.fields(d['flux'],steady)[:,0].reshape(d['fw'].shape)*d['fw']).sum(1)
        parts['flux']=(flow/(2*c.U*c.H/3)-1).square().mean()
    if not steady:
        q,v,a,active,drag=s.drag_state(d['t']);qinit=initial_positions(d['t'][:,1:],c) if c.rest_lengths=='case' else None
        ac,_,gaps=mechanics(q,v,c,qinit)
        parts['particle']=(c.St*(a-drag-ac)/c.U).square().sum((1,2)).mean()
        parts['domain']=((-gaps).clamp_min(0)/c.H).square().mul(active[:,:,None]).sum((1,2)).mean()
        excess=(c.rp-gaps-.1*c.rp).clamp_min(0)/c.rp
        parts['overlap']=excess.square().mul(active[:,:,None]).sum((1,2)).mean()
        for i,j in EDGES:
            distance=torch.linalg.vector_norm(q[:,i]-q[:,j],dim=1)
            parts['overlap']+=((1.8*c.rp-distance).clamp_min(0)/(2*c.rp)).square().mul(active[:,i]*active[:,j]).mean()
    weights={'pde':1,'primitive':c.primitive_weight,'wall':40,'outlet':5,
             'flux':10,'particle':3,'domain':50,'overlap':2}
    loss=sum((weights[k]*v for k,v in parts.items()),zero)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    return loss,parts


def train(s,sampler,stage,adam_steps,lbfgs_steps,history,extra=None,lr=None):
    c=s.c
    for p in s.parameters():p.requires_grad_(False)
    modules=[s.fluid.base] if stage=='steady' else [s.parcels.net]+([s.fluid.delta] if stage=='coupled' else [])
    for module in modules:
        for p in module.parameters():p.requires_grad_(True)
    params=[p for p in s.parameters() if p.requires_grad]
    rate=lr if lr is not None else (c.steady_lr if stage=='steady' else c.coupled_lr)
    opt=torch.optim.Adam(params,lr=rate)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(adam_steps,1),eta_min=min(1e-5,rate*.1))
    started=time.perf_counter()
    def record(step,loss,parts):
        row={'stage':stage,'step':step,'total':float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
        history.append(row)
        print(f"{stage:8s} {step:5d}  loss={row['total']:.3e}  "
              f"mixed={row.get('pde',0):.2e}  NS={row.get('primitive',0):.2e}  "
              f"particles={row.get('particle',0):.2e}",flush=True)
    print(f'{stage}: Adam {adam_steps}, L-BFGS {lbfgs_steps}',flush=True)
    for step in range(1,adam_steps+1):
        horizon=c.T if stage=='steady' else c.T*min(1,.2+.8*step/max(adam_steps//3,1))
        d=({'t':sampler.times(c.particle_points,horizon)} if stage=='particle'
           else sampler.batch(stage=='steady',horizon,extra=extra))
        opt.zero_grad(set_to_none=True);loss,parts=objective(s,d,stage);loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('Nonfinite Adam gradient')
        opt.step();schedule.step()
        if step==1 or step%c.log_every==0 or step==adam_steps:record(step,loss,parts)
    if lbfgs_steps:
        # One fixed objective throughout each L-BFGS optimization.
        d=sampler.batch(stage=='steady',c.T,n=max(c.batch*2,1536),extra=extra)
        opt=torch.optim.LBFGS(params,max_iter=lbfgs_steps,history_size=50,
             line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
        calls=0
        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True);loss,_=objective(s,d,stage);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('Nonfinite L-BFGS gradient')
            calls+=1
            if calls%max(c.log_every,250)==0:
                print(f'{stage} L-BFGS eval {calls}: loss={float(loss.detach()):.3e}',flush=True)
            return loss
        opt.step(closure);loss,parts=objective(s,d,stage)
        used=opt.state[params[0]].get('n_iter',0);record(adam_steps+used,loss,parts)
    print(f'{stage} completed in {time.perf_counter()-started:.0f} s',flush=True)


def screen(s,steady=False,theta=None,seed_offset=0):
    c=s.c;sp=Samples(s,(c.seed+902 if steady else c.seed+903)+seed_offset,theta)
    z=sp.interior(c.screening_points,steady,c.T);rs=[]
    for batch in z.split(256):rs.append(fluid_residual(s,batch,steady,primitive=True).detach().cpu().numpy())
    rms=np.sqrt(np.mean(np.concatenate(rs)**2,axis=0));fluxerr=wallspeed=outlet=0.
    for t in ([0.] if steady else np.linspace(0,c.T,c.screening_times)):
        d=sp.batch(steady,c.T,flux_sections=65)
        if not steady:
            for key in ('wall','end','flux'):d[key][:,2]=float(t)
        with torch.no_grad():
            flow=(s.fluid.fields(d['flux'],steady)[:,0].reshape(d['fw'].shape)*d['fw']).sum(1)
            fluxerr=max(fluxerr,float((flow/(2*c.U*c.H/3)-1).abs().max()))
            wallspeed=max(wallspeed,float(torch.linalg.vector_norm(s.fluid.fields(d['wall'],steady)[:,:2],dim=1).max()/c.U))
            f,j=s.fluid.fields(d['end'],steady,jac=True)
            outlet=max(outlet,float((f[:,3]/c.U**2).square().mean().sqrt()),
                       float(((-f[:,2]+2*j[:,0,0]/c.Re)/c.U**2).square().mean().sqrt()))
    checks={'mixed_rms':bool(rms[:6].max()<.01),'primitive_rms':bool(rms[6:].max()<.05),
            'flux':fluxerr<.01,'no_slip':wallspeed<.01,'outlet':outlet<.01}
    report={'scaled_PDE_rms':rms.tolist(),
            'residual_order':['continuity','mixed_x','mixed_y','sxx_law','sxy_law',
                              'syy_law','primitive_x','primitive_y'],
            'flux_relative_error':fluxerr,
            'wall_speed_over_U':wallspeed,'outlet_scaled_rms':outlet,'checks':checks}
    if not steady:
        t=sp.trajectory(np.linspace(0,c.T,501)[:,None]);q,v,a,active,drag=s.drag_state(t)
        qinit=initial_positions(t[:,1:],c) if c.rest_lengths=='case' else None
        ac,_,g=mechanics(q,v,c,qinit)
        pr=(c.St*(a-drag-ac)/c.U).detach().square().mean(0).sqrt().cpu().numpy()
        pair=min(float(torch.linalg.vector_norm(q[:,i]-q[:,j],dim=1).min().detach()) for i,j in EDGES)
        wall_overlap=float(((c.rp-g).clamp_min(0)*active[:,:,None]/c.rp).max().detach())
        checks.update(particle_ODE=bool(pr.max()<.01),noncoincident=pair>1e-8,
                      pair_overlap=pair>1.8*c.rp,wall_overlap=wall_overlap<.1)
        report.update(particle_scaled_rms=pr.tolist(),minimum_pair_distance=pair,
                      max_wall_overlap_fraction=wall_overlap)
    report['passed']=bool(all(checks.values()))
    return report,z


def radau_check(s,theta=None):
    c=s.c;theta=nominal_theta(c) if theta is None else np.asarray(theta)
    qinit=initial_positions_np(theta,c);qrest=qinit if c.rest_lengths=='case' else np.asarray(c.q0)
    device=next(s.parameters()).device
    def carrier(q,t):
        z=np.c_[np.clip(q[:,0],0,c.L),q[:,1],np.full(4,t),np.broadcast_to(theta,(4,4))]
        with torch.no_grad():return s.fluid.fields(torch.as_tensor(z,device=device,dtype=torch.float64))[:,:2].cpu().numpy()
    def rhs(t,z):
        q,v=z[:8].reshape(4,2),z[8:].reshape(4,2);active=(q[:,0]>0)&(q[:,0]<c.L)
        a=(carrier(q,t)-v)/c.St*active[:,None]+mechanics_np(q,v,c,qrest)
        return np.r_[v.ravel(),a.ravel()]
    times=np.linspace(0,c.T,501)
    def contact_event(i,j):
        def event(t,state):
            pos=state[:8].reshape(4,2)
            if not (0 < pos[i,0] < c.L and 0 < pos[j,0] < c.L): return 1.0
            return np.linalg.norm(pos[i]-pos[j])-2*c.rp
        event.direction=-1;event.terminal=False
        return event
    sol=solve_ivp(rhs,(0,c.T),np.r_[qinit.ravel(),np.zeros(8)],method='Radau',
                  t_eval=times,rtol=1e-9,atol=1e-11,max_step=min(c.T/150,c.St/5),
                  events=[contact_event(i,j) for i,j in EDGES])
    if not sol.success:return {'success':False,'message':sol.message,'passed':False},None
    with torch.no_grad():q,v,_=s.parcels.motion(torch.as_tensor(times[:,None],device=device,dtype=torch.float64),theta)
    pe=float(np.max(np.abs(sol.y[:8].T-q.cpu().numpy().reshape(-1,8)))/c.H)
    ve=float(np.max(np.abs(sol.y[8:].T-v.cpu().numpy().reshape(-1,8)))/c.U)
    return {'success':True,'position_error_over_H':pe,'velocity_error_over_U':ve,
            'passed':pe<.01 and ve<.03,'contact_onset_times':{f'{i+1}-{j+1}':ts.tolist() for (i,j),ts in zip(EDGES,sol.t_events)},
            'scope':'particle equations in the SAME learned carrier; not independent CFD' },sol.y.T


def save(s,history,out,report=None):
    torch.save({'format':'stenosis_four_parametric_ic_v1','config':asdict(s.c),
        'state_dict':s.state_dict(),'history':history,
        'smoke_test':bool(getattr(s,'smoke_test',False))},out/'model.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    if report is not None:(out/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def render(s,out,report,reference,gif=True,theta=None):
    c=s.c;theta=nominal_theta(c) if theta is None else np.asarray(theta)
    qinit=initial_positions_np(theta,c);qrest=qinit if c.rest_lengths=='case' else np.asarray(c.q0)
    device=next(s.parameters()).device
    tt=np.linspace(0,c.T,501)
    with torch.no_grad():q,v,a=s.parcels.motion(torch.tensor(tt[:,None],device=device,dtype=torch.float64),theta)
    q,v,a=[z.cpu().numpy() for z in (q,v,a)]
    np.savetxt(out/'trajectory.csv',np.c_[tt,q.reshape(-1,8),v.reshape(-1,8),a.reshape(-1,8)],
        delimiter=',',comments='',header='t,'+','.join(f'{kind}{i+1}{axis}' for kind in ('q','v','a') for i in range(4) for axis in ('x','y')))
    if reference is not None:np.savetxt(out/'Radau_reference.csv',np.c_[tt,reference],delimiter=',')
    xs,ys=np.linspace(0,c.L,181),np.linspace(0,c.H,73);xx,yy=np.meshgrid(xs,ys)
    lo,hi=bounds_np(xs,c);mask=(yy>lo)&(yy<hi)
    def field(t):
        z=torch.as_tensor(np.c_[xx[mask],yy[mask],np.full(mask.sum(),t),np.broadcast_to(theta,(mask.sum(),4))],device=device,dtype=torch.float64)
        with torch.no_grad():values=torch.cat([s.fluid.fields(a) for a in z.split(4096)]).cpu().numpy()
        arr=np.full((*xx.shape,6),np.nan);arr[mask]=values;return arr
    times=np.array([0.,c.T/2,c.T]);fields=np.stack([field(t) for t in times])
    np.savez_compressed(out/'fields.npz',x=xs,y=ys,t=times,fields=fields,mask=mask)
    colors=['#fb7185','#fbbf24','#38bdf8','#c084fc'];vmax=float(np.nanmax(np.linalg.norm(fields[:,:,:,:2],axis=-1)))
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    status='CHECKS PASSED' if report['passed'] else 'UNCONVERGED - DIAGNOSTIC ONLY'
    def backdrop(ax):
        ax.fill_between(xs,0,lo,color='#334155');ax.fill_between(xs,hi,c.H,color='#334155')
        ax.plot(xs,lo,color='#0f172a',lw=1.2);ax.plot(xs,hi,color='#0f172a',lw=1.2)
        ax.set(xlim=(0,c.L),ylim=(0,c.H),xlabel='x',ylabel='y',aspect='equal')
    def beads(ax,pos):
        lines=[ax.plot(pos[[i,j],0],pos[[i,j],1],color='white',lw=1.3,alpha=.9)[0] for i,j in EDGES]
        circles=[plt.Circle(tuple(pos[i]),c.rp,facecolor=colors[i],edgecolor='white',lw=1.2,zorder=4) for i in range(4)]
        for patch in circles:ax.add_patch(patch)
        return lines,circles
    fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for k,ax in enumerate(axes[0]):
        arr=fields[k];speed=np.linalg.norm(arr[:,:,:2],axis=-1)
        im=ax.pcolormesh(xx,yy,speed,cmap='viridis',shading='auto',vmin=0,vmax=vmax);backdrop(ax)
        ax.streamplot(xs,ys,np.ma.masked_invalid(arr[:,:,0]),np.ma.masked_invalid(arr[:,:,1]),
                      density=.65,color='white',linewidth=.5,arrowsize=.65)
        index=np.argmin(np.abs(tt-times[k]));beads(ax,q[index])
        for i in range(4):ax.plot(q[:index+1,i,0],q[:index+1,i,1],color=colors[i],lw=1.3)
        ax.set_title(f't = {times[k]:.2f}')
    fig.colorbar(im,ax=list(axes[0]),label='Speed')
    for i in range(4):
        axes[1,0].plot(tt,q[:,i,0],color=colors[i],label=f'Parcel {i+1}')
        if reference is not None:axes[1,0].plot(tt,reference[:,2*i],'--',color=colors[i],alpha=.6)
    axes[1,0].set(xlabel='t',ylabel='x position',title='PINN solid / Radau dashed');axes[1,0].legend(fontsize=8)
    strain=[]
    for i,j in EDGES:
        l0=np.linalg.norm(qrest[i]-qrest[j]);e=(np.linalg.norm(q[:,i]-q[:,j],axis=-1)-l0)/l0
        strain.append(e);axes[1,1].plot(tt,100*e,label=f'{i+1}-{j+1}')
    axes[1,1].set(xlabel='t',ylabel='Bond strain (%)',title='Six spring-dashpot links');axes[1,1].legend(ncol=3,fontsize=8)
    np.savetxt(out/'bond_strain.csv',np.c_[tt,np.array(strain).T],delimiter=',',comments='',header='t,'+','.join(f'{i+1}-{j+1}' for i,j in EDGES))
    history=json.loads((out/'history.json').read_text())
    for key in ('total','pde','primitive','particle','flux'):
        axes[1,2].semilogy([max(r.get(key,np.nan),1e-16) for r in history],label=key)
    axes[1,2].set(xlabel='Logged training evaluations',ylabel='Loss',title='Convergence history');axes[1,2].legend(fontsize=8)
    fig.suptitle(f'Four bonded parcels through a stenosis | Re={c.Re:g} | {status}',fontsize=14)
    fig.savefig(out/'overview.png',dpi=180);fig.savefig(out/'overview.pdf');plt.close(fig)
    if gif:
        fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
        im=ax.pcolormesh(xx,yy,np.linalg.norm(fields[0,:,:,:2],axis=-1),cmap='viridis',shading='auto',vmin=0,vmax=vmax)
        backdrop(ax);fig.colorbar(im,ax=ax,label='Speed');lines,circles=beads(ax,q[0])
        trails=[ax.plot([],[],color=color,lw=1.5)[0] for color in colors]
        title=ax.set_title(status)
        def update(k):
            t=c.T*k/59;index=np.argmin(np.abs(tt-t));pos=q[index]
            im.set_array(np.linalg.norm(field(t)[:,:,:2],axis=-1).ravel())
            for line,(i,j) in zip(lines,EDGES):line.set_data(pos[[i,j],0],pos[[i,j],1])
            for i,patch in enumerate(circles):patch.center=tuple(pos[i]);trails[i].set_data(q[:index+1,i,0],q[:index+1,i,1])
            title.set_text(f'{status} | t={t:.2f}');return [im,title,*lines,*circles,*trails]
        FuncAnimation(fig,update,frames=60,interval=70,blit=False).save(out/'motion.gif',writer=PillowWriter(fps=15));plt.close(fig)


def self_test():
    c=Config();c.validate();q=np.asarray(c.q0);v=np.zeros((4,2))
    assert np.max(np.abs(mechanics_np(q,v,c)))==0
    qm=q.copy();qm[1]+=[.006,.004];vm=np.array([[.1,.2],[.2,-.1],[-.2,.1],[0,-.1]])
    a=mechanics_np(qm,vm,c);assert np.max(np.abs(a.sum(0)))<1e-10
    assert abs(np.sum(qm[:,0]*a[:,1]-qm[:,1]*a[:,0]))<1e-10
    lo,hi=bounds_np(c.xc,c);assert abs(hi-lo-(c.H-2*c.R))<1e-12
    # Quadrature resolves the original Gaussian feedback inside this fixed domain.
    nodes,w=np.polynomial.legendre.leggauss(48);edges=[0,c.xc-c.R,c.xc,c.xc+c.R,c.L]
    for xp,yp in ((.4,.14),(.5,.26),(.99,.2),(.31,.04)):
        def fn(x):
            lo,hi=bounds_np(x,c)
            return np.exp(-.5*((x-xp)/c.eps)**2)/(math.sqrt(2*math.pi)*c.eps)*(ndtr((hi-yp)/c.eps)-ndtr((lo-yp)/c.eps))
        ref=sum(quad(fn,a,b,epsabs=1e-11)[0] for a,b in zip(edges[:-1],edges[1:]))
        val=sum(np.sum(fn((a+b)/2+(b-a)/2*nodes)*w)*(b-a)/2 for a,b in zip(edges[:-1],edges[1:]))
        assert abs(val/ref-1)<1e-3,(xp,yp,val,ref)
    print('PASS: fixed geometry, four-bead force/torque balance, equilibrium and Gaussian normalization')
    if torch is None:
        print('SKIPPED: neural derivative/gradient checks (PyTorch is not installed)');return
    torch.manual_seed(c.seed);torch.set_num_threads(2)
    s=System(c).double()
    with torch.no_grad():
        s.fluid.delta.layers[-1].weight.normal_(0,.02)
        s.fluid.delta.layers[-1].bias.normal_(0,.01)
    z=torch.tensor([[.2,.15,.04],[.4,.2,.12],[.7,.26,.5]],dtype=torch.float64,requires_grad=True)
    for steady in (True,False):
        zz=z[:,:2] if steady else z
        f,j=s.fluid.fields(zz,steady,True)
        for i in range(6):
            expected=torch.autograd.grad(f[:,i].sum(),zz,retain_graph=True,create_graph=True)[0]
            assert (expected-j[:,i]).abs().max()<1e-10
    t=z[:,2:3];q,v,a=s.parcels.motion(t)
    for i in range(4):
        for k in range(2):
            vv=torch.autograd.grad(q[:,i,k].sum(),t,retain_graph=True,create_graph=True)[0]
            aa=torch.autograd.grad(vv.sum(),t,retain_graph=True)[0]
            assert (vv[:,0]-v[:,i,k]).abs().max()<1e-10
            assert (aa[:,0]-a[:,i,k]).abs().max()<1e-10
    ac,_,_=mechanics(q,v,c)
    for i in range(len(q)):assert np.max(np.abs(ac[i].detach().numpy()-mechanics_np(q[i].detach().numpy(),v[i].detach().numpy(),c)))<1e-9
    with torch.no_grad():
        for l in s.fluid.base.layers:l.weight.zero_();l.bias.zero_()
    assert fluid_residual(s,z,True,True).abs().max()<1e-10
    t0=torch.zeros((2,1),dtype=torch.float64);q,v,_=s.parcels.motion(t0)
    assert torch.equal(q,s.parcels.q0.expand_as(q)) and v.abs().max()==0
    sp=Samples(s,c.seed);loss,_=objective(s,sp.batch(False,c.T,n=12),'coupled');loss.backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in s.parcels.net.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in s.parameters())
    print('PASS: analytic network derivatives vs autograd, NumPy/Torch mechanics, Poiseuille, exact IC and finite coupled gradients')
    # Regression: the primitive objective must contribute actual finite gradients.
    s=System(replace(c,batch=16,pde_chunk=8,boundary=12,particle_points=8)).double()
    sp=Samples(s,c.seed+44);d=sp.batch(True,c.T,n=16)
    _,parts=objective(s,d,'steady')
    assert 'primitive' in parts and parts['primitive'].requires_grad
    s.zero_grad(set_to_none=True);parts['primitive'].backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in s.fluid.base.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in s.parameters())
    print('PASS: primitive NS is in the training loss and produces nonzero finite gradients')
    # Independent reference: use ordinary autograd on FORWARD outputs, not the
    # manually propagated Jacobian used in the production training code.
    for steady in (True,False):
        zz=torch.tensor([[.2,.15,.04],[.4,.2,.12],[.7,.26,.5]],dtype=torch.float64)
        if steady:zz=zz[:,:2]
        zz=zz.detach().requires_grad_(True)
        f=s.fluid.fields(zz,steady);u,v,p=f[:,0],f[:,1],f[:,2]
        g=lambda a:torch.autograd.grad(a,zz,torch.ones_like(a),create_graph=True)[0]
        du,dv,dp=g(u),g(v),g(p)
        lapu=g(du[:,0])[:,0]+g(du[:,1])[:,1]
        lapv=g(dv[:,0])[:,0]+g(dv[:,1])[:,1]
        ut,vt=(u*0,v*0) if steady else (du[:,2],dv[:,2])
        src=zz[:,:2]*0 if steady else s.source(zz)
        reference=torch.stack(((ut+u*du[:,0]+v*du[:,1]+dp[:,0]-lapu/c.Re-src[:,0])*c.H/c.U**2,
                               (vt+u*dv[:,0]+v*dv[:,1]+dp[:,1]-lapv/c.Re-src[:,1])*c.H/c.U**2),1)
        actual=fluid_residual(s,zz,steady,primitive=True)[:,6:8]
        assert (actual-reference).abs().max()<1e-10
    print('PASS: primitive residual agrees with independent forward/autograd NS in steady and transient modes')
    # Explains the reported failure: primitive-minus-mixed involves DERIVATIVES
    # of constitutive/continuity errors, which the old value-only loss did not bound.
    z=torch.tensor([[.2,.15],[.4,.2],[.7,.26]],dtype=torch.float64,requires_grad=True)
    f,j=s.fluid.fields(z,True,jac=True)
    g=lambda a:torch.autograd.grad(a,z,torch.ones_like(a),create_graph=True)[0]
    div=j[:,0,0]+j[:,1,1]
    cxx=f[:,3]+f[:,2]-2*j[:,0,0]/c.Re
    cxy=f[:,4]-(j[:,0,1]+j[:,1,0])/c.Re
    cyy=f[:,5]+f[:,2]-2*j[:,1,1]/c.Re
    gd,gxx,gxy,gyy=g(div),g(cxx),g(cxy),g(cyy)
    expected=torch.stack((gd[:,0]/c.Re+gxx[:,0]+gxy[:,1],
                          gd[:,1]/c.Re+gxy[:,0]+gyy[:,1]),1)*c.H/c.U**2
    r=fluid_residual(s,z,True,primitive=True)
    assert ((r[:,6:8]-r[:,1:3])-expected).abs().max()<1e-10
    print('PASS: exact mixed/primitive discrepancy identity including continuity and stress-error gradients')


def print_screen(report,label):
    rms=report['scaled_PDE_rms'];failed=[k for k,v in report['checks'].items() if not v]
    print(f"{label}: mixed_RMS={max(rms[:6]):.3e}; "
          f"primitive_x={rms[6]:.3e}, primitive_y={rms[7]:.3e}; "
          f"flux={100*report['flux_relative_error']:.3f}%; "
          f"wall={100*report['wall_speed_over_U']:.3f}%",flush=True)
    print('Failed checks: '+(', '.join(failed) if failed else 'none'),flush=True)


def carrier_score(report):
    r=report['scaled_PDE_rms']
    return max(max(r[:6])/.01,max(r[6:8])/.05,
        report['flux_relative_error']/.01,report['wall_speed_over_U']/.01,
        report['outlet_scaled_rms']/.01)


def residual_score(s,z,steady):
    # Score ALL trained equations, including the previously omitted primitive NS.
    # Chunking avoids a large Hessian graph; candidates are not held-out points.
    score=[]
    for block in z.split(192):
        with torch.enable_grad():r=fluid_residual(s,block,steady,primitive=True)
        weights=r.new_tensor([s.c.continuity_weight]+[1]*5+[s.c.primitive_weight]*2)
        score.append((r.square()*weights).sum(1).detach())
    return torch.cat(score)


def resume_config(path,c):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format') not in ('stenosis_four_parametric_ic_v1',):
        raise ValueError('Resume requires a parametric IC checkpoint; old time-only models need retraining')
    saved=payload['config']
    # Preserve the trained physical case and NN shapes, not its 120000-step budget.
    keys=('L','H','xc','R','Re','U','T','St','beta','rp','eps','q0','bond_kn','bond_cn',
          'pair_kn','pair_cn','wall_kn','wall_cn','width','depth','seed',
          'variable_beads','ic_bounds','initial_clearance','rest_lengths')
    c=replace(c,**{k:saved[k] for k in keys if k in saved})
    return c,payload



def validation_conditions(c):
    """Fixed stress cases plus independent, reproducible held-out conditions."""
    rng=np.random.default_rng(c.seed+19031)
    box=np.asarray(c.ic_bounds)
    proposed=[nominal_theta(c),np.array([.13,.14,.163,.14]),
              np.array([.10,.09,.24,.20]),np.array([.17,.19,.16,.10])]
    cases=[]
    for a in proposed:
        if bool(valid_initial(a,c)) and not any(np.array_equal(a,b) for b in cases): cases.append(a)
        if len(cases)==c.validation_cases:return np.asarray(cases)
    if len(cases)<c.validation_cases:
        cases.extend(sample_initial(rng,c.validation_cases-len(cases),c))
    return np.asarray(cases)


def contact_diagnostics(s,theta,out):
    c=s.c;device=next(s.parameters()).device
    ts=np.linspace(0,c.T,2001)
    with torch.no_grad():
        q,v,_=s.parcels.motion(torch.as_tensor(ts[:,None],device=device,dtype=torch.float64),theta)
    q,v=q.cpu().numpy(),v.cpu().numpy();cols=[ts];headers=['t'];pairs={}
    for i,j in EDGES:
        dr=q[:,i]-q[:,j];dist=np.linalg.norm(dr,axis=1)
        n=dr/np.maximum(dist[:,None],1e-12);vn=((v[:,i]-v[:,j])*n).sum(-1)
        active=(q[:,i,0]>0)&(q[:,i,0]<c.L)&(q[:,j,0]>0)&(q[:,j,0]<c.L)
        overlap=np.maximum(2*c.rp-dist,0)*active
        share=.5*np.maximum(c.pair_kn*(2*c.rp-dist)-c.pair_cn*vn,0)*(dist<2*c.rp)*active
        mask=(overlap>0);onsets=np.flatnonzero(mask&~np.r_[False,mask[:-1]])
        prefix=f'p{i+1}_p{j+1}'
        headers += [prefix+'_distance',prefix+'_overlap',prefix+'_relative_normal_velocity',prefix+'_contact_acceleration_share']
        cols += [dist,overlap,vn,share]
        pairs[f'{i+1}-{j+1}']={'minimum_center_distance':float(dist.min()),
            'maximum_overlap_over_diameter':float(overlap.max()/(2*c.rp)),
            'contact_observed_on_time_grid':bool(mask.any()),
            'sampled_contact_onsets':ts[onsets].tolist(),
            'maximum_contact_acceleration_share':float(share.max())}
    np.savetxt(out/'contacts.csv',np.column_stack(cols),delimiter=',',comments='',header=','.join(headers))
    result={'time_samples':len(ts),'pairs':pairs,
            'scope':'PINN samples; Radau onset events are reported separately. Soft contacts permit finite overlap.'}
    (out/'contacts.json').write_text(json.dumps(result,indent=2))
    return result


def plot_contact_diagnostics(s,out,reference,passed):
    c=s.c;data=np.loadtxt(out/'contacts.csv',delimiter=',',skiprows=1)
    ts=data[:,0];cols=data.T
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,constrained_layout=True)
    for k,(i,j) in enumerate(EDGES):
        color=plt.get_cmap('tab10')(k)
        axes[0].plot(ts,cols[1+4*k]/(2*c.rp),color=color,label=f'{i+1}-{j+1}')
        axes[1].plot(ts,cols[2+4*k]/(2*c.rp),color=color)
        if reference is not None:
            rq=reference[:,:8].reshape(-1,4,2)
            rd=np.linalg.norm(rq[:,i]-rq[:,j],axis=-1)
            rt=np.linspace(0,c.T,len(rq))
            axes[0].plot(rt,rd/(2*c.rp),'--',color=color,alpha=.7)
    axes[0].axhline(1,color='black',ls=':',label='Contact threshold')
    axes[0].set(ylabel='Center distance / diameter',title='PINN solid / Radau dashed')
    axes[0].legend(ncol=4,fontsize=8)
    axes[1].set(xlabel='t',ylabel='Overlap / diameter (PINN)')
    fig.suptitle('CONTACT DIAGNOSTICS | '+('CHECKS PASSED' if passed else 'UNCONVERGED - DIAGNOSTIC ONLY'))
    fig.savefig(out/'contact_profiles.png',dpi=200);fig.savefig(out/'contact_profiles.pdf');plt.close(fig)


def validate_cases(s,conditions,out,history,steady_report,smoke=False,gif=True):
    """Screen every case separately; never hide a bad case in a pooled average."""
    out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for k,theta in enumerate(conditions):
        require_valid_initial(theta,s.c)
        folder=out/f'case_{k:03d}';folder.mkdir(parents=True,exist_ok=True)
        print(f'Validation case {k+1}/{len(conditions)} | initial inputs={theta.tolist()}',flush=True)
        report,_=screen(s,theta=theta,seed_offset=1009*k)
        print_screen(report,f'Case {k}')
        refreport,reference=radau_check(s,theta)
        print(f'Radau: success={refreport["success"]}; passed={refreport["passed"]}',flush=True)
        report['Radau']=refreport;report['steady']=steady_report
        report['passed']=bool(report['passed'] and steady_report['passed'] and refreport['passed'])
        report['smoke_test']=smoke
        report['initial_inputs']=np.asarray(theta).tolist()
        report['initial_positions']=initial_positions_np(theta,s.c).tolist()
        report['rest_length_policy']=s.c.rest_lengths
        restq=initial_positions_np(theta,s.c) if s.c.rest_lengths=='case' else np.asarray(s.c.q0)
        report['bond_rest_lengths']={f'{i+1}-{j+1}':float(np.linalg.norm(restq[i]-restq[j])) for i,j in EDGES}
        report['external_CFD_validated']=False;report['bond_material_calibrated']=False
        # Calculate contact statistics first, then label the final plots after all checks.
        contacts=contact_diagnostics(s,theta,folder)
        report['contacts']=contacts
        if refreport['success']:
            agreement={}
            for pair,stats in contacts['pairs'].items():
                starts=refreport['contact_onset_times'][pair]
                agreement[pair]=stats['contact_observed_on_time_grid']==bool(starts)
                predicted=stats['sampled_contact_onsets']
                stats['first_onset_error']=abs(predicted[0]-starts[0]) if predicted and starts else None
            report['contact_classification_agreement']=agreement
            report['checks']['contact_classification']=bool(all(agreement.values()))
            report['passed']=bool(report['passed'] and all(agreement.values()))
        report['numerical_screening_passed']=report['passed']
        report['eligible_for_numerical_use']=bool(report['passed'] and not smoke)
        if smoke:report['passed']=False  # execution smoke cannot certify a learned solution
        plot_contact_diagnostics(s,folder,reference,report['passed'])
        (folder/'contacts.json').write_text(json.dumps(contacts,indent=2,allow_nan=False))
        (folder/'history.json').write_text(json.dumps(history,indent=2))
        (folder/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        render(s,folder,report,reference,gif=gif,theta=theta)
        reports.append(report)
    summary={'passed':bool(all(r['passed'] for r in reports)),
        'smoke_test':smoke,'number_of_cases':len(reports),
        'conditions':np.asarray(conditions).tolist(),'cases':reports,
        'scope':'Finite held-out initial conditions, not a certificate over the entire four-dimensional domain',
        'external_CFD_validated':False,'bond_material_calibrated':False}
    (out/'validation.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


def parametric_self_test():
    self_test()
    if torch is None: raise RuntimeError('PyTorch is required for parametric validation')
    c=replace(Config(),width=12,depth=2,batch=16,boundary=8,particle_points=8,pde_chunk=8)
    c.validate();torch.manual_seed(c.seed);s=System(c).double();sp=Samples(s,c.seed)
    theta=sp.tensor(sample_initial(np.random.default_rng(831),24,c))
    t=torch.zeros((24,1),dtype=torch.float64)
    q,v,_=s.parcels.motion(t,theta)
    qinit=initial_positions(theta,c)
    assert torch.equal(q,qinit) and v.abs().max()==0
    # Six bonds have zero initial extension for every condition in the default policy.
    acc,_,_=mechanics(q,v,c,qinit);assert acc.abs().max()<1e-10
    for k in range(len(q)):
        assert np.max(np.abs(acc[k].detach().numpy()-mechanics_np(q[k].detach().numpy(),v[k].detach().numpy(),c,qinit[k].numpy())))<1e-9
    # Fixed nominal rest lengths are a distinct, correctly implemented prestressed case.
    acc_nominal,_,_=mechanics(q,v,c)
    assert acc_nominal.abs().max()>0
    for k in range(len(q)):
        assert np.max(np.abs(acc_nominal[k].detach().numpy()-mechanics_np(q[k].detach().numpy(),v[k].detach().numpy(),c)))<1e-9
    # Rejection sampling respects all pair, wall, end-plane and box constraints.
    many=sample_initial(np.random.default_rng(41),4096,c)
    assert valid_initial(many,c).all()
    invalid=nominal_theta(c);invalid[2:]=invalid[:2]
    assert not valid_initial(invalid,c)
    try:require_valid_initial(invalid,c)
    except ValueError:pass
    else:raise AssertionError('overlapping initial condition accepted')
    # Nonzero conditional fluid corrections: full seven-dimensional Jacobian
    # versus autograd, including the four condition columns.
    with torch.no_grad():
        s.fluid.delta.layers[-1].weight.normal_(0,.02)
        s.fluid.delta.layers[-1].bias.normal_(0,.01)
    z=torch.cat((sp.tensor([[.2,.15,.04],[.4,.20,.12],[.7,.26,.5]]),theta[:3]),1).requires_grad_(True)
    f,j=s.fluid.fields(z,jac=True)
    for i in range(6):
        expected=torch.autograd.grad(f[:,i].sum(),z,retain_graph=True,create_graph=True)[0]
        assert (expected-j[:,i]).abs().max()<1e-10
    inp=torch.cat((z[:,2:3],z[:,3:]),1).detach().requires_grad_(True)
    q,v,a=s.parcels.motion(inp)
    for i in range(4):
        for axis in range(2):
            vv=torch.autograd.grad(q[:,i,axis].sum(),inp,retain_graph=True,create_graph=True)[0][:,0]
            aa=torch.autograd.grad(vv.sum(),inp,retain_graph=True)[0][:,0]
            assert (vv-v[:,i,axis]).abs().max()<1e-10
            assert (aa-a[:,i,axis]).abs().max()<1e-10
    # Confirm parameters affect both forward outputs, not just their array sizes.
    t=sp.tensor([[.12],[.12]])
    conditions=sp.tensor([nominal_theta(c),[.13,.14,.163,.14]])
    q=s.parcels.motion(t,conditions)[0];assert not torch.allclose(q[0],q[1])
    zz=torch.cat((sp.tensor([[.4,.2,.12],[.4,.2,.12]]),conditions),1)
    ff=s.fluid.fields(zz);assert not torch.allclose(ff[0],ff[1],atol=1e-12,rtol=0)
    # A fluid quadrature section is one physical case, not 40 different ICs.
    d=sp.batch(False,c.T,n=16)
    groups=d['flux'][:,3:].reshape(-1,40,4)
    assert torch.equal(groups,groups[:,:1].expand_as(groups))
    s.zero_grad(set_to_none=True);loss,_=objective(s,d,'coupled');loss.backward()
    for module in (s.parcels.net,s.fluid.delta):
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in module.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    # Conditional primitive NS agrees with ordinary autograd on physical columns;
    # parameter derivatives must NOT enter spatial divergence or the Laplacian.
    zz=z.detach().requires_grad_(True);f=s.fluid.fields(zz);u,v,p=f[:,0],f[:,1],f[:,2]
    grad=lambda a:torch.autograd.grad(a,zz,torch.ones_like(a),create_graph=True)[0]
    du,dv,dp=grad(u),grad(v),grad(p)
    lapu=grad(du[:,0])[:,0]+grad(du[:,1])[:,1]
    lapv=grad(dv[:,0])[:,0]+grad(dv[:,1])[:,1]
    src=s.source(zz)
    ref=torch.stack(((du[:,2]+u*du[:,0]+v*du[:,1]+dp[:,0]-lapu/c.Re-src[:,0])*c.H/c.U**2,
                     (dv[:,2]+u*dv[:,0]+v*dv[:,1]+dp[:,1]-lapv/c.Re-src[:,1])*c.H/c.U**2),1)
    assert (fluid_residual(s,zz,primitive=True)[:,6:]-ref).abs().max()<1e-10
    # Actual active-contact mechanics, with approaching and separating velocities.
    cq=np.asarray(c.q0).copy();cq[1]=cq[0]+[.028,0]
    cv=np.zeros((4,2));cv[0,0]=.1;cv[1,0]=-.1
    for sign in (1,-1):
        qt=sp.tensor(cq[None]);vt=sp.tensor((sign*cv)[None])
        at,_,_=mechanics(qt,vt,c,qt)
        ref=mechanics_np(cq,sign*cv,c,cq)
        assert np.max(np.abs(at[0].detach().numpy()-ref))<1e-9
        assert at.sum(1).abs().max()<1e-9
    # Independent damped contact relaxation is integrated (not just algebra tested).
    cc=replace(c,bond_kn=0,bond_cn=0)
    def rhs(t,state):
        qq=state[:8].reshape(4,2);vv=state[8:].reshape(4,2)
        return np.r_[vv.ravel(),mechanics_np(qq,vv,cc,cq).ravel()]
    sol=solve_ivp(rhs,(0,.025),np.r_[cq.ravel(),cv.ravel()],method='Radau',rtol=1e-9,atol=1e-11,max_step=.0002)
    assert sol.success and np.isfinite(sol.y).all()
    distances=np.linalg.norm(sol.y[2:4]-sol.y[0:2],axis=0)
    assert distances[-1]>2*c.rp and distances.min()<2*c.rp
    assert np.max(np.abs(sol.y[8:].reshape(4,2,-1).sum(0)))<1e-9
    energy=.5*np.sum(sol.y[8:]**2,axis=0)+.25*c.pair_kn*np.maximum(2*c.rp-distances,0)**2
    assert np.max(np.diff(energy))<1e-7 and energy[-1]<energy[0]
    print('PASS: conditional ICs, admissible sampling, seven-input fluid derivatives, five-input time derivatives, conditional NS, quadrature consistency and gradients')
    print('PASS: approaching/separating contact mechanics and independent Radau contact relaxation')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--quick-test',action='store_true',help='Execution smoke only; never a converged solution')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--outdir',default=Config.outdir);p.add_argument('--no-gif',action='store_true')
    p.add_argument('--resume',help='Continue a parametric-IC model.pt')
    p.add_argument('--evaluate',help='Evaluate a parametric model.pt without training')
    p.add_argument('--carrier-from',help='Transfer the steady carrier from a compatible original FOUR-particle checkpoint')
    p.add_argument('--config',help='JSON Config overrides; saved physics/architecture take precedence on resume/evaluate')
    p.add_argument('--initial',type=float,nargs=4,metavar=('XA0','YA0','XB0','YB0'))
    p.add_argument('--cases',help='JSON list of four-coordinate conditions for evaluation')
    p.add_argument('--steady-only',action='store_true')
    args=p.parse_args(argv)
    if args.self_test:parametric_self_test();return
    if torch is None:raise SystemExit('PyTorch missing; install dependencies from requirements.txt')
    if sum(bool(a) for a in (args.resume,args.evaluate,args.carrier_from))>1:
        p.error('Use only one of --resume, --evaluate, --carrier-from')
    if args.initial and args.cases:p.error('Use --initial OR --cases')
    if (args.initial or args.cases) and not args.evaluate:p.error('--initial/--cases require --evaluate')
    if args.quick_test and (args.resume or args.evaluate or args.carrier_from or args.steady_only):
        p.error('--quick-test requires a fresh smoke model')
    c=Config(device=args.device,outdir=args.outdir)
    if args.config:
        overrides=json.loads(Path(args.config).read_text(encoding='utf-8'))
        unknown=set(overrides)-set(asdict(c))
        if unknown:raise ValueError('Unknown Config fields: '+', '.join(sorted(unknown)))
        c=replace(c,**overrides);c=replace(c,device=args.device,outdir=args.outdir)
    payload=None;checkpoint=args.resume or args.evaluate
    if checkpoint:c,payload=resume_config(checkpoint,c)
    if args.quick_test:
        c=replace(c,width=12,depth=2,batch=32,boundary=16,particle_points=16,pde_chunk=16,
            steady_adam=8,steady_lbfgs=2,coupled_adam=8,coupled_lbfgs=2,
            refinement_rounds=0,log_every=100,validation_cases=2,screening_points=128,screening_times=3)
    c.validate();torch.set_num_threads(2);torch.manual_seed(c.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if c.device=='auto' else c.device
    s=System(c).to(device=device,dtype=torch.float64);sp=Samples(s,c.seed)
    out=Path(c.outdir)
    if checkpoint and out.resolve()==Path(checkpoint).resolve().parent and args.evaluate:
        raise ValueError('Evaluation outdir must differ from checkpoint directory')
    # Preserve prior outputs; resuming into an existing run is an explicit exception.
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Output directory is nonempty; choose a new --outdir')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(c),indent=2));history=[]
    started=time.perf_counter()
    if payload:
        s.load_state_dict(payload['state_dict'],strict=True);history=list(payload.get('history',[]))
        print(f'Loaded parametric checkpoint: {checkpoint}',flush=True)
    s.smoke_test=bool(args.quick_test or (payload and payload.get('smoke_test',False)))
    if args.carrier_from:
        old=torch.load(args.carrier_from,map_location='cpu',weights_only=True)
        if old.get('format') not in ('stenosis_four_fixed_v1','stenosis_four_corrected_v2'):
            raise ValueError('Carrier transfer needs an original FOUR-particle checkpoint')
        for k in ('L','H','xc','R','Re','U','width','depth'):
            if old['config'][k]!=getattr(c,k):raise ValueError(f'Carrier transfer mismatch: {k}')
        s.fluid.base.load_state_dict({k.removeprefix('fluid.base.'):v for k,v in old['state_dict'].items() if k.startswith('fluid.base.')},strict=True)
    ids=[i+1 for i in c.variable_beads]
    print(f'Parametric IC | fluid=7 | trajectory=5 | variable particles={ids} | rest lengths={c.rest_lengths} | {device}',flush=True)
    if args.evaluate:
        conditions=np.asarray([args.initial] if args.initial else
            json.loads(Path(args.cases).read_text()) if args.cases else validation_conditions(c),dtype=float)
        if conditions.ndim!=2 or conditions.shape[1]!=4 or len(conditions)==0:
            raise ValueError('Evaluation conditions must be a nonempty list of four-coordinate inputs')
        require_valid_initial(conditions,c)
        base,_=screen(s,True);print_screen(base,'Steady validation')
        # Checkpoint execution mode follows its provenance, even on evaluation.
        smoke=bool(payload.get('smoke_test',False))
        report=validate_cases(s,conditions,out,history,base,smoke=smoke,gif=not args.no_gif)
        print(f'Evaluation outputs: {out.resolve()} | passed={report["passed"]}',flush=True)
        if not report['passed'] and not smoke:raise RuntimeError('Evaluation failed numerical screening; see validation.json')
        return
    try:
        if payload or args.carrier_from:
            base,_=screen(s,True);print_screen(base,'Loaded carrier')
            if not base['passed']:train(s,sp,'steady',c.repair_adam,c.repair_lbfgs,history,lr=min(c.steady_lr,1e-4))
        else:train(s,sp,'steady',c.steady_adam,c.steady_lbfgs,history)
        base,_=screen(s,True);print_screen(base,'Steady validation')
        best_base=base;best_state={k:v.detach().cpu().clone() for k,v in s.state_dict().items()}
        for k in range(c.refinement_rounds):
            if base['passed']:break
            candidate=sp.interior(4096,True,c.T);score=residual_score(s,candidate,True)
            extra=candidate[score.topk(512).indices].detach()
            train(s,sp,'steady',1200,600,history,extra,lr=min(c.steady_lr,1e-4))
            base,_=screen(s,True);print_screen(base,'Steady refinement')
            if carrier_score(base)<carrier_score(best_base):
                best_base=base;best_state={key:v.detach().cpu().clone() for key,v in s.state_dict().items()}
            else:s.load_state_dict(best_state,strict=True);base=best_base
        (out/'steady_validation.json').write_text(json.dumps(base,indent=2));save(s,history,out)
        if not base['passed'] and not args.quick_test:
            raise RuntimeError('Steady carrier not converged. Resume saved weights; see steady_validation.json')
        if args.steady_only:return
        # Exact v(0)=0 is retained from the supplied ansatz and independent ODE.
        # The old u0 buffer was only a trainable-motion offset, not initial velocity.
        for par in s.fluid.base.parameters():par.requires_grad_(False)
        train(s,sp,'particle',8 if args.quick_test else 600,0,history)
        train(s,sp,'coupled',c.coupled_adam,c.coupled_lbfgs,history)
        report,_=screen(s)  # random-IC residual screening for adaptive training only
        for k in range(c.refinement_rounds):
            if report['passed']:break
            candidate=sp.interior(4096,False,c.T);score=residual_score(s,candidate,False)
            extra=candidate[score.topk(512).indices].detach()
            train(s,sp,'coupled',1200,400,history,extra,lr=min(c.coupled_lr,1e-4))
            report,_=screen(s)
        report=validate_cases(s,validation_conditions(c),out,history,base,smoke=s.smoke_test,gif=not args.no_gif)
        report['seconds']=time.perf_counter()-started
        save(s,history,out,report)
        print(f'Outputs: {out.resolve()} | checks_passed={report["passed"]}',flush=True)
        if not report['passed'] and not args.quick_test:
            raise RuntimeError('Held-out validation failed; inspect validation.json before numerical use')
    except (KeyboardInterrupt,RuntimeError,FloatingPointError):
        save(s,history,out);raise


if __name__=='__main__':main()
