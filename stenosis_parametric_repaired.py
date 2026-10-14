"""Four bonded particles in a fixed stenosis: parametric initial positions.
Fluid transient input: (x,y,t,x_a0,y_a0,x_b0,y_b0).
Trajectory input: (t,x_a0,y_a0,x_b0,y_b0); initial velocities are zero.
Hybrid physics training with independent ODE anchors, separated fluid optimization,
and explicit ODE fallback when the trajectory surrogate fails validation.
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
from threadpoolctl import threadpool_limits
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
    trajectory_width: int = 64
    trajectory_depth: int = 4
    reference_cases: int = 32
    boundary_anchors: bool = True
    reference_workers: int = 4
    carrier_polish_adam: int = 300
    carrier_polish_lbfgs: int = 40
    reference_times: int = 201
    surrogate_adam: int = 3500
    surrogate_lbfgs: int = 300
    coupling_rounds: int = 2
    fluid_adam: int = 300
    fluid_lbfgs: int = 20
    reference_rtol: float = 1e-9
    reference_atol: float = 1e-11
    reference_max_step: float = 0.002
    export_zip: bool = True

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
        for k in ('trajectory_width','trajectory_depth','reference_cases','reference_times'):
            if not isinstance(getattr(self,k),int) or getattr(self,k)<1:
                raise ValueError(k+' must be a positive integer')
        if self.reference_times<3: raise ValueError('reference_times must be >= 3')
        for k in ('surrogate_adam','surrogate_lbfgs','coupling_rounds','fluid_adam','fluid_lbfgs',
                  'carrier_polish_adam','carrier_polish_lbfgs'):
            if not isinstance(getattr(self,k),int) or getattr(self,k)<0:
                raise ValueError(k+' must be a nonnegative integer')
        for k in ('reference_rtol','reference_atol','reference_max_step'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0:raise ValueError(k+' must be positive')
        if not isinstance(self.reference_workers,int) or self.reference_workers<1:
            raise ValueError('reference_workers must be a positive integer')
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

    def fields(self,z,steady=False,jac=False,physical_jac=False):
        c=self.c;x,y=z[:,:1],z[:,1:2];xi=x/c.L;s=2*y/c.H-1
        f=xi*(1-s.square());vf=f*(1-xi);ps=max(c.U**2,8*c.U*c.L/(c.Re*c.H**2))
        p=8*c.U*(c.L-x)/(c.Re*c.H**2)
        u=c.U*(1-s.square());shear=-4*c.U*s/(c.Re*c.H)
        value=torch.cat((u,0*u,p,-p,shear,-p),1)
        zero=x*0;one=zero+1
        factors=torch.cat((c.U*f,c.U*vf*s,ps*one,ps*one,ps*s,ps*one),1)
        features=torch.cat((2*xi-1,2*s.square()-1),1)
        if jac:
            dim=3 if physical_jac and not steady else z.shape[1]
            dvalue=z.new_zeros((len(z),6,dim))
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
                dim=3 if physical_jac else z.shape[1];jf=z.new_zeros((len(z),7,dim))
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
        super().__init__();self.c=c;self.net=MLP(5,8,c.trajectory_width,c.trajectory_depth)
        nn.init.zeros_(self.net.layers[-1].weight);nn.init.zeros_(self.net.layers[-1].bias)
        self.initial_velocity_hint=None
        self.register_buffer('q0',torch.tensor(c.q0,dtype=torch.float64))
        self.register_buffer('u0',torch.zeros(4,2,dtype=torch.float64))

    def motion(self,t,theta=None):
        if t.shape[1]==5:
            if theta is not None: raise ValueError('Provide conditions once')
            theta=t[:,1:];t=t[:,:1]
        c=self.c;theta=condition_like(t,c,theta);q0=initial_positions(theta,c)
        inputs=torch.cat((t,normalized_theta(theta,c)),1)
        raw,dr,ddr=self.net.time_derivatives(inputs,c.T)
        hint=(self.initial_velocity_hint(q0) if self.initial_velocity_hint is not None
              else self.u0[None].expand_as(q0))
        raw,dr,ddr=[a.reshape(-1,4,2) for a in (raw,dr,ddr)]
        gate=(-torch.expm1(-t/c.St))[:,:,None]
        dgate=(torch.exp(-t/c.St)/c.St)[:,:,None];ddgate=-dgate/c.St
        b=hint+c.U*gate*raw
        db=c.U*(dgate*raw+gate*dr)
        ddb=c.U*(ddgate*raw+2*dgate*dr+gate*ddr)
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
        self.parcels.initial_velocity_hint=self.initial_motion_hint
        self.reference_bank=None
        nodes,w=np.polynomial.legendre.leggauss(48)
        edges=[0,c.xc-c.R,c.xc,c.xc+c.R,c.L]
        xs=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in zip(edges[:-1],edges[1:])])
        ws=np.concatenate([(b-a)/2*w for a,b in zip(edges[:-1],edges[1:])])
        lo,hi=bounds_np(xs,c)
        for name,a in (('gx',xs),('gw',ws),('glo',lo),('ghi',hi)):
            self.register_buffer(name,torch.tensor(a,dtype=torch.float64))

    def initial_motion_hint(self,q):
        hint=self.fluid.fields(q.reshape(-1,2),True)[:,:2].reshape(-1,4,2)
        if self.c.rest_lengths=='nominal':
            acceleration,_,_=mechanics(q,torch.zeros_like(q),self.c)
            hint=hint+self.c.St*acceleration
        return hint

    def drag_state(self,t,theta=None):
        if t.shape[1]==5:
            if theta is not None: raise ValueError('Provide conditions once')
            theta=t[:,1:];t=t[:,:1]
        c=self.c;theta=condition_like(t,c,theta)
        if self.reference_bank is None:
            q,v,a=self.parcels.motion(t,theta)
        else:
            q,v,a=self.reference_bank.state(t,theta)
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


def fluid_residual(s,z,steady=False,primitive=False,derivative_graph=True):
    c=s.c
    if primitive:z=z.detach().requires_grad_(True)
    f,j=s.fluid.fields(z,steady,jac=True,physical_jac=True);u,v,p,xx,xy,yy=f.unbind(1)
    ux,uy=j[:,0,0],j[:,0,1];vx,vy=j[:,1,0],j[:,1,1]
    ut,vt=(u*0,v*0) if steady else (j[:,0,2],j[:,1,2])
    # The integrated particle source is fixed data at collocation coordinates.
    # Keep its fluid-weight gradients, without building unused coordinate graphs.
    source_points=z.detach() if s.reference_bank is not None else z
    src=z[:,:2]*0 if steady else s.source(source_points);k=c.H/c.U**2
    rr=[(ux+vy)*c.H/c.U,(ut+u*ux+v*uy-j[:,3,0]-j[:,4,1]-src[:,0])*k,
        (vt+u*vx+v*vy-j[:,4,0]-j[:,5,1]-src[:,1])*k,
        (xx+p-2*ux/c.Re)/c.U**2,(xy-(uy+vx)/c.Re)/c.U**2,(yy+p-2*vy/c.Re)/c.U**2]
    if primitive:
        g=lambda a:torch.autograd.grad(a,z,torch.ones_like(a),create_graph=derivative_graph,retain_graph=True)[0]
        lapu=g(ux)[:,0]+g(uy)[:,1];lapv=g(vx)[:,0]+g(vy)[:,1]
        rr.extend(((ut+u*ux+v*uy+j[:,2,0]-lapu/c.Re-src[:,0])*k,
                   (vt+u*vx+v*vy+j[:,2,1]-lapv/c.Re-src[:,1])*k))
    return torch.stack(rr,1)


class Samples:
    def __init__(self,s,seed,theta=None):
        self.fixed_theta=theta
        self.condition_bank=None
        self.s=s;self.c=s.c;self.rng=np.random.default_rng(seed)
        self.device=next(s.parameters()).device
    def tensor(self,a):
        if not torch.is_tensor(a): a=np.asarray(a)
        return torch.as_tensor(a,device=self.device,dtype=torch.float64)
    def trajectory(self,t):
        t=self.tensor(t) if not torch.is_tensor(t) else t
        theta=(self.condition_bank[self.rng.integers(len(self.condition_bank),size=len(t))]
               if self.condition_bank is not None and self.fixed_theta is None else
               sample_initial(self.rng,len(t),self.c) if self.fixed_theta is None else
               np.broadcast_to(self.fixed_theta,(len(t),4)))
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
            with torch.no_grad():q=self.s.drag_state(t)[0].cpu().numpy()
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
    if not steady and stage!='fluid':
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
    modules=([s.fluid.base] if stage=='steady' else [s.fluid.delta] if stage=='fluid'
             else [s.parcels.net]+([s.fluid.delta] if stage=='coupled' else []))
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
        horizon=c.T if stage in ('steady','fluid') else c.T*min(1,.2+.8*step/max(adam_steps//3,1))
        d=({'t':sampler.times(c.particle_points,horizon)} if stage=='particle'
           else sampler.batch(stage=='steady',horizon,extra=extra))
        opt.zero_grad(set_to_none=True);loss,parts=objective(s,d,stage);loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('Nonfinite Adam gradient')
        opt.step();schedule.step()
        if step==1 or step%c.log_every==0 or step==adam_steps:record(step,loss,parts)
    if lbfgs_steps:
        # One fixed objective throughout each L-BFGS optimization.
        n_lbfgs=min(max(c.batch,256),512) if stage=='fluid' else max(c.batch*2,1536)
        d=sampler.batch(stage=='steady',c.T,n=n_lbfgs,extra=extra)
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


def _screen_impl(s,steady=False,theta=None,seed_offset=0):
    c=s.c;sp=Samples(s,(c.seed+902 if steady else c.seed+903)+seed_offset,theta)
    z=sp.interior(c.screening_points,steady,c.T);rs=[]
    for batch in z.split(256):rs.append(fluid_residual(s,batch,steady,primitive=True,derivative_graph=False).detach().cpu().numpy())
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
        # Screen the SURROGATE, even when the fluid source uses numerical anchors.
        t=sp.trajectory(np.linspace(0,c.T,501)[:,None])
        bank=s.reference_bank;s.reference_bank=None
        try:q,v,a,active,drag=s.drag_state(t)
        finally:s.reference_bank=bank
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


def screen(s,steady=False,theta=None,seed_offset=0):
    flags=[p.requires_grad for p in s.parameters()]
    try:
        for p in s.parameters():p.requires_grad_(False)
        return _screen_impl(s,steady,theta,seed_offset)
    finally:
        for p,flag in zip(s.parameters(),flags):p.requires_grad_(flag)


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
    torch.save({'format':'stenosis_four_parametric_repaired_v2','config':asdict(s.c),
        'state_dict':s.state_dict(),'history':history,
        'training_conditions':getattr(s,'training_conditions',[]),
        'trajectory_ansatz':'exact_initial_acceleration_v1',
        'smoke_test':bool(getattr(s,'smoke_test',False))},out/'model.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    if report is not None:(out/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def render(s,out,report,reference,gif=True,theta=None,output_state=None):
    c=s.c;theta=nominal_theta(c) if theta is None else np.asarray(theta)
    qinit=initial_positions_np(theta,c);qrest=qinit if c.rest_lengths=='case' else np.asarray(c.q0)
    device=next(s.parameters()).device
    tt=np.linspace(0,c.T,501)
    with torch.no_grad():q,v,a=s.parcels.motion(torch.tensor(tt[:,None],device=device,dtype=torch.float64),theta)
    q,v,a=[z.cpu().numpy() for z in (q,v,a)]
    q_surrogate=q.copy()
    np.savetxt(out/'PINN_trajectory.csv',np.c_[tt,q.reshape(-1,8),v.reshape(-1,8),a.reshape(-1,8)],
        delimiter=',',comments='',header='t,'+','.join(f'{kind}{i+1}{axis}' for kind in ('q','v','a') for i in range(4) for axis in ('x','y')))
    if output_state is not None:
        q,v,a=output_state
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
    status=('VALIDATED NUMERICAL TRAJECTORY' if report.get('output_validated',False)
            else 'CHECKS PASSED' if report['passed'] else 'UNCONVERGED - DIAGNOSTIC ONLY')
    status+= ' | '+report.get('output_method','PINN')
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
        axes[1,0].plot(tt,q_surrogate[:,i,0],'--',color=colors[i],alpha=.6)
    axes[1,0].set(xlabel='t',ylabel='x position',title='Selected output solid / PINN dashed');axes[1,0].legend(fontsize=8)
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
        with torch.enable_grad():r=fluid_residual(s,block,steady,primitive=True,derivative_graph=False)
        weights=r.new_tensor([s.c.continuity_weight]+[1]*5+[s.c.primitive_weight]*2)
        score.append((r.square()*weights).sum(1).detach())
    return torch.cat(score)


def resume_config(path,c):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format') not in ('stenosis_four_parametric_repaired_v2',):
        raise ValueError('Resume requires a parametric IC checkpoint; old time-only models need retraining')
    saved=payload['config']
    # Preserve the trained physical case and NN shapes, not its 120000-step budget.
    keys=('L','H','xc','R','Re','U','T','St','beta','rp','eps','q0','bond_kn','bond_cn',
          'pair_kn','pair_cn','wall_kn','wall_cn','width','depth','seed',
          'variable_beads','ic_bounds','initial_clearance','rest_lengths','trajectory_width','trajectory_depth')
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


def contact_diagnostics(s,theta,out,output_state=None):
    c=s.c;device=next(s.parameters()).device
    ts=np.linspace(0,c.T,2001)
    with torch.no_grad():
        q,v,_=s.parcels.motion(torch.as_tensor(ts[:,None],device=device,dtype=torch.float64),theta)
    q,v=q.cpu().numpy(),v.cpu().numpy()
    if output_state is not None:
        q,v=output_state
    cols=[ts];headers=['t'];pairs={}
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
    axes[0].set(ylabel='Center distance / diameter',title='Selected output solid / PINN dashed')
    axes[0].legend(ncol=4,fontsize=8)
    axes[1].set(xlabel='t',ylabel='Overlap / diameter (selected output)')
    fig.suptitle('CONTACT DIAGNOSTICS | '+('OUTPUT CHECKS PASSED' if passed else 'UNCONVERGED - DIAGNOSTIC ONLY'))
    fig.savefig(out/'contact_profiles.png',dpi=200);fig.savefig(out/'contact_profiles.pdf');plt.close(fig)


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



class NumpyCarrier:
    """Exact float64 forward evaluation for tiny ODE batches, not interpolation."""
    def __init__(self,fluid):
        self.c=replace(fluid.c)
        self.base=[(l.weight.detach().cpu().numpy().copy(),l.bias.detach().cpu().numpy().copy()) for l in fluid.base.layers]
        self.delta=[(l.weight.detach().cpu().numpy().copy(),l.bias.detach().cpu().numpy().copy()) for l in fluid.delta.layers]
        for layers in (self.base,self.delta):
            w,b=layers[-1];layers[-1]=(w[:2],b[:2])
        self.has_delta=bool(np.any(self.delta[-1][0]) or np.any(self.delta[-1][1]))

    @staticmethod
    def forward(x,layers):
        for k,(weight,bias) in enumerate(layers):
            x=x@weight.T+bias
            if k<len(layers)-1:x=np.tanh(x)
        return x

    def velocity(self,q,t,theta):
        c=self.c;shape=np.asarray(q).shape
        x=np.clip(np.asarray(q)[...,0].reshape(-1,1),0,c.L)
        y=np.asarray(q)[...,1].reshape(-1,1)
        xi=x/c.L;vcoord=2*y/c.H-1;f=xi*(1-vcoord**2);vf=f*(1-xi)
        raw=self.forward(np.c_[2*xi-1,2*vcoord**2-1],self.base)
        uv=np.c_[c.U*(1-vcoord[:,0]**2)+c.U*f[:,0]*raw[:,0],c.U*vf[:,0]*vcoord[:,0]*raw[:,1]]
        if self.has_delta:
            times=np.repeat(np.broadcast_to(np.asarray(t),shape[:-2]).reshape(-1),4)[:,None]
            box=np.asarray(c.ic_bounds)
            cond=np.broadcast_to(2*(np.asarray(theta)-box[:,0])/(box[:,1]-box[:,0])-1,(len(x),4))
            features=np.c_[2*xi-1,vcoord,2*times/c.T-1,cond]
            raw=self.forward(features,self.delta)
            gate=-np.expm1(-times/c.St);loading=c.beta/max(c.beta,1e-4)
            uv+=loading*np.c_[c.U*f[:,0]*gate[:,0]*raw[:,0],c.U*vf[:,0]*gate[:,0]*raw[:,1]]
        return uv.reshape(shape)


class NumericalTrajectory:
    """Independent stiff ODE solution. No position projection or altered paths."""
    def __init__(self,s,theta,fluid=None):
        self.c=s.c;self.theta=np.asarray(theta,dtype=float)
        require_valid_initial(self.theta,self.c)
        self.fluid=fluid if isinstance(fluid,NumpyCarrier) else NumpyCarrier(s.fluid if fluid is None else fluid)
        self.q0=initial_positions_np(self.theta,self.c)
        self.rest=self.q0 if self.c.rest_lengths=='case' else np.asarray(self.c.q0)
        self.sol=None

    def carrier(self,q,t):
        return self.fluid.velocity(q,t,self.theta)

    def rhs(self,t,state):
        q,v=state[:8].reshape(4,2),state[8:].reshape(4,2)
        active=(q[:,0]>0)&(q[:,0]<self.c.L)
        a=(self.carrier(q,t)-v)/self.c.St*active[:,None]+mechanics_np(q,v,self.c,self.rest)
        return np.r_[v.ravel(),a.ravel()]

    def solve(self):
        c=self.c;events=[];self.event_names=[]
        for i,j in EDGES:
            def event(t,state,i=i,j=j):
                q=state[:8].reshape(4,2)
                if not (0<q[i,0]<c.L and 0<q[j,0]<c.L):return 1.
                return np.linalg.norm(q[i]-q[j])-2*c.rp
            event.direction=-1;event.terminal=False;events.append(event)
            self.event_names.append(f'pair_{i+1}_{j+1}')
        for i in range(4):
            for wall in range(4):
                def event(t,state,i=i,wall=wall):
                    q=state[:8].reshape(4,2)
                    if not 0<q[i,0]<c.L:return 1.
                    return gaps_np(q[i],c)[wall]-c.rp
                event.direction=-1;event.terminal=False;events.append(event)
                self.event_names.append(f'wall_{i+1}_{wall}')
        # Tiny 16-state systems suffer from threaded BLAS overhead. Torch/GPU
        # parallelism for PINN training is unaffected by this scoped limit.
        with threadpool_limits(limits=1,user_api='blas'):
            self.sol=solve_ivp(self.rhs,(0,c.T),np.r_[self.q0.ravel(),np.zeros(8)],
                method='Radau',dense_output=True,rtol=c.reference_rtol,atol=c.reference_atol,
                max_step=min(c.reference_max_step,c.St/8),events=events)
        if not self.sol.success:raise RuntimeError('Numerical trajectory failed: '+self.sol.message)
        return self

    def state(self,times,acceleration=True):
        times=np.asarray(times,dtype=float).reshape(-1)
        state=self.sol.sol(times).T;q=state[:,:8].reshape(-1,4,2);v=state[:,8:].reshape(-1,4,2)
        if not acceleration:return q,v
        active=(q[:,:,0]>0)&(q[:,:,0]<self.c.L)
        drag=(self.carrier(q,times)-v)/self.c.St*active[:,:,None]
        ac=np.stack([mechanics_np(a,b,self.c,self.rest) for a,b in zip(q,v)])
        return q,v,drag+ac

    def events(self):
        return {name:times.tolist() for name,times in zip(self.event_names,self.sol.t_events)}


class TrajectoryBank:
    """Cubic Hermite q/v source, on exact stored conditions, separated from the PINN."""
    def __init__(self,s,conditions,references=None,quiet=False):
        self.c=s.c;self.conditions=np.asarray(conditions,dtype=float)
        require_valid_initial(self.conditions,s.c)
        shared=NumpyCarrier(s.fluid)
        if references is not None:self.references=list(references)
        elif len(conditions)>1 and s.c.reference_workers>1:
            self.references=parallel_references(s,self.conditions,shared,quiet)
        else:
            self.references=[]
            for k,theta in enumerate(self.conditions):
                if not quiet:print(f'Numerical anchor {k+1}/{len(conditions)}',flush=True)
                self.references.append(NumericalTrajectory(s,theta,shared).solve())
        self.times=np.linspace(0,s.c.T,max(2001,s.c.reference_times))
        pairs=[r.state(self.times,acceleration=False) for r in self.references]
        device=next(s.parameters()).device
        self.qt=torch.as_tensor(np.stack([p[0] for p in pairs]),device=device,dtype=torch.float64)
        self.vt=torch.as_tensor(np.stack([p[1] for p in pairs]),device=device,dtype=torch.float64)
        self.theta=torch.as_tensor(self.conditions,device=device,dtype=torch.float64)
        self.dt=float(self.times[1]-self.times[0])

    def state(self,t,theta):
        distance=(theta[:,None,:]-self.theta[None,:,:]).abs().amax(-1)
        error,ids=distance.min(-1)
        if bool((error>1e-9).any()):raise ValueError('Numerical source requested an unregistered initial condition')
        frac=(t[:,0]/self.dt).clamp(0,len(self.times)-1)
        index=frac.floor().to(torch.long).clamp(max=len(self.times)-2)
        r=(frac-index)[:,None,None];dt=self.dt
        q0,q1=self.qt[ids,index],self.qt[ids,index+1]
        v0,v1=self.vt[ids,index],self.vt[ids,index+1]
        q=(2*r**3-3*r**2+1)*q0+(r**3-2*r**2+r)*dt*v0+(-2*r**3+3*r**2)*q1+(r**3-r**2)*dt*v1
        v=(6*r**2-6*r)/dt*q0+(3*r**2-4*r+1)*v0+(-6*r**2+6*r)/dt*q1+(3*r**2-2*r)*v1
        a=(12*r-6)/dt**2*q0+(6*r-4)/dt*v0+(-12*r+6)/dt**2*q1+(6*r-2)/dt*v1
        return q,v,a

    def interpolation_check(self,s):
        rng=np.random.default_rng(s.c.seed+66091);worst_q=worst_v=0.
        device=next(s.parameters()).device
        for theta,reference in zip(self.conditions,self.references):
            times=rng.uniform(0,s.c.T,129)
            # Include neighborhoods of contact onset, where interpolation is hardest.
            extra=[float(x)+sign*self.dt/3 for values in reference.events().values()
                   for x in values for sign in (-1,1)]
            times=np.r_[times,np.clip(extra,0,s.c.T)]
            tt=torch.as_tensor(times[:,None],device=device,dtype=torch.float64)
            with torch.no_grad():q,v,_=self.state(tt,condition_like(tt,s.c,theta))
            rq,rv=reference.state(times,acceleration=False)
            worst_q=max(worst_q,float(np.max(np.abs(q.cpu().numpy()-rq))/s.c.H))
            worst_v=max(worst_v,float(np.max(np.abs(v.cpu().numpy()-rv))/s.c.U))
        return {'position_error_over_H':worst_q,'velocity_error_over_U':worst_v,
                'passed':worst_q<1e-4 and worst_v<1e-3,'time_knots':len(self.times)}

    def supervision(self,s):
        times=np.linspace(0,s.c.T,s.c.reference_times)
        # Resolve initial relaxation and both sides of numerical contact onset.
        times=np.unique(np.r_[times,np.linspace(0,min(3*s.c.St,s.c.T),65)])
        xs=[];qs=[];vs=[];accs=[]
        for theta,reference in zip(self.conditions,self.references):
            events=[x for values in reference.events().values() for x in values]
            around=[x+offset for x in events for offset in np.linspace(-.006,.010,33)]
            tt=np.unique(np.r_[times,np.clip(around,0,s.c.T)])
            q,v,a=reference.state(tt,acceleration=True)
            xs.append(np.c_[tt,np.broadcast_to(theta,(len(tt),4))]);qs.append(q);vs.append(v);accs.append(a)
        sp=Samples(s,s.c.seed)
        return {'x':sp.tensor(np.concatenate(xs)), 'q':sp.tensor(np.concatenate(qs)),
                'v':sp.tensor(np.concatenate(vs)),'a':sp.tensor(np.concatenate(accs))}


def fit_surrogate(s,bank,history,steps=None,lbfgs=None):
    """Numerical supervision plus actual particle physics; fluid weights stay fixed."""
    c=s.c;steps=c.surrogate_adam if steps is None else steps
    lbfgs=c.surrogate_lbfgs if lbfgs is None else lbfgs
    for p in s.parameters():p.requires_grad_(False)
    for p in s.parcels.net.parameters():p.requires_grad_(True)
    data=bank.supervision(s);params=list(s.parcels.net.parameters())
    rng=np.random.default_rng(c.seed+55107)
    def loss_for(indices,physics_weight):
        x=data['x'][indices];qtarget=data['q'][indices];vtarget=data['v'][indices]
        q,v,a=s.parcels.motion(x);q0=initial_positions(x[:,1:],c)
        qloss=((q-qtarget)/c.H).square().mean()
        vloss=((v-vtarget)/c.U).square().mean()
        aloss=(c.St*(a-data['a'][indices])/c.U).square().mean()
        t=x[:,:1];h=(t+c.St*torch.expm1(-t/c.St))[:,:,None]
        # The raw motion correction is easier to learn than a near-zero initial velocity.
        hint=s.parcels.initial_velocity_hint(q0)
        raw=s.parcels.net(torch.cat((2*t/c.T-1,normalized_theta(x[:,1:],c)),1)).reshape(-1,4,2)
        gate=(-torch.expm1(-t/c.St))[:,:,None]
        target=((qtarget-q0)/h.clamp_min(1e-7)-hint)/(c.U*gate.clamp_min(1e-7))
        valid=(t[:,0]>1e-4)
        rawloss=(raw[valid]-target[valid]).square().mean() if bool(valid.any()) else qloss*0
        mechanics_acc,active,gaps=mechanics(q,v,c,q0 if c.rest_lengths=='case' else None)
        xy=torch.stack((q[:,:,0].clamp(0,c.L),q[:,:,1]),-1)
        z=torch.cat((xy.reshape(-1,2),x[:,0:1].repeat_interleave(4,0),x[:,1:].repeat_interleave(4,0)),1)
        uv=s.fluid.fields(z)[:,:2].reshape(-1,4,2)
        ode=(c.St*(a-(uv-v)/c.St*active[:,:,None]-mechanics_acc)/c.U).square().mean()
        wall=((c.rp-gaps).clamp_min(0)/c.rp).square().mul(active[:,:,None]).mean()
        loss=10*qloss+2*vloss+rawloss+.2*aloss+physics_weight*ode+5*wall
        return loss,{'q_supervision':qloss,'v_supervision':vloss,'a_supervision':aloss,'raw_supervision':rawloss,'particle':ode,'wall_penalty':wall}
    def record(step,loss,parts):
        row={'stage':'surrogate','step':step,'total':float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
        history.append(row)
        print(f'surrogate {step:5d} | q_MSE={row["q_supervision"]:.3e} | v_MSE={row["v_supervision"]:.3e} | ODE={row["particle"]:.3e}',flush=True)
    opt=torch.optim.Adam(params,lr=8e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(steps,1),eta_min=3e-5)
    started=time.perf_counter();print(f'Surrogate anchors: {len(data["x"])} states; Adam {steps}; L-BFGS {lbfgs}',flush=True)
    best=math.inf;best_state=None;monitor=np.arange(0,len(data['x']),max(1,len(data['x'])//2048))
    for step in range(1,steps+1):
        ids=rng.integers(0,len(data['x']),size=min(512,len(data['x'])))
        # Contact stiffness can dominate gradients before paths are learned.
        physics_weight=.05*min(1,max(0,(step/ max(steps,1)-.7)/.3))
        opt.zero_grad(set_to_none=True);loss,parts=loss_for(ids,physics_weight)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite surrogate loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,10.);opt.step();schedule.step()
        if step==1 or step%c.log_every==0 or step==steps:
            record(step,loss,parts)
            with torch.no_grad():
                score=float(loss_for(monitor,.05)[0])
            if score<best:
                best=score;best_state={k:v.detach().cpu().clone() for k,v in s.parcels.net.state_dict().items()}
    if best_state is not None:s.parcels.net.load_state_dict(best_state)
    if lbfgs:
        # Fixed reference targets throughout optimization; preserve the best fit.
        ids=np.arange(0,len(data['x']),max(1,len(data['x'])//4096))
        before={k:v.detach().cpu().clone() for k,v in s.parcels.net.state_dict().items()}
        with torch.no_grad():before_score=float(loss_for(ids,.05)[0])
        opt=torch.optim.LBFGS(params,max_iter=lbfgs,history_size=40,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
        def closure():
            opt.zero_grad(set_to_none=True);loss,_=loss_for(ids,.05);loss.backward();return loss
        opt.step(closure)
        with torch.no_grad():after_score=float(loss_for(ids,.05)[0])
        if after_score>before_score:s.parcels.net.load_state_dict(before);print('Restored the best surrogate fit after L-BFGS.',flush=True)
    print(f'Surrogate fit completed in {time.perf_counter()-started:.1f} s',flush=True)
    return {'states':len(data['x']),'conditions':len(bank.conditions),'reference_interpolation':bank.interpolation_check(s)}


class DenseReference:
    """Replay the solver's actual dense polynomial, without pickle or resampling."""
    def __init__(self,ts,q,yold):self.ts=ts;self.q=q;self.yold=yold
    def __call__(self,t):
        t=np.asarray(t);scalar=t.ndim==0;times=t.reshape(-1)
        ids=np.searchsorted(self.ts,times,side='right')-1
        ids=np.clip(ids,0,len(self.ts)-2)
        x=(times-self.ts[ids])/(self.ts[ids+1]-self.ts[ids])
        powers=np.stack((x,x*x,x*x*x),-1)
        values=self.yold[ids]+(self.q[ids]*powers[:,None,:]).sum(-1)
        return values[0] if scalar else values.T


def _reference_job(payload):
    from types import SimpleNamespace
    c=Config(**payload['config'])
    carrier=NumpyCarrier.__new__(NumpyCarrier);carrier.c=replace(c)
    carrier.base=payload['base'];carrier.delta=payload['delta']
    carrier.has_delta=bool(np.any(carrier.delta[-1][0]) or np.any(carrier.delta[-1][1]))
    reference=NumericalTrajectory(SimpleNamespace(c=c),payload['theta'],carrier).solve()
    solution=reference.sol
    return {'ts':solution.sol.ts,'Q':np.stack([r.Q for r in solution.sol.interpolants]),
            'yold':np.stack([r.y_old for r in solution.sol.interpolants]),
            'events':solution.t_events,'names':reference.event_names,
            'nfev':solution.nfev,'njev':solution.njev,'nlu':solution.nlu}


def parallel_references(s,conditions,carrier,quiet=False):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from types import SimpleNamespace
    import multiprocessing
    import sys
    main_file=getattr(sys.modules.get('__main__'),'__file__',None)
    if not main_file or not Path(main_file).is_file():
        if not quiet:print('Interactive kernel: using serial numerical anchors. Run the saved .py script to enable parallel workers.',flush=True)
        return [NumericalTrajectory(s,theta,carrier).solve() for theta in conditions]
    payloads=[{'config':asdict(s.c),'theta':theta,'base':carrier.base,'delta':carrier.delta} for theta in conditions]
    outputs=[None]*len(conditions)
    try:
        with ProcessPoolExecutor(max_workers=min(s.c.reference_workers,len(conditions)),
             mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(_reference_job,p):i for i,p in enumerate(payloads)}
            for finished,future in enumerate(as_completed(pending),1):
                i=pending[future];data=future.result()
                reference=NumericalTrajectory(s,conditions[i],carrier)
                reference.event_names=data['names']
                reference.sol=SimpleNamespace(sol=DenseReference(data['ts'],data['Q'],data['yold']),
                    t_events=data['events'],t=data['ts'],success=True,nfev=data['nfev'],njev=data['njev'],nlu=data['nlu'])
                outputs[i]=reference
                if not quiet:print(f'Numerical anchors completed {finished}/{len(conditions)}',flush=True)
    except (OSError,RuntimeError) as error:
        if not quiet:print(f'Parallel anchors unavailable ({type(error).__name__}); continuing serially.',flush=True)
        for i,theta in enumerate(conditions):
            if outputs[i] is None:outputs[i]=NumericalTrajectory(s,theta,carrier).solve()
    return outputs


def polish_carrier(s,sampler,history):
    """A single validation seed is insufficient for a marginal steady solution."""
    reports=[screen(s,True,seed_offset=offset)[0] for offset in (0,91001,91002)]
    score=max(carrier_score(r) for r in reports)
    if score<=.8 or not (s.c.carrier_polish_adam or s.c.carrier_polish_lbfgs):
        return {'performed':False,'worst_score':score}
    before={k:v.detach().cpu().clone() for k,v in s.fluid.base.state_dict().items()}
    candidate=sampler.interior(4096,True,s.c.T)
    scores=residual_score(s,candidate,True)
    extra=candidate[scores.topk(256).indices].detach()
    print(f'Carrier robustness polish: worst scaled check={score:.3f}',flush=True)
    train(s,sampler,'steady',s.c.carrier_polish_adam,s.c.carrier_polish_lbfgs,history,extra,lr=min(s.c.steady_lr,1e-4))
    new_reports=[screen(s,True,seed_offset=offset)[0] for offset in (0,91001,91002)]
    new_score=max(carrier_score(r) for r in new_reports)
    accepted=bool(new_score<score)
    if not accepted:s.fluid.base.load_state_dict(before)
    else:
        with torch.no_grad():
            s.fluid.delta.layers[-1].weight.zero_();s.fluid.delta.layers[-1].bias.zero_()
        print('Improved steady carrier; reset the old transient correction before coupling.',flush=True)
    return {'performed':True,'accepted':accepted,'before_score':score,'after_score':new_score,
            'reports':new_reports,'scope':'Three independent tuning seeds; full case validation still runs afterward'}


def fluid_monitor(s,conditions):
    """Independent tuning set; numerical paths, not surrogate paths, feed the PDE."""
    reports=[];original=s.reference_bank
    try:
        for k,theta in enumerate(conditions):
            reference=NumericalTrajectory(s,theta).solve()
            s.reference_bank=TrajectoryBank(s,[theta],[reference],quiet=True)
            report,_=screen(s,theta=theta,seed_offset=32001+k)
            reports.append(report)
    finally:s.reference_bank=original
    return max(carrier_score(r) for r in reports),reports


def training_conditions(c):
    """Cover the requested parameter-box boundary, not only random interiors."""
    rng=np.random.default_rng(c.seed+51003);anchors=[]
    if c.boundary_anchors:
        proposed=[nominal_theta(c)]
        proposed.extend(np.asarray(x) for x in itertools.product(*np.asarray(c.ic_bounds)))
        box=np.asarray(c.ic_bounds);first=(box[0,0]+box[0,1])/2
        distance=2*c.rp+c.initial_clearance+.0005
        for y in (.10,.14,.19):proposed.append(np.array([first,y,first+distance,y]))
        for theta in proposed:
            if bool(valid_initial(theta,c)) and not any(np.array_equal(theta,a) for a in anchors):anchors.append(theta)
            if len(anchors)==c.reference_cases:return np.asarray(anchors)
    if len(anchors)<c.reference_cases:anchors.extend(sample_initial(rng,c.reference_cases-len(anchors),c))
    return np.asarray(anchors)


def safeguarded_fluid_fit(s,bank,history,conditions):
    """Keep fixed particle anchors while optimizing fluid; roll back degradation."""
    s.reference_bank=bank;sp=Samples(s,s.c.seed+39041);sp.condition_bank=bank.conditions
    before={k:v.detach().cpu().clone() for k,v in s.fluid.delta.state_dict().items()}
    old_score,old_reports=fluid_monitor(s,conditions)
    print(f'Fluid monitor before update: worst scaled check={old_score:.3f}',flush=True)
    try:
        train(s,sp,'fluid',s.c.fluid_adam,s.c.fluid_lbfgs,history,lr=min(s.c.coupled_lr,1e-4))
        new_score,new_reports=fluid_monitor(s,conditions)
    except (RuntimeError,FloatingPointError) as error:
        s.fluid.delta.load_state_dict(before);s.reference_bank=bank
        return {'accepted':False,'before_score':old_score,'reason':str(error)}
    except KeyboardInterrupt:
        s.fluid.delta.load_state_dict(before);s.reference_bank=bank
        raise
    accepted=bool(math.isfinite(new_score) and new_score<=old_score)
    if not accepted:s.fluid.delta.load_state_dict(before)
    s.reference_bank=bank
    print(f'Fluid monitor after update: {new_score:.3f}; accepted={accepted}',flush=True)
    return {'accepted':accepted,'before_score':old_score,'after_score':new_score,
            'before_reports':old_reports,'after_reports':new_reports,
            'scope':'Tuning conditions; final evaluation uses a separate set'}


def reference_checks(s,theta,reference):
    """Check actual numerical paths and repeat integration with tighter settings."""
    c=s.c;times=np.linspace(0,c.T,501);q,v,a=reference.state(times)
    fine=NumericalTrajectory(s,theta,reference.fluid)
    fine.c=replace(c,reference_rtol=c.reference_rtol/10,reference_atol=c.reference_atol/10,
                   reference_max_step=c.reference_max_step/2)
    fine.solve();fq,fv=fine.state(times,acceleration=False)
    pe=float(np.max(np.abs(q-fq))/c.H);ve=float(np.max(np.abs(v-fv))/c.U)
    active=(q[:,:,0]>0)&(q[:,:,0]<c.L)
    gaps=gaps_np(q,c);overlap=float((np.maximum(c.rp-gaps,0)*active[:,:,None]/c.rp).max())
    pair=min(float(np.linalg.norm(q[:,i]-q[:,j],axis=-1).min()) for i,j in EDGES)
    checks={'integrator_success':True,'time_refinement':pe<1e-4 and ve<1e-3,
            'wall_overlap':overlap<.1,'pair_overlap':pair>1.8*c.rp,'finite':bool(np.isfinite(q).all() and np.isfinite(v).all())}
    checks['contact_refinement']=all(bool(reference.events()[name])==bool(fine.events()[name]) for name in reference.event_names)
    return {'passed':bool(all(checks.values())),'checks':checks,'position_refinement_error_over_H':pe,
            'velocity_refinement_error_over_U':ve,'max_wall_overlap_fraction':overlap,
            'minimum_pair_distance':pair,'contact_onset_times':reference.events(),
            'scope':'Stiff ODE in the learned fluid; tighter-step comparison does not independently validate CFD'},(q,v,a)


def wall_diagnostics(s,theta,reference,out):
    c=s.c;ts=np.linspace(0,c.T,2001);q,v=reference.state(ts,acceleration=False)
    gaps=gaps_np(q,c);active=(q[:,:,0]>0)&(q[:,:,0]<c.L)
    up=np.stack((q[:,:,1]*0,q[:,:,1]*0+1),-1);normals=[up,-up]
    for cy in (0.,c.H):
        dr=q-np.array([c.xc,cy]);normals.append(dr/np.maximum(np.linalg.norm(dr,axis=-1)[:,:,None],1e-12))
    normals=np.stack(normals,axis=2)
    normal_velocity=(v[:,:,None,:]*normals).sum(-1)
    force=np.maximum(c.wall_kn*(c.rp-gaps)-c.wall_cn*normal_velocity,0)*(gaps<c.rp)*active[:,:,None]
    magnitude=np.linalg.norm((force[:,:,:,None]*normals).sum(2),axis=-1)
    penetration=np.maximum(c.rp-gaps,0)*active[:,:,None]/c.rp
    columns=np.c_[ts,gaps.min(-1),penetration.max(-1),magnitude]
    header='t,'+','.join(f'{kind}_p{i+1}' for kind in ('minimum_wall_gap','maximum_overlap_over_radius','wall_acceleration') for i in range(4))
    np.savetxt(out/'wall_contacts.csv',columns,delimiter=',',comments='',header=header)
    result={'minimum_gap_over_radius':float(gaps.min()/c.rp),'maximum_wall_acceleration':float(magnitude.max()),
            'maximum_overlap_over_radius':float(penetration.max()),'onset_events':{k:v for k,v in reference.events().items() if k.startswith('wall_')},
            'scope':'Numerical particle-wall contacts; spring connectors have no wall surface'}
    (out/'wall_contacts.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True,constrained_layout=True)
    for i in range(4):
        axes[0].plot(ts,gaps[:,i].min(-1)/c.rp,label=f'Particle {i+1}')
        axes[1].plot(ts,magnitude[:,i])
    axes[0].axhline(1,color='black',ls=':',label='Contact threshold')
    axes[0].set(ylabel='Minimum wall distance / radius');axes[0].legend(ncol=3,fontsize=8)
    axes[1].set(xlabel='t',ylabel='Wall acceleration')
    fig.suptitle('Numerical wall-contact diagnostics')
    fig.savefig(out/'wall_contacts.png',dpi=180);fig.savefig(out/'wall_contacts.pdf');plt.close(fig)
    return result


def validate_cases(s,conditions,out,history,steady_report,smoke=False,gif=True):
    """Always export every requested case; surrogate failure triggers an explicit ODE fallback."""
    out.mkdir(parents=True,exist_ok=True);reports=[];old_bank=s.reference_bank
    for k,theta in enumerate(conditions):
        folder=out/f'case_{k:03d}';folder.mkdir(parents=True,exist_ok=True)
        print(f'Validation case {k+1}/{len(conditions)} | inputs={np.asarray(theta).tolist()}',flush=True)
        report={'initial_inputs':np.asarray(theta).tolist(),'smoke_test':smoke,
                'external_CFD_validated':False,'bond_material_calibrated':False,
                'condition_seen_in_training':any(np.array_equal(theta,np.asarray(a)) for a in getattr(s,'training_conditions',[]))}
        try:
            require_valid_initial(theta,s.c)
            reference=NumericalTrajectory(s,theta).solve()
            bank=TrajectoryBank(s,[theta],[reference],quiet=True);s.reference_bank=bank
            fluid_report,_=screen(s,theta=theta,seed_offset=1009*k)
            print_screen(fluid_report,f'Case {k}')
            numerical_report,numerical_state=reference_checks(s,theta,reference)
            t=torch.as_tensor(np.linspace(0,s.c.T,501)[:,None],device=next(s.parameters()).device,dtype=torch.float64)
            with torch.no_grad():pq,pv,_=s.parcels.motion(t,theta)
            rq,rv,ra=numerical_state
            pe=float(np.max(np.abs(pq.cpu().numpy()-rq))/s.c.H)
            ve=float(np.max(np.abs(pv.cpu().numpy()-rv))/s.c.U)
            surrogate_passed=bool(fluid_report['passed'] and pe<.01 and ve<.03)
            fluid_keys=('mixed_rms','primitive_rms','flux','no_slip','outlet')
            field_passed=bool(all(fluid_report['checks'][key] for key in fluid_keys))
            interpolation=bank.interpolation_check(s)
            numerical_validated=bool(field_passed and steady_report['passed'] and numerical_report['passed'] and interpolation['passed'] and not smoke)
            report.update(fluid_report)
            report.update(numerical_reference=numerical_report,steady=steady_report,
                source_method='independent_ODE_anchors',source_interpolation=interpolation,
                Radau={'success':True,'position_error_over_H':pe,'velocity_error_over_U':ve,
                       'passed':pe<.01 and ve<.03,'contact_onset_times':reference.events(),
                       'scope':'surrogate vs independent ODE in the SAME learned fluid'},
                fluid_passed=field_passed,surrogate_passed=surrogate_passed,
                passed=bool(surrogate_passed and numerical_validated),
                output_method='Radau' if not surrogate_passed else 'validated_PINN',
                output_validated=numerical_validated,eligible_for_numerical_use=numerical_validated,
                initial_positions=initial_positions_np(theta,s.c).tolist(),rest_length_policy=s.c.rest_lengths)
            rest=reference.rest;report['bond_rest_lengths']={f'{i+1}-{j+1}':float(np.linalg.norm(rest[i]-rest[j])) for i,j in EDGES}
            # Preserve the original neural prediction alongside the selected output.
            pinn_contacts=contact_diagnostics(s,theta,folder)
            (folder/'contacts.csv').replace(folder/'PINN_contacts.csv')
            (folder/'contacts.json').replace(folder/'PINN_contacts.json')
            ts=np.linspace(0,s.c.T,2001);oq,ov=reference.state(ts,acceleration=False)
            if surrogate_passed:
                selected_state=None;selected_contacts=pinn_contacts
                # Restore primary contact files when the surrogate qualifies.
                import shutil
                shutil.copyfile(folder/'PINN_contacts.csv',folder/'contacts.csv')
                (folder/'contacts.json').write_text(json.dumps(pinn_contacts,indent=2))
            else:
                selected_state=numerical_state
                selected_contacts=contact_diagnostics(s,theta,folder,output_state=(oq,ov))
            report['contacts']=selected_contacts
            report['surrogate_contact_classification_agreement']={
                f'{i+1}-{j+1}':pinn_contacts['pairs'][f'{i+1}-{j+1}']['contact_observed_on_time_grid']==bool(reference.events()[f'pair_{i+1}_{j+1}']) for i,j in EDGES}
            if not all(report['surrogate_contact_classification_agreement'].values()) and surrogate_passed:
                # A small aggregate error cannot justify a wrong collision classification.
                surrogate_passed=False;report.update(surrogate_passed=False,passed=False,output_method='Radau')
                selected_state=numerical_state
                selected_contacts=contact_diagnostics(s,theta,folder,output_state=(oq,ov))
                report['contacts']=selected_contacts
            plot_contact_diagnostics(s,folder,None,report['output_validated'])
            report['wall_contacts']=wall_diagnostics(s,theta,reference,folder)
            (folder/'history.json').write_text(json.dumps(history,indent=2))
            (folder/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            render(s,folder,report,np.c_[rq.reshape(-1,8),rv.reshape(-1,8)],gif=gif,theta=theta,output_state=selected_state)
            print(f'Case {k}: PINN position error={100*pe:.3f}% H; velocity error={100*ve:.3f}% U; output={report["output_method"]}; output_validated={numerical_validated}',flush=True)
        except (RuntimeError,FloatingPointError,ValueError) as error:
            report.update(passed=False,output_validated=False,eligible_for_numerical_use=False,error=str(error),output_method='failed')
            (folder/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print(f'Case {k} failed: {error}; continuing remaining cases.',flush=True)
        finally:s.reference_bank=old_bank
        reports.append(report)
    summary={'passed':bool(all(r['passed'] for r in reports)),
        'all_outputs_validated':bool(all(r['output_validated'] for r in reports)),
        'smoke_test':smoke,'number_of_cases':len(reports),'conditions':np.asarray(conditions).tolist(),'cases':reports,
        'scope':'Finite evaluated cases. ODE fallback is labeled and is not a claim that the PINN generalizes over the full domain.',
        'external_CFD_validated':False,'bond_material_calibrated':False}
    (out/'validation.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


def export_results_zip(out):
    import zipfile
    archive=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Output ZIP integrity check failed')
    print(f'Complete output ZIP: {archive.resolve()}',flush=True)
    return archive


def repaired_self_test():
    parametric_self_test()
    # Direct impacts against both a flat wall and a curved stenosis, with unchanged forces.
    c=replace(Config(),bond_kn=0,bond_cn=0,pair_kn=0,pair_cn=0)
    for label,normal,center in (
        ('flat',np.array([0.,1.]),np.array([.2,0.])),
        ('curved',np.array([-1.,1.])/math.sqrt(2),np.array([c.xc,0.]))):
        radius=0 if label=='flat' else c.R
        q0=np.asarray(c.q0).copy();q0[0]=center+(radius+c.rp+.005)*normal
        v0=np.zeros((4,2));v0[0]=-.3*normal
        def rhs(t,state):
            q,v=state[:8].reshape(4,2),state[8:].reshape(4,2)
            return np.r_[v.ravel(),mechanics_np(q,v,c,q0).ravel()]
        sol=solve_ivp(rhs,(0,.08),np.r_[q0.ravel(),v0.ravel()],method='Radau',rtol=1e-10,atol=1e-12,max_step=.0002,dense_output=True)
        assert sol.success
        q=sol.y[:8].T.reshape(-1,4,2);v=sol.y[8:].T.reshape(-1,4,2)
        g=gaps_np(q,c)
        assert g[:,0].min()<c.rp and g[:,0].min()>.9*c.rp
        assert np.dot(v[-1,0],normal)>0
        impact=np.argmin(g[:,0].min(-1));qn,vn=q[impact],v[impact]
        at,_,_=mechanics(torch.tensor(qn[None]),torch.tensor(vn[None]),c,torch.tensor(q0[None]))
        assert np.max(np.abs(at[0].numpy()-mechanics_np(qn,vn,c,q0)))<1e-8
        replay=DenseReference(sol.sol.ts,np.stack([r.Q for r in sol.sol.interpolants]),np.stack([r.y_old for r in sol.sol.interpolants]))
        probes=np.random.default_rng(9231).uniform(0,.08,127)
        assert np.max(np.abs(replay(probes)-sol.sol(probes)))<1e-12
        print(f'PASS: {label} wall impact rebounds; maximum overlap={100*(c.rp-g[:,0].min())/c.rp:.3f}% radius',flush=True)
    # Hermite interpolation and time derivatives for an exactly cubic trajectory.
    c=replace(Config(),width=12,depth=2,trajectory_width=12,trajectory_depth=2)
    s=System(c).double();theta=nominal_theta(c);tt=np.linspace(0,c.T,21)
    class Cubic:
        def state(self,times,acceleration=False):
            times=np.asarray(times);q=initial_positions_np(theta,c)[None]+times[:,None,None]**3*np.ones((1,4,2))
            v=3*times[:,None,None]**2*np.ones((1,4,2));return q,v
        def events(self):return {}
    bank=TrajectoryBank(s,[theta],[Cubic()],quiet=True)
    t=torch.tensor([[0.],[.1323],[c.T]],dtype=torch.float64,requires_grad=True)
    q,v,a=bank.state(t,condition_like(t,c,theta))
    assert (q-initial_positions(condition_like(t,c,theta),c)-t[:,:,None]**3).abs().max()<1e-12
    assert (v-3*t[:,:,None]**2).abs().max()<1e-11
    assert (a-6*t[:,:,None]).abs().max()<1e-7
    print('PASS: numerical-source Hermite state and derivatives, including t=0 and t=T')
    # The accelerated ODE carrier must reproduce the actual learned network.
    rng=np.random.default_rng(83321)
    with torch.no_grad():
        s.fluid.delta.layers[-1].weight.normal_(0,.02)
        s.fluid.delta.layers[-1].bias.normal_(0,.01)
    q=rng.uniform([0,0],[c.L,c.H],size=(19,4,2));times=rng.uniform(0,c.T,19)
    for beta in (0.,5e-5,2e-4):
        s.fluid.c=replace(c,beta=beta)
        theta=sample_initial(rng,1,s.fluid.c)[0]
        z=np.c_[q.reshape(-1,2),np.repeat(times,4),np.broadcast_to(theta,(76,4))]
        with torch.no_grad():truth=s.fluid.fields(torch.tensor(z,dtype=torch.float64))[:,:2].numpy().reshape(19,4,2)
        fast=NumpyCarrier(s.fluid)
        assert np.max(np.abs(truth-fast.velocity(q,times,theta)))<1e-12
        assert np.max(np.abs(truth[0]-fast.velocity(q[0],times[0],theta)))<1e-12
    print('PASS: exact NumPy carrier vs Torch, scalar/batched states and zero/nonzero feedback')
    s.fluid.c=c
    qprobe=torch.tensor(sample_initial(np.random.default_rng(3107),8,c),dtype=torch.float64)
    for policy in ('case','nominal'):
        model=System(replace(c,rest_lengths=policy)).double()
        with torch.no_grad():model.parcels.net.layers[-1].weight.normal_(0,.02)
        q,v,a=model.parcels.motion(torch.zeros((8,1),dtype=torch.float64),qprobe)
        ac,_,_=mechanics(q,v,model.c,q if policy=='case' else None)
        uv=model.fluid.fields(q.reshape(-1,2),True)[:,:2].reshape_as(q)
        assert (a-uv/model.c.St-ac).abs().max()<1e-10
    print('PASS: exact initial acceleration with equilibrium and prestressed bonds')
    theta=nominal_theta(c)
    zz=torch.tensor(np.c_[[.2,.4,.7],[.15,.20,.26],[.04,.12,.20],np.broadcast_to(theta,(3,4))],dtype=torch.float64,requires_grad=True)
    _,full=s.fluid.fields(zz,jac=True)
    _,physical=s.fluid.fields(zz,jac=True,physical_jac=True)
    assert (full[:,:,:3]-physical).abs().max()<1e-12
    normal=fluid_residual(s,zz,primitive=True)
    lightweight=fluid_residual(s,zz,primitive=True,derivative_graph=False)
    assert (normal-lightweight).abs().max()<1e-12
    print('PASS: full vs physical Jacobians and lightweight validation retain all NS residual values')



def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test',action='store_true');p.add_argument('--quick-test',action='store_true')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--outdir',default='results_repaired');p.add_argument('--no-gif',action='store_true')
    p.add_argument('--config');p.add_argument('--resume');p.add_argument('--evaluate')
    p.add_argument('--repair-from',help='Reuse the validated base from the original parametric model; discard its failed transient/trajectory weights')
    p.add_argument('--initial',type=float,nargs=4);p.add_argument('--cases')
    p.add_argument('--steady-only',action='store_true')
    p.add_argument('--strict-validation',action='store_true',help='Raise only AFTER all output files and the ZIP are saved')
    args=p.parse_args(argv)
    if args.self_test:repaired_self_test();return
    if torch is None:raise SystemExit('PyTorch required; install requirements.txt')
    if sum(bool(x) for x in (args.resume,args.evaluate,args.repair_from))>1:p.error('Select only one checkpoint option')
    if args.initial and args.cases:p.error('Use --initial OR --cases')
    if (args.initial or args.cases) and not args.evaluate:p.error('--initial/--cases require --evaluate')
    if args.quick_test and (args.resume or args.evaluate):p.error('Smoke requires fresh training or --repair-from')
    c=Config(device=args.device,outdir=args.outdir)
    if args.config:
        overrides=json.loads(Path(args.config).read_text());unknown=set(overrides)-set(asdict(c))
        if unknown:raise ValueError('Unknown fields: '+', '.join(sorted(unknown)))
        c=replace(c,**overrides);c=replace(c,device=args.device,outdir=args.outdir)
    payload=None;old=None
    if args.resume or args.evaluate:c,payload=resume_config(args.resume or args.evaluate,c)
    if args.repair_from:
        old=torch.load(args.repair_from,map_location='cpu',weights_only=True)
        if old.get('format')!='stenosis_four_parametric_ic_v1':raise ValueError('--repair-from needs the original parametric checkpoint')
        physics=('L','H','xc','R','Re','U','T','St','beta','rp','eps','q0','bond_kn','bond_cn','pair_kn','pair_cn',
                 'wall_kn','wall_cn','width','depth','seed','variable_beads','ic_bounds','initial_clearance','rest_lengths')
        c=replace(c,**{k:old['config'][k] for k in physics})
    if args.quick_test:
        c=replace(c,batch=24,boundary=12,pde_chunk=12,particle_points=12,
            steady_adam=8,steady_lbfgs=2,refinement_rounds=0,reference_cases=3,reference_times=41,
            surrogate_adam=20,surrogate_lbfgs=2,coupling_rounds=1,fluid_adam=8,fluid_lbfgs=2,
            validation_cases=2,screening_points=128,screening_times=3,log_every=100)
        if old is None:c=replace(c,width=12,depth=2)
    c.validate();torch.set_num_threads(2);torch.manual_seed(c.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if c.device=='auto' else c.device
    s=System(c).to(device=device,dtype=torch.float64)
    s.smoke_test=bool(args.quick_test or (payload and payload.get('smoke_test',False)))
    history=list(payload.get('history',[])) if payload else []
    if payload:s.load_state_dict(payload['state_dict'],strict=True)
    s.training_conditions=list(payload.get('training_conditions',[])) if payload else []
    if old:
        s.fluid.base.load_state_dict({k.removeprefix('fluid.base.'):v for k,v in old['state_dict'].items() if k.startswith('fluid.base.')},strict=True)
        print('Loaded original STEADY carrier; reset failed transient and trajectory weights.',flush=True)
    out=Path(c.outdir)
    if out.exists() and any(out.iterdir()) and not args.resume:raise ValueError('Choose a new empty output directory')
    if args.evaluate and out.resolve()==Path(args.evaluate).resolve().parent:raise ValueError('Evaluation output must differ from checkpoint folder')
    out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(asdict(c),indent=2))
    print(f'Repaired hybrid | fluid=7 | trajectory=5 | {device} | separated optimizers | independent ODE anchors',flush=True)
    sp=Samples(s,c.seed);started=time.perf_counter()
    try:
        if not args.evaluate:
            if payload or old:
                base,_=screen(s,True);print_screen(base,'Loaded carrier')
                if not base['passed'] and not args.quick_test:train(s,sp,'steady',c.repair_adam,c.repair_lbfgs,history,lr=min(c.steady_lr,1e-4))
            else:train(s,sp,'steady',c.steady_adam,c.steady_lbfgs,history)
            base,_=screen(s,True);print_screen(base,'Steady carrier')
            best_base=base;best_state={k:v.detach().cpu().clone() for k,v in s.fluid.base.state_dict().items()}
            for k in range(c.refinement_rounds):
                if base['passed']:break
                candidate=sp.interior(4096,True,c.T);score=residual_score(s,candidate,True)
                train(s,sp,'steady',1200,600,history,candidate[score.topk(512).indices].detach(),lr=min(c.steady_lr,1e-4))
                base,_=screen(s,True)
                if carrier_score(base)<carrier_score(best_base):best_base=base;best_state={key:v.detach().cpu().clone() for key,v in s.fluid.base.state_dict().items()}
                else:s.fluid.base.load_state_dict(best_state);base=best_base
            polish=polish_carrier(s,sp,history) if not args.quick_test else {'performed':False}
            base,_=screen(s,True);print_screen(base,'Robust steady carrier')
            (out/'carrier_robustness.json').write_text(json.dumps(polish,indent=2))
            (out/'steady_validation.json').write_text(json.dumps(base,indent=2));save(s,history,out)
            if not base['passed']:print('WARNING: steady checks failed; continuing with unvalidated-output labels.',flush=True)
            if args.steady_only:
                if c.export_zip:export_results_zip(out)
                return
            for p0 in s.fluid.base.parameters():p0.requires_grad_(False)
            conditions=training_conditions(c);s.training_conditions=conditions.tolist()
            tuning=sample_initial(np.random.default_rng(c.seed+51004),3,c)
            bank=TrajectoryBank(s,conditions)
            anchor_report=fit_surrogate(s,bank,history)
            coupling=[]
            for k in range(c.coupling_rounds):
                print(f'Coupling round {k+1}/{c.coupling_rounds}',flush=True)
                update=safeguarded_fluid_fit(s,bank,history,tuning);coupling.append(update)
                if update['accepted']:bank=TrajectoryBank(s,conditions)
                else:print('Reusing exact anchors after fluid rollback; carrier weights are unchanged.',flush=True)
                anchor_report=fit_surrogate(s,bank,history,steps=20 if args.quick_test else max(300,c.surrogate_adam//3),lbfgs=2 if args.quick_test else c.surrogate_lbfgs//2)
            s.reference_bank=None
            (out/'training_diagnostics.json').write_text(json.dumps({'anchors':anchor_report,'training_conditions':conditions.tolist(),'tuning_conditions':tuning.tolist(),'coupling':coupling},indent=2))
        else:
            base,_=screen(s,True);print_screen(base,'Steady carrier')
        conditions=np.asarray([args.initial] if args.initial else json.loads(Path(args.cases).read_text()) if args.cases else validation_conditions(c),dtype=float)
        if conditions.ndim!=2 or conditions.shape[1]!=4 or len(conditions)==0:raise ValueError('Expected nonempty conditions list with four columns')
        report=validate_cases(s,conditions,out,history,base,smoke=s.smoke_test,gif=not args.no_gif)
        report['seconds']=time.perf_counter()-started
        if not args.evaluate:save(s,history,out,report)
        if c.export_zip:export_results_zip(out)
        print(f'Completed all cases | PINN_passed={report["passed"]} | all_outputs_validated={report["all_outputs_validated"]}',flush=True)
        if not report['passed']:print('WARNING: surrogate checks failed. Inspect per-case output_method and output_validated; numerical fallback is explicitly labeled.',flush=True)
        if args.strict_validation and not report['passed']:
            raise RuntimeError('Validation failed; ALL generated results and ZIP were already saved')
    except (KeyboardInterrupt,RuntimeError,FloatingPointError):
        if not args.evaluate:save(s,history,out)
        raise


if __name__=='__main__':main()
