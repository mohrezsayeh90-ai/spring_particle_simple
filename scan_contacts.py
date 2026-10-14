"""Find initial conditions worth validating; scan results are surrogate predictions.

python scan_contacts.py --model results_parametric/model.pt --samples 256 --outdir contact_scan
Then run stenosis_parametric_ic.py --evaluate ... --cases contact_scan/selected_cases.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
from stenosis_parametric_ic import (Config, EDGES, System, resume_config,
                                   sample_initial, nominal_theta, torch)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--time-points',type=int,default=1001)
    p.add_argument('--seed',type=int,default=9417)
    p.add_argument('--outdir',default='contact_scan')
    args=p.parse_args()
    if torch is None:p.error('PyTorch required')
    if args.samples<2 or args.time_points<2:p.error('samples/time-points must be >= 2')
    out=Path(args.outdir)
    if out.exists() and any(out.iterdir()):p.error('Choose an empty output directory')
    c,payload=resume_config(args.model,Config())
    c.validate();torch.set_num_threads(2)
    s=System(c).double();s.load_state_dict(payload['state_dict'],strict=True);s.eval()
    conditions=np.vstack((nominal_theta(c),sample_initial(np.random.default_rng(args.seed),args.samples-1,c)))
    times=torch.tensor(np.linspace(0,c.T,args.time_points)[:,None],dtype=torch.float64)
    rows=[]
    with torch.no_grad():
        for case,theta in enumerate(conditions):
            q=s.parcels.motion(times,theta)[0].numpy()
            active=(q[:,:,0]>0)&(q[:,:,0]<c.L)
            minimum=[];overlap=[]
            for i,j in EDGES:
                d=np.linalg.norm(q[:,i]-q[:,j],axis=1)
                minimum.append(float(d.min()))
                overlap.append(float((np.maximum(2*c.rp-d,0)*active[:,i]*active[:,j]).max()/(2*c.rp)))
            rows.append([case,*theta,min(minimum),max(overlap),*minimum])
    out.mkdir(parents=True,exist_ok=True)
    rows=np.asarray(rows);most_overlap=int(rows[np.argmax(rows[:,6]),0]);order=np.argsort(rows[:,5])
    rows=rows[order]
    np.savetxt(out/'scan.csv',rows,delimiter=',',comments='',
        header='case_id,xa0,ya0,xb0,yb0,min_pair_distance,max_overlap_over_diameter,'+
        ','.join(f'dmin_{i+1}_{j+1}' for i,j in EDGES))
    # Closest, most overlapping and most separated predictions. All require evaluation.
    idxs=[int(order[0]),most_overlap,int(order[-1])]
    idxs=list(dict.fromkeys(idxs))
    selected=conditions[idxs]
    (out/'selected_cases.json').write_text(json.dumps(selected.tolist(),indent=2))
    metadata={'status':'UNVALIDATED_SURROGATE_SCAN','smoke_checkpoint':bool(payload.get('smoke_test',False)),
        'model':str(Path(args.model).resolve()),'samples':args.samples,'time_points':args.time_points,
        'seed':args.seed,'selected_case_ids':idxs,
        'contact_predicted_on_grid':int(np.sum(rows[:,6]>0)),
        'next_step':'Evaluate selected_cases.json with full fluid/ODE screening and independent Radau. Absence of sampled contact is not a continuous-time proof.'}
    (out/'scan_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
