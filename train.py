"""Train a Bayer-to-RGB model on registered pairs; inputs may be RAW, NPY or RGB."""
import argparse, random, json, time, copy
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from model import ModISP,load_model
from isp import decode_bayer,mosaic_rgb,pack,RAW_EXT,white_balance

EXT=RAW_EXT|{'.npy','.mat','.png','.jpg','.jpeg','.tif','.tiff'}
def pairs(root):
    targets={p.stem:p for p in (root/'target').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}}
    files=[(p,targets[p.stem]) for p in sorted((root/'input').iterdir()) if p.suffix.lower() in EXT and p.stem in targets]
    if not files: raise ValueError('No matching input/target stems in '+str(root))
    return files

def read_pair(src,tgt):
    cfa_shift=(0,0)
    if src.suffix.lower()=='.mat':
        from scipy.io import loadmat
        raw=loadmat(src).get('data')
        if not isinstance(raw,np.ndarray) or raw.ndim!=2: raise ValueError(f'{src.name} must contain a 2-D Bayer mosaic under key data')
        meta_path=src.parent.parent/'metadata'/f'{src.stem}.MAT'
        if not meta_path.exists(): meta_path=src.with_name(src.name.replace('_NOISY_RAW_','_METADATA_RAW_'))
        meta=loadmat(meta_path)
        cfa=str(np.asarray(meta.get('cfa_str','')).squeeze()).lower()
        shifts={'rggb':(0,0),'grbg':(0,1),'gbrg':(1,0),'bggr':(1,1)}
        if cfa not in shifts: raise ValueError(f'{src.name}: unsupported SIDD CFA pattern {cfa!r}')
        cfa_shift=shifts[cfa]
        m=raw.astype(np.float32)[cfa_shift[0]:,cfa_shift[1]:]
        neutral=np.asarray(meta.get('AsShotNeutral',[]),dtype=np.float32).reshape(-1)
        if neutral.size!=3 or not np.isfinite(neutral).all() or np.any(neutral<=0): raise ValueError(f'{src.name}: invalid AsShotNeutral metadata')
        gains=[float(neutral[1]/neutral[0]),1.0,1.0,float(neutral[1]/neutral[2])]
        m=m.copy()
        # SIDD-CC is already black-subtracted and white-level normalized, but
        # deliberately retains sub-black noise (and occasional values > 1).
        # Preserve those measurements through white balance; clipping here
        # would erase the very noise statistics the RAW model must learn.
        for gain,(y,x) in zip(gains,[(0,0),(0,1),(1,0),(1,1)]): m[y::2,x::2]*=gain
    elif src.suffix.lower() in RAW_EXT|{'.npy'}:
        m,meta=decode_bayer(src.read_bytes(),src.name);m=white_balance(m,meta['gains'],clip=not meta.get('preserve_bayer_range',False))
    else: m=mosaic_rgb(Image.open(src).convert('RGB'))
    t=np.asarray(Image.open(tgt).convert('RGB'),dtype=np.float32)/255
    if cfa_shift!=(0,0): t=t[cfa_shift[0]:,cfa_shift[1]:]
    h,w=m.shape;h-=h%2;w-=w%2;m=m[:h,:w];t=t[:h,:w]
    if m.shape!=t.shape[:2]: raise ValueError(f'Pair dimensions differ: {src.name} and {tgt.name}')
    return m,t

def sample_crop_positions(h,w,patch,rng,count):
    """Sample distinct even-aligned Bayer crop origins where possible."""
    max_y=(h-patch)//2;max_x=(w-patch)//2
    total=(max_y+1)*(max_x+1)
    if count>total: raise ValueError(f'Requested {count} distinct crops, but image supports only {total}')
    positions=set()
    while len(positions)<count:
        positions.add((2*rng.randrange(max_y+1),2*rng.randrange(max_x+1)))
    return sorted(positions)

def validation_crop_positions(h,w,patch,count):
    """Deterministic center/corner Bayer crops for stable checkpoint selection."""
    max_y=h-patch;max_x=w-patch
    candidates=[
        (max_y//2//2*2,max_x//2//2*2),
        (0,0),(0,max_x//2*2),(max_y//2*2,0),(max_y//2*2,max_x//2*2),
    ]
    unique=[]
    for y,x in candidates:
        pos=(min(y,max_y//2*2),min(x,max_x//2*2))
        if pos not in unique: unique.append(pos)
    if count>len(unique): raise ValueError(f'At most {len(unique)} distinct validation positions are available')
    return unique[:count]

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--validation-data',help='Optional scene-disjoint Bayer validation split used to select the best checkpoint');p.add_argument('--init-checkpoint',help='Optional bayer-v1 checkpoint to fine-tune');p.add_argument('--epochs',type=int,default=5);p.add_argument('--patch',type=int,default=128);p.add_argument('--crops-per-image',type=int,default=1,help='Distinct random Bayer-aligned crops per training pair per epoch');p.add_argument('--validation-crops',type=int,default=1,help='Fixed center/corner crops averaged per validation pair');p.add_argument('--teacher');p.add_argument('--teacher-format',choices=['bayer-v1','torchscript','metaisp2023'],default='bayer-v1',help='TorchScript teacher contract: NCHW packed RGGB [0,1] -> NCHW RGB [0,1] at 2x spatial size');p.add_argument('--teacher-neutral',nargs=3,type=float,help='Teacher camera AsShotNeutral R G B');p.add_argument('--teacher-device',type=int,choices=[0,1,2],help='Teacher target style: 0 Pixel, 1 Samsung, 2 iPhone');p.add_argument('--out',default='models/mod_isp.pt');p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if args.patch<4 or args.patch%4 or args.epochs<1 or args.crops_per_image<1 or not 1<=args.validation_crops<=5: p.error('patch must be a positive multiple of 4; epochs/crops must be positive; validation-crops must be 1..5')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    files=pairs(Path(args.data));validation_files=pairs(Path(args.validation_data)) if args.validation_data else []
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'); model=ModISP();init_steps=0
    if args.init_checkpoint:
        initial,initial_meta=load_model(args.init_checkpoint);model.load_state_dict(initial.state_dict())
        init_steps=int(initial_meta.get('trained_steps',0))
    model=model.to(device);optim=torch.optim.Adam(model.parameters(),lr=1e-4)
    teacher=None
    if args.teacher:
        if args.teacher_format=='torchscript': teacher=torch.jit.load(args.teacher,map_location=device)
        elif args.teacher_format=='metaisp2023':
            from metaisp2023_adapter import MetaISP2023
            if args.teacher_neutral is None or args.teacher_device is None: p.error('metaisp2023 teacher requires --teacher-neutral R G B and --teacher-device 0/1/2')
            teacher=MetaISP2023(args.teacher, neutral=args.teacher_neutral, target_device=args.teacher_device)
        else: teacher,_=load_model(args.teacher)
        teacher.to(device).eval()
    steps=0; history=[]; started=time.time();best_validation=float('inf');best_epoch=args.epochs;best_state=None
    for epoch in range(args.epochs):
        random.shuffle(files);total=0
        for src,tgt in files:
            m,t=read_pair(src,tgt);h,w=m.shape
            if min(h,w)<args.patch: raise ValueError(f'{src.name} is smaller than patch size')
            positions=sample_crop_positions(h,w,args.patch,random,args.crops_per_image)
            for y,x in positions:
                a=pack(m[y:y+args.patch,x:x+args.patch]);b=t[y:y+args.patch,x:x+args.patch]
                inp=torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0).to(device);target=torch.from_numpy(b.transpose(2,0,1)).unsqueeze(0).to(device)
                out=model(inp);loss=F.l1_loss(out,target)
                if teacher:
                    with torch.no_grad(): soft=teacher(inp)
                    if not isinstance(soft,torch.Tensor) or soft.shape!=target.shape or not torch.isfinite(soft).all(): raise ValueError('Teacher must return finite NCHW RGB with same shape as target. Adapt its RAW packing and color domain explicitly.')
                    loss=loss+0.2*F.l1_loss(out,soft.clamp(0,1))
                optim.zero_grad();loss.backward();optim.step();total+=loss.item();steps+=1
        row={'epoch':epoch+1,'train_l1':float(total/(len(files)*args.crops_per_image)),'optimizer_steps':steps}
        if validation_files:
            model.eval();validation_total=0
            with torch.inference_mode():
                for src,tgt in validation_files:
                    m,t=read_pair(src,tgt);h,w=m.shape
                    positions=validation_crop_positions(h,w,args.patch,args.validation_crops)
                    for y,x in positions:
                        a=pack(m[y:y+args.patch,x:x+args.patch]);b=t[y:y+args.patch,x:x+args.patch]
                        inp=torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0).to(device)
                        target=torch.from_numpy(b.transpose(2,0,1)).unsqueeze(0).to(device)
                        validation_total+=F.l1_loss(model(inp),target).item()
            row['validation_l1']=float(validation_total/(len(validation_files)*args.validation_crops))
            if row['validation_l1']<best_validation:
                best_validation=row['validation_l1'];best_epoch=epoch+1
                best_state=copy.deepcopy(model.state_dict())
            model.train()
        history.append(row)
        print(f"epoch {row['epoch']}/{args.epochs} train_l1={row['train_l1']:.5f}"+
              (f" validation_l1={row['validation_l1']:.5f}" if 'validation_l1' in row else ''),flush=True)
    if best_state is not None: model.load_state_dict(best_state)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    parameters=sum(p.numel() for p in model.parameters());device_name=str(device)
    checkpoint={'architecture':'bayer-v1','model':model.cpu().state_dict(),'trained_steps':steps,
      'dataset':str(Path(args.data).resolve()),'seed':args.seed,'teacher':args.teacher or '',
      'teacher_format':args.teacher_format if args.teacher else '', 'teacher_neutral':args.teacher_neutral or [],
      'teacher_device':args.teacher_device if args.teacher_device is not None else -1,
      'epochs':args.epochs,'best_epoch':best_epoch,'validation_l1':best_validation if validation_files else None,
      'validation_dataset':str(Path(args.validation_data).resolve()) if args.validation_data else None,
      'init_checkpoint':str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
      'init_trained_steps':init_steps,
      'patch_size':args.patch,'train_pairs':len(files),'device':device_name}
    torch.save(checkpoint,args.out)
    report={'task':'Bayer mosaic to sRGB reconstruction','dataset':str(Path(args.data).resolve()),
      'input_pairs':sorted(src.name for src,_ in files),'train_pairs':len(files),'epochs':args.epochs,
      'validation_dataset':str(Path(args.validation_data).resolve()) if args.validation_data else None,
      'init_checkpoint':str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
      'init_trained_steps':init_steps,
      'validation_pairs':len(validation_files),'best_epoch':best_epoch,
      'best_validation_l1':best_validation if validation_files else None,
      'patch_size':args.patch,'crops_per_image_per_epoch':args.crops_per_image,
      'validation_crops_per_pair':args.validation_crops,'seed':args.seed,'device':device_name,'optimizer_steps':steps,
      'parameters':parameters,'checkpoint_bytes':Path(args.out).stat().st_size,
      'teacher':args.teacher or None,'teacher_format':args.teacher_format if args.teacher else None,
      'history':history,'elapsed_seconds':time.time()-started}
    Path(args.out).with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(f'Saved {args.out} and {Path(args.out).with_suffix(".json")}',flush=True)
if __name__=='__main__':main()
