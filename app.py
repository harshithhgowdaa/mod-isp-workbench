import base64, io, os, platform, sys, time, tracemalloc
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from isp import decode_bayer, white_balance, demosaic, mosaic_rgb, pack, rgb_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
ROOT = Path(__file__).parent
CHECKPOINT = ROOT / 'models' / 'mod_isp.pt'
RGB_CHECKPOINT = ROOT / 'models' / 'sidd_srgb_nafnet_lite.pt'
RGB_MODEL_CACHE = {}
BAYER_MODEL_CACHE = {}
LPIPS_MODEL = None
RAW_EXT = {'.dng','.nef','.cr2','.cr3','.arw','.raf','.rw2','.npy'}

def samsung_repo_path():
    configured=os.environ.get('MOD_ISP_SAMSUNG_REPO')
    candidates=[]
    if configured: candidates.append(Path(configured).expanduser())
    candidates.extend([ROOT.parent.parent / 'work' / 'modular_neural_isp',
                       ROOT / 'third_party' / 'modular_neural_isp'])
    return next((path for path in candidates if path.exists()),None)

def model_status():
    try:
        import torch
        if not CHECKPOINT.exists(): return {'available':False,'message':'No trained checkpoint found. Classical processing is ready.'}
        from model import load_model
        model, metadata = load_model(CHECKPOINT)
        return {'available':True,'message':'Trained checkpoint loaded','parameters':sum(p.numel() for p in model.parameters()),'size_kb':round(CHECKPOINT.stat().st_size/1024,1),'metadata':metadata}
    except Exception as e: return {'available':False,'message':f'Model unavailable: {str(e)}'}

def rgb_model_status():
    if not RGB_CHECKPOINT.exists():
        return {'available':False,'message':'No trained sRGB denoiser found.'}
    try:
        import torch
        repo = samsung_repo_path()
        if repo is None: raise FileNotFoundError('Samsung NAFNet model definition is missing.')
        checkpoint = torch.load(RGB_CHECKPOINT, map_location='cpu', weights_only=True)
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model,_=load_rgb_denoiser(device)
        return {'available':True,'message':'SIDD-CC sRGB denoiser available',
                'parameters':sum(p.numel() for p in model.parameters()),
                'size_kb':round(RGB_CHECKPOINT.stat().st_size/1024,1),
                'best_epoch':checkpoint.get('best_epoch'), 'dataset':'SIDD-CC SRGB'}
    except Exception as e: return {'available':False,'message':f'sRGB model unavailable: {str(e)}'}

def decode(file):
    ext=Path(file.filename or '').suffix.lower()
    if ext in RAW_EXT:
        m, meta=decode_bayer(file.read(),file.filename)
        m=white_balance(m,meta['gains'],clip=not meta.get('preserve_bayer_range',False))
        h,w=m.shape; m=m[:h-h%2,:w-w%2]
        if min(m.shape)<2: raise ValueError('Image must be at least 2 by 2 pixels.')
        preview=meta.get('raw_render')
        if preview is None:
            preview=Image.fromarray(np.uint8(np.clip(linear_to_srgb(demosaic(m)),0,1)*255))
        meta['raw_ccm_available']=bool(meta.get('camera_matrix_available',False))
        return preview,meta['kind'],m,'bayer',meta
    if ext not in {'.png','.jpg','.jpeg','.webp','.tif','.tiff'}: raise ValueError('Use a supported RAW, NPY, JPEG, PNG, WebP, or TIFF image.')
    try:
        im=ImageOps.exif_transpose(Image.open(file.stream)); im.load(); im=im.convert('RGB')
        if min(im.size)<2: raise ValueError('Image must be at least 2 by 2 pixels.')
        original_size=im.size
        if im.width*im.height>MAX_IMAGE_PIXELS:
            scale=(MAX_IMAGE_PIXELS/(im.width*im.height))**0.5
            size=(max(2,int(im.width*scale)//2*2),max(2,int(im.height*scale)//2*2))
            im=im.resize(size,Image.Resampling.LANCZOS)
        kind='Rendered RGB / sRGB image · RGB-compatible enhancement path'
        if im.size!=original_size: kind+=f' · resized from {original_size[0]}×{original_size[1]} to {im.width}×{im.height} (16 MP processing cap)'
        return im,kind,None,'rgb',{}
    except Exception: raise ValueError('The image could not be read. Try another file.')

def encode(im):
    b=io.BytesIO(); im.convert('RGB').save(b,format='JPEG',quality=92,optimize=True); return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()

def srgb_to_linear(a):
    return np.where(a <= 0.04045, a/12.92, ((a+0.055)/1.055)**2.4)

def linear_to_srgb(a):
    a=np.clip(a,0,1)
    return np.where(a <= 0.0031308, 12.92*a, 1.055*np.power(a,1/2.4)-0.055)

def exposure_gain(im, mode, strength):
    """Estimate exposure from rendered RGB luminance, without inventing sensor RAW."""
    a=np.asarray(im,dtype=np.float32)/255
    lin=srgb_to_linear(a)
    y=lin @ np.array([0.2126,0.7152,0.0722],dtype=np.float32)
    measured=float(np.percentile(y,55))
    target=0.050 if mode=='human' else 0.030
    ev=float(np.clip(np.log2(target/max(measured,1e-5)),-1.0,4.5))*strength
    return 2.0**ev

def finish_rgb(im, mode, strength=1.0, style=None, apply_exposure=True, sharpen=True):
    """sRGB-safe color/tone finishing for already rendered RGB inputs."""
    if not apply_exposure:
        out=im.convert('RGB')
        if style is not None:
            from cnilut import apply_cnilut
            out=apply_cnilut(out,style)
        return out.filter(ImageFilter.UnsharpMask(radius=1.15,percent=int((48 if mode=='human' else 30)*strength),threshold=3)) if sharpen else out
    a=np.asarray(im,dtype=np.float32)/255
    lin=srgb_to_linear(a)
    if apply_exposure:
        gain=exposure_gain(im,mode,strength)
        exposed=np.maximum(lin*gain,0)
        y=exposed @ np.array([0.2126,0.7152,0.0722],dtype=np.float32)
        # White-point Reinhard shoulder lifts dark detail while rolling highlights smoothly.
        white=max(float(np.percentile(y,99.7)),0.12)
        yd=np.clip(y*(1+y/(white*white))/(1+y),0,1)
        ratio=yd/np.maximum(y,1e-6)
        lin=np.clip(exposed*ratio[...,None],0,1)
    out=Image.fromarray(np.uint8(np.clip(linear_to_srgb(lin),0,1)*255))
    if style is not None:
        from cnilut import apply_cnilut
        out=apply_cnilut(out,style)
    return out.filter(ImageFilter.UnsharpMask(radius=1.15,percent=int((48 if mode=='human' else 30)*strength),threshold=3)) if sharpen else out

def classical_rgb(im, mode, strength, style=None):
    """Classical degraded-RGB path: adaptive exposure, conservative denoise, tone and detail."""
    gain=exposure_gain(im,mode,strength)
    a=np.asarray(im,dtype=np.float32)/255
    lin=srgb_to_linear(a)*gain
    y=lin @ np.array([0.2126,0.7152,0.0722],dtype=np.float32)
    white=max(float(np.percentile(y,99.7)),0.12)
    yd=np.clip(y*(1+y/(white*white))/(1+y),0,1)
    lin=np.clip(lin*(yd/np.maximum(y,1e-6))[...,None],0,1)
    base=Image.fromarray(np.uint8(np.clip(linear_to_srgb(lin),0,1)*255))
    # A small bilateral filter smooths chroma/noise while retaining high-contrast edges.
    try:
        import cv2
        arr=np.asarray(base)
        den=cv2.bilateralFilter(arr,5,18,2.0)
        base=Image.blend(base,Image.fromarray(den),min(0.28,0.12*strength))
    except ImportError:
        base=Image.blend(base,base.filter(ImageFilter.GaussianBlur(0.45)),min(0.14,0.06*strength))
    if style is not None:
        from cnilut import apply_cnilut
        base=apply_cnilut(base,style)
    return base.filter(ImageFilter.UnsharpMask(radius=1.15,percent=int((48 if mode=='human' else 30)*strength),threshold=3))

def classical_bayer(mosaic, mode, strength, style=None):
    # Bayer demosaic returns linear RGB; render once, then perform the perceptual stages.
    rgb=np.clip(linear_to_srgb(demosaic(mosaic)),0,1)
    return finish_rgb(Image.fromarray(np.uint8(rgb*255)),mode,strength,style,apply_exposure=True)

def load_rgb_denoiser(device):
    import torch
    key=str(device)
    if key in RGB_MODEL_CACHE: return RGB_MODEL_CACHE[key],torch
    repo=samsung_repo_path()
    if repo is None: raise FileNotFoundError('Samsung NAFNet model definition is missing.')
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    from denoising.nafnet_arch import NAFNet
    checkpoint=torch.load(RGB_CHECKPOINT,map_location='cpu',weights_only=True)
    model=NAFNet(**checkpoint['config']);model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device).eval()
    RGB_MODEL_CACHE[key]=model
    return model,torch

def denoise_rgb_tiles(image, hybrid=False, route_fraction=0.5, tile=512, overlap=64,
                      precomputed=None, return_cache=False, batch_size=8):
    """Run the SIDD sRGB denoiser on RGB tiles, optionally reusing learned tiles."""
    import torch
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=None if precomputed is not None else load_rgb_denoiser(device)[0]
    src=np.asarray(image,dtype=np.float32)/255
    h,w=src.shape[:2];step=tile-overlap
    coords=[(y,x,min(y+tile,h),min(x+tile,w)) for y in list(range(0,max(h-tile,0)+1,step))+([max(h-tile,0)] if h>tile and (h-tile)%step else []) for x in list(range(0,max(w-tile,0)+1,step))+([max(w-tile,0)] if w>tile and (w-tile)%step else [])]
    coords=list(dict.fromkeys(coords))
    if not coords: coords=[(0,0,h,w)]
    selected=set(range(len(coords)))
    if hybrid:
        # Robust high-pass MAD is a useful routing heuristic for RGB sensor noise;
        # it is deliberately not labeled as the calibrated Bayer gate.
        from scipy.ndimage import gaussian_filter
        scores=[]
        for i,(y,x,y1,x1) in enumerate(coords):
            patch=src[y:y1,x:x1].mean(axis=2)
            residual=patch-gaussian_filter(patch,sigma=1.0)
            scores.append((float(np.median(np.abs(residual))),i))
        count=max(1,int(np.ceil(len(coords)*float(route_fraction))))
        selected={i for _,i in sorted(scores,key=lambda item:(-item[0],item[1]))[:count]}
    acc=np.zeros_like(src);weight_sum=np.zeros((h,w,1),np.float32);routed=0
    routed_coords=[(i,*coords[i]) for i in sorted(selected)]
    tile_cache={}
    for start in range(0,len(routed_coords),batch_size):
        group=routed_coords[start:start+batch_size]
        if precomputed is not None:
            preds=[precomputed[(y,x)] for _,y,x,_,_ in group]
        else:
            patches=[src[y:y1,x:x1] for _,y,x,y1,x1 in group]
            max_h=max(p.shape[0] for p in patches);max_w=max(p.shape[1] for p in patches)
            batch=[]
            for patch in patches:
                ph,pw=patch.shape[:2]
                # All coordinates share dimensions except at image edges; replicate those borders.
                pad_h=max_h-ph;pad_w=max_w-pw
                padded=np.pad(patch,((0,pad_h),(0,pad_w),(0,0)),mode='reflect') if pad_h or pad_w else patch
                batch.append(padded.transpose(2,0,1))
            ten=torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(device)
            pad_h=(-ten.shape[-2])%4;pad_w=(-ten.shape[-1])%4
            if pad_h or pad_w: ten=torch.nn.functional.pad(ten,(0,pad_w,0,pad_h),mode='reflect')
            with torch.inference_mode():
                preds=model(ten).clamp(0,1).cpu().numpy().transpose(0,2,3,1)
        for pred,(i,y,x,y1,x1) in zip(preds,group):
            ph,pw=y1-y,x1-x;pred=pred[:ph,:pw]
            tile_cache[(y,x)]=pred.astype(np.float16,copy=True) if return_cache else pred
            # Smooth overlap seams and keep the outer image boundary fully weighted.
            wy=np.ones(ph,np.float32);wx=np.ones(pw,np.float32);edge=min(overlap//2,ph//4,pw//4)
            if edge>0:
                ramp=np.linspace(0,1,edge,endpoint=False,dtype=np.float32)
                ramp=0.5-0.5*np.cos(np.pi*ramp)
                if y>0: wy[:edge]=ramp
                if y1<h: wy[-edge:]=ramp[::-1]
                if x>0: wx[:edge]=ramp
                if x1<w: wx[-edge:]=ramp[::-1]
            weight=(wy[:,None]*wx[None,:])[...,None]
            acc[y:y1,x:x1]+=pred*weight;weight_sum[y:y1,x:x1]+=weight;routed+=1
    pred=acc/np.maximum(weight_sum,1e-8)
    if hybrid:
        coverage=np.clip(weight_sum,0,1)
        out=src*(1-coverage)+pred*coverage
    else: out=pred
    result=(Image.fromarray(np.uint8(np.clip(out,0,1)*255)),round(100*routed/max(len(coords),1),1))
    return (*result,tile_cache) if return_cache else result

def blend_rgb_paths(classical, learned_output, source, route_fraction=0.5, tile=512, overlap=64):
    """Compose a true hybrid view from the measured classical and learned RGB outputs."""
    from scipy.ndimage import gaussian_filter
    src=np.asarray(source,dtype=np.float32)/255
    base=np.asarray(classical,dtype=np.float32)/255
    neural=np.asarray(learned_output,dtype=np.float32)/255
    if src.shape!=base.shape or src.shape!=neural.shape:
        raise ValueError('Classical, learned, and source RGB dimensions must match.')
    h,w=src.shape[:2];step=tile-overlap
    ys=list(range(0,max(h-tile,0)+1,step))+([max(h-tile,0)] if h>tile and (h-tile)%step else [])
    xs=list(range(0,max(w-tile,0)+1,step))+([max(w-tile,0)] if w>tile and (w-tile)%step else [])
    coords=list(dict.fromkeys((y,x,min(y+tile,h),min(x+tile,w)) for y in ys for x in xs)) or [(0,0,h,w)]
    scores=[]
    for i,(y,x,y1,x1) in enumerate(coords):
        patch=src[y:y1,x:x1].mean(axis=2)
        residual=patch-gaussian_filter(patch,sigma=1.0)
        scores.append((float(np.median(np.abs(residual))),i))
    count=max(1,int(np.ceil(len(coords)*float(route_fraction)))) if route_fraction>0 else 0
    selected=[i for _,i in sorted(scores,key=lambda item:(-item[0],item[1]))[:count]]
    # Blur routing at a coarse spatial scale. A full 12 MP Gaussian kernel at
    # 100 px sigma dominates latency even though tile boundaries are 512 px.
    scale=8
    mask=np.zeros(((h+scale-1)//scale,(w+scale-1)//scale),np.float32)
    for i in selected:
        y,x,y1,x1=coords[i]
        mask[y//scale:(y1+scale-1)//scale,x//scale:(x1+scale-1)//scale]=1
    alpha=gaussian_filter(mask,sigma=max(tile/5/scale,1))
    alpha=np.repeat(np.repeat(alpha,scale,axis=0),scale,axis=1)[:h,:w]
    alpha=np.clip(alpha,0,1)[...,None]
    return rgb_image(base*(1-alpha)+neural*alpha),round(100*len(selected)/max(len(coords),1),1)

def blend_bayer_paths(classical, learned_output, mosaic, tile=56, gate_config=None):
    """Apply the calibrated Bayer gate to compose the two completed ISP outputs."""
    from scipy.ndimage import gaussian_filter
    import json
    if gate_config is None:
        try: gate=json.loads((ROOT/'models'/'gate.json').read_text())
        except (OSError,ValueError): gate={'mode':'relative_rank','route_fraction':0.5}
    else: gate=gate_config
    h,w=mosaic.shape;blocks=[]
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            block=mosaic[y:min(y+tile,h),x:min(x+tile,w)]
            score=float(block.var())/(float(block.mean())**2+1e-6)
            blocks.append((score,y,x))
    if gate.get('mode')=='relative_rank':
        fraction=float(gate.get('route_fraction',0.5))
        if not 0<=fraction<=1: raise ValueError('Invalid calibrated Bayer routing fraction.')
        count=int(np.ceil(len(blocks)*fraction))
        chosen={(y,x) for _,y,x in sorted(blocks,key=lambda q:(-q[0],q[1],q[2]))[:count]}
    else:
        threshold=float(gate.get('variance_threshold',0.006))
        chosen={(y,x) for score,y,x in blocks if score>threshold}
    mask=np.zeros((h,w),np.float32)
    for y,x in chosen: mask[y:min(y+tile,h),x:min(x+tile,w)]=1
    alpha=np.clip(gaussian_filter(mask,sigma=max(tile/5,1)),0,1)[...,None]
    base=np.asarray(classical,dtype=np.float32)/255
    neural=np.asarray(learned_output,dtype=np.float32)/255
    return rgb_image(base*(1-alpha)+neural*alpha),round(100*len(chosen)/max(len(blocks),1),1)

def learned(mosaic, hybrid=False, threshold=None, route_fraction=None, gate_config=None,
            base_output=None, mode='human', strength=1.0, style=None):
    import torch
    from model import load_model
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cache_key=(str(CHECKPOINT.resolve()),CHECKPOINT.stat().st_mtime_ns,str(device))
    if cache_key not in BAYER_MODEL_CACHE:
        model,_=load_model(CHECKPOINT)
        model.to(device).eval()
        BAYER_MODEL_CACHE.clear()
        BAYER_MODEL_CACHE[cache_key]=model
    model=BAYER_MODEL_CACHE[cache_key]
    if hybrid and gate_config is None and threshold is None and route_fraction is None:
        try:
            import json
            config=json.loads((ROOT / 'models' / 'gate.json').read_text())
            if config.get('mode')=='relative_rank': route_fraction=float(config['route_fraction'])
            else: threshold=float(config['variance_threshold'])
        except (OSError, ValueError, KeyError): threshold=0.006
    h,w=mosaic.shape
    # The restoration network is trained against rendered sRGB targets. Keep
    # the classical side of a Hybrid blend in the same display-referred space.
    if hybrid:
        base=np.asarray(base_output if base_output is not None else classical_bayer(mosaic,mode,strength,style),dtype=np.float32)/255
        if gate_config is not None and 'classical_affine_srgb' in gate_config:
            affine=np.asarray(gate_config['classical_affine_srgb'],dtype=np.float32)
            if affine.shape!=(4,3) or not np.isfinite(affine).all():
                raise ValueError('Gate classical_affine_srgb must be a finite 4x3 matrix.')
            design=np.concatenate((base,np.ones((*base.shape[:2],1),dtype=np.float32)),axis=2)
            base=np.clip(design @ affine,0,1)
    else:
        # Full Learned inference overwrites every location; a classical
        # demosaic here was redundant and material on megapixel RAW frames.
        base=np.empty((h,w,3),dtype=np.float32)
    out=base.copy(); tile=56; halo=8; routed=0; total=0
    selected=None
    if hybrid and gate_config is not None and gate_config.get('mode') in (
            'raw_feature_ridge', 'raw_feature_ridge_rank',
            'raw_feature_extra_trees', 'raw_feature_extra_trees_rank'):
        # Keep only coordinate positions selected by the frozen RAW-feature policy.
        from tile_quality_gate import predict_benefit
        predictions=predict_benefit(mosaic,gate_config,tile)
        if gate_config.get('mode') in ('raw_feature_ridge_rank', 'raw_feature_extra_trees_rank'):
            fraction=float(gate_config.get('route_fraction',1.0))
            if not 0<=fraction<=1: raise ValueError('Invalid calibrated Bayer routing fraction.')
            count=int(np.ceil(len(predictions)*fraction))
            selected={(yy,xx) for _,yy,xx in sorted(predictions,key=lambda q:(-q[0],q[1],q[2]))[:count]}
        else:
            selected={(yy,xx) for score,yy,xx in predictions if score >= float(gate_config['threshold_db'])}
    elif hybrid and gate_config is not None and gate_config.get('mode') == 'relative_rank':
        route_fraction=float(gate_config.get('route_fraction',1.0))
    if hybrid and route_fraction is not None:
        if not 0<=route_fraction<=1: raise ValueError('Routing fraction must be between 0 and 1.')
        blocks=[]
        for yy in range(0,h,tile):
            for xx in range(0,w,tile):
                block=mosaic[yy:min(yy+tile,h),xx:min(xx+tile,w)]
                # Rank contrast relative to illumination. This is a compute budget, not a noise estimate.
                score=float(block.var())/(float(block.mean())**2+1e-6)
                blocks.append((score,yy,xx))
        count=int(np.ceil(len(blocks)*route_fraction))
        selected={(yy,xx) for _,yy,xx in sorted(blocks,key=lambda item:(-item[0],item[1],item[2]))[:count]}
    elif hybrid and selected is None:
        selected=set()
        for yy in range(0,h,tile):
            for xx in range(0,w,tile):
                block=mosaic[yy:min(yy+tile,h),xx:min(xx+tile,w)]
                if float(block.var())>threshold: selected.add((yy,xx))
    # Group same-shaped tiles so the compact model runs in batches instead of
    # thousands of tiny framework calls. Equal-shaped edge tiles stay grouped;
    # the exact input crop and halo are unchanged from the scalar implementation.
    groups={}
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            y1=min(y+tile,h);x1=min(x+tile,w);total+=1
            if hybrid and selected is not None and (y,x) not in selected: continue
            if hybrid and (y,x) not in selected: continue
            routed+=1
            y0=max(0,y-halo);x0=max(0,x-halo);ey=min(h,y1+halo);ex=min(w,x1+halo)
            y0-=y0%2;x0-=x0%2;ey-=ey%2;ex-=ex%2
            packed_shape=((ey-y0)//2,(ex-x0)//2)
            groups.setdefault(packed_shape,[]).append((y,x,y0,x0,y1,x1))
    batch_size=256 if device.type=='mps' else 128
    with torch.inference_mode():
        for group in groups.values():
            for start in range(0,len(group),batch_size):
                chunk=group[start:start+batch_size]
                batch=np.stack([pack(mosaic[y0: min(y1+halo,h)-(min(y1+halo,h)%2),
                                         x0: min(x1+halo,w)-(min(x1+halo,w)%2)]).transpose(2,0,1)
                                for y,x,y0,x0,y1,x1 in chunk]).astype(np.float32,copy=False)
                tensor=torch.from_numpy(np.ascontiguousarray(batch)).to(device)
                preds=model(tensor).clamp(0,1).cpu().numpy().transpose(0,2,3,1)
                for pred,(y,x,y0,x0,y1,x1) in zip(preds,chunk):
                    if hybrid:
                        # Finish the neural tile while its model halo is still
                        # present. This matches the full Learned path without
                        # sharpening or styling the classical-only regions.
                        pred=np.asarray(finish_rgb(rgb_image(pred),mode,strength,style,
                                                   apply_exposure=False),dtype=np.float32)/255
                    yy=y-y0;xx=x-x0
                    piece=pred[yy:yy+y1-y,xx:xx+x1-x]
                    out[y:y1,x:x1]=piece
    if hybrid:
        scale=8
        small=np.zeros(((h+scale-1)//scale,(w+scale-1)//scale),np.float32)
        for y,x in selected:
            small[y//scale:min((y+tile+scale-1)//scale,small.shape[0]),
                  x//scale:min((x+tile+scale-1)//scale,small.shape[1])]=1
        if gate_config is not None and 'feather_pixels' in gate_config:
            feather=int(gate_config['feather_pixels'])
            if feather < 0: raise ValueError('Gate feather_pixels must be non-negative.')
            if feather == 0:
                alpha=np.repeat(np.repeat(small,scale,axis=0),scale,axis=1)[:h,:w]
            else:
                from scipy.ndimage import distance_transform_edt
                full_mask=np.repeat(np.repeat(small,scale,axis=0),scale,axis=1)[:h,:w] >= .5
                if full_mask.all(): alpha=full_mask.astype(np.float32)
                elif not full_mask.any(): alpha=full_mask.astype(np.float32)
                else:
                    signed=distance_transform_edt(full_mask)-distance_transform_edt(~full_mask)
                    alpha=np.clip(.5+signed/(2*feather),0,1).astype(np.float32)
        else:
            from scipy.ndimage import gaussian_filter
            small=gaussian_filter(small,sigma=max(tile/5/scale,1))
            alpha=np.asarray(Image.fromarray(small).resize((w,h),Image.Resampling.BILINEAR),dtype=np.float32)
        alpha=np.clip(alpha,0,1)[...,None]
        np.multiply(out,alpha,out=out)
        out+=base*(1-alpha)
    return rgb_image(out),round(100*routed/max(total,1),1)

def metrics(im, ref):
    global LPIPS_MODEL
    if im.size != ref.size: raise ValueError('Reference dimensions must match the input image.')
    # Keep full-resolution metrics bounded in memory. PSNR is accumulated in
    # row strips; SSIM and color error are computed by strips with context.
    a=np.asarray(im,dtype=np.uint8); b=np.asarray(ref,dtype=np.uint8)
    h,w=a.shape[:2]
    squared_error=0.0
    for y in range(0,h,256):
        delta=a[y:y+256].astype(np.float32)-b[y:y+256].astype(np.float32)
        squared_error+=float(np.sum(delta*delta,dtype=np.float64))
    mse=squared_error/(h*w*3*255.0*255.0)
    result={'psnr':round(float(-10*np.log10(mse)),2) if mse else '∞'}
    try:
        from skimage.metrics import structural_similarity
        from skimage.color import rgb2lab, deltaE_ciede2000
        if min(h,w)<7:
            result['ssim']=None
        else:
            # Structural similarity uses a 7x7 window by default. Three rows
            # of halo on either side make strip means match the full-frame
            # valid-window mean while limiting temporary arrays.
            pad=3; total=0.0; count=0
            for y in range(0,h,256):
                y1=min(y+256,h); ext0=max(0,y-pad); ext1=min(h,y1+pad)
                aa=a[ext0:ext1].astype(np.float32)/255
                bb=b[ext0:ext1].astype(np.float32)/255
                _,score=structural_similarity(aa,bb,channel_axis=2,data_range=1,full=True)
                keep0=max(y,pad)-ext0
                keep1=min(y1,h-pad)-ext0
                if keep1>keep0:
                    valid=score[keep0:keep1,pad:-pad]
                    total+=float(valid.sum(dtype=np.float64));count+=valid.size
            result['ssim']=round(total/count,4) if count else None
        delta_total=0.0;pixel_count=0
        for y in range(0,h,256):
            aa=a[y:y+256].astype(np.float32)/255
            bb=b[y:y+256].astype(np.float32)/255
            delta=deltaE_ciede2000(rgb2lab(aa),rgb2lab(bb))
            delta_total+=float(delta.sum(dtype=np.float64));pixel_count+=delta.size
        result['delta_e']=round(delta_total/pixel_count,2) if pixel_count else None
    except ImportError: result.update(ssim=None,delta_e=None)
    try:
        import torch, lpips
        if LPIPS_MODEL is None: LPIPS_MODEL=lpips.LPIPS(net='squeeze',verbose=False).eval()
        tile=256; batch_size=4; chunks=[]
        for y in range(0,h,tile):
            for x in range(0,w,tile):
                y1=min(y+tile,h);x1=min(x+tile,w)
                aa=a[y:y1,x:x1].astype(np.float32)/127.5-1
                bb=b[y:y1,x:x1].astype(np.float32)/127.5-1
                ph=tile-aa.shape[0];pw=tile-aa.shape[1]
                if ph or pw:
                    mode='reflect' if min(aa.shape[:2])>1 else 'edge'
                    aa=np.pad(aa,((0,ph),(0,pw),(0,0)),mode=mode)
                    bb=np.pad(bb,((0,ph),(0,pw),(0,0)),mode=mode)
                chunks.append((aa.transpose(2,0,1),bb.transpose(2,0,1),(y1-y)*(x1-x)))
        total=0.0;weight=0
        with torch.inference_mode():
            for start in range(0,len(chunks),batch_size):
                group=chunks[start:start+batch_size]
                x=torch.from_numpy(np.ascontiguousarray(np.stack([v[0] for v in group])))
                y=torch.from_numpy(np.ascontiguousarray(np.stack([v[1] for v in group])))
                values=LPIPS_MODEL(x,y).reshape(-1).cpu().numpy()
                for value,item in zip(values,group):total+=float(value)*item[2];weight+=item[2]
        result['lpips']=round(total/weight,4) if weight else None
        result['lpips_method']='256px non-overlapping tiles; area-weighted mean'
    except Exception: result['lpips']=None
    return result

@app.get('/')
def index(): return render_template('showcase.html', local_runtime=True)

@app.get('/workbench')
def workbench(): return render_template('index.html')

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error='Upload exceeds the 30 MB limit. Choose a smaller file.'), 413

@app.get('/api/status')
def status(): return jsonify(bayer=model_status(),rgb=rgb_model_status())

def process_decoded(im, kind, mosaic, input_type, input_meta, reference=None,
                    mode='human', strength=1.0, style=None):
    if im.width*im.height>MAX_IMAGE_PIXELS:
        raise ValueError('This RAW/NPY image exceeds the 16 megapixel processing limit. Use a lower-resolution sensor export.')
    if reference is not None and reference.size != im.size:
        raise ValueError(f'Reference dimensions {reference.width}×{reference.height} do not match the processed image {im.width}×{im.height}.')
    camera_managed_raw=input_type=='bayer' and input_meta.get('raw_render') is not None
    tracemalloc.start(); start=time.perf_counter()
    result=(classical_rgb(im,mode,strength,style) if input_type=='rgb' or camera_managed_raw
            else classical_bayer(mosaic,mode,strength,style))
    elapsed=(time.perf_counter()-start)*1000; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    outputs={'classical':{'image':encode(result),'latency_ms':round(elapsed,1),'memory_mb':round(peak/1048576,2),'metrics':metrics(result,reference) if reference else None}}
    # Camera DNGs already have a camera-matched LibRaw ISP render. Applying
    # the generic Bayer student trained on other camera pipelines caused
    # strong color casts in the supplied iPhone SE example. Keep Bayer
    # reconstruction for normalized RGGB arrays, and denoise camera-managed
    # RAW renders in sRGB until camera-specific paired validation exists.
    status=rgb_model_status() if input_type=='rgb' or camera_managed_raw else model_status()
    if status['available']:
        use_rgb_model=input_type=='rgb' or camera_managed_raw
        if use_rgb_model:
            # Train-domain match: denoise the original rendered sRGB, then
            # apply display finishing. Hybrid reuses these same predictions.
            tracemalloc.start();start=time.perf_counter()
            denoised,_=denoise_rgb_tiles(im)
            learned_output=finish_rgb(denoised,mode,strength,style,apply_exposure=True)
            elapsed=(time.perf_counter()-start)*1000;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            outputs['learned']={'image':encode(learned_output),'latency_ms':round(elapsed,1),
                'memory_mb':round(peak/1048576,2),'metrics':metrics(learned_output,reference) if reference else None}
            tracemalloc.start();start=time.perf_counter()
            hybrid_output,routed=blend_rgb_paths(result,learned_output,im)
            elapsed=(time.perf_counter()-start)*1000;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            outputs['hybrid']={'image':encode(hybrid_output),'latency_ms':round(elapsed,1),
                'memory_mb':round(peak/1048576,2),'routed':routed,'reused_inference':True,
                'metrics':metrics(hybrid_output,reference) if reference else None}
        else:
            tracemalloc.start();start=time.perf_counter()
            bayer_output,_=learned(mosaic,hybrid=False)
            learned_output=finish_rgb(bayer_output,mode,strength,style,apply_exposure=False)
            elapsed=(time.perf_counter()-start)*1000;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            outputs['learned']={'image':encode(learned_output),'latency_ms':round(elapsed,1),
                'memory_mb':round(peak/1048576,2),'metrics':metrics(learned_output,reference) if reference else None}
            try:
                import json
                tile_gate=json.loads((ROOT/'models'/'gate_quality_tile_runtimeaware.json').read_text())
                if tile_gate.get('checkpoint') != str(CHECKPOINT.resolve()):
                    tile_gate=None
            except (OSError,ValueError): tile_gate=None
            tracemalloc.start();start=time.perf_counter()
            hybrid_output,routed=learned(mosaic,hybrid=True,gate_config=tile_gate,
                route_fraction=1.0 if tile_gate is None else None,base_output=result,
                mode=mode,strength=strength,style=style)
            elapsed=(time.perf_counter()-start)*1000;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            outputs['hybrid']={'image':encode(hybrid_output),'latency_ms':round(elapsed,1),
                'memory_mb':round(peak/1048576,2),'routed':routed,'reused_inference':False,
                'gate':({'name':'SIDD-CC supervised RAW-tile gate','status':'experimental · held-out PSNR budget not met',
                         'deployment_eligible':False,'validation_route_percent':tile_gate.get('validation_route_percent'),
                         'validation_tile_call_reduction_percent':100-float(tile_gate.get('validation_route_percent',100)),
                         'validation_psnr_delta_db':tile_gate.get('validation_mean_psnr_delta_db'),
                         'validation_psnr_budget_passed':tile_gate.get('validation_psnr_budget_passed')}
                        if tile_gate is not None else {'name':'All tiles','status':'fallback · no calibrated tile gate'}),
                'metrics':metrics(hybrid_output,reference) if reference else None}
    processing_note=('Camera Bayer mosaic → LibRaw camera-managed RGB; learned/hybrid paths denoise that sRGB render.' if camera_managed_raw else ('White-balanced RGGB mosaic → MOD-ISP learned RGB reconstruction' if input_type=='bayer' else 'Rendered sRGB → SIDD-CC sRGB denoiser'))
    return dict(input=encode(im),width=im.width,height=im.height,input_kind=kind,input_type=input_type,
                camera_managed_raw=bool(camera_managed_raw),raw_integrity=input_meta.get('raw_integrity'),
                outputs=outputs,model=status,processing_note=processing_note,
                environment=f'{platform.system()} {platform.machine()} · Python {platform.python_version()}')

@app.post('/api/process')
def process():
    try:
        if 'image' not in request.files: raise ValueError('Choose an image to process.')
        im, kind, mosaic, input_type,input_meta=decode(request.files['image'])
        style_raw=request.form.get('style','off')
        if style_raw not in ('off','0','1','2'): raise ValueError('Invalid CNILUT style.')
        style=None if style_raw=='off' else int(style_raw)
        mode=request.form.get('mode','human'); strength=float(request.form.get('strength','1')); strength=max(0,min(2,strength))
        reference=None
        if 'reference' in request.files and request.files['reference'].filename: reference,_,_,_,_=decode(request.files['reference'])
        result=process_decoded(im,kind,mosaic,input_type,input_meta,reference,mode,strength,style)
        return jsonify(result)
    except RequestEntityTooLarge:
        return jsonify(error='Upload exceeds the 30 MB limit. Choose a smaller file.'),413
    except (ValueError, OSError) as e: return jsonify(error=str(e)),400
    except Exception as e: app.logger.exception('Processing failed'); return jsonify(error=f'Processing failed: {e}'),500

@app.get('/api/demo')
def benchmark_demo():
    """Run one fixed published SIDD-CC test pair through the live workbench paths."""
    source_path=ROOT/'static'/'demo'/'siddcc_scene0164_block01_input.png'
    reference_path=ROOT/'static'/'demo'/'siddcc_scene0164_block01_reference.png'
    try:
        if not source_path.is_file() or not reference_path.is_file():
            return jsonify(error='The benchmark demo images are missing from this installation.'),503
        with Image.open(source_path) as source:
            im=source.convert('RGB').copy()
        with Image.open(reference_path) as reference:
            gt=reference.convert('RGB').copy()
        if im.size != (256,256) or gt.size != im.size:
            return jsonify(error='The benchmark demo pair failed its 256×256 integrity check.'),500
        result=process_decoded(im,
            'SIDD-CC held-out sRGB · scene 0164 · block 1 of 928',None,'rgb',{},gt,
            mode='human',strength=1.0,style=None)
        result['demo']={
            'dataset':'SIDD-CC SRGB test split', 'scene_id':'0164', 'block_index':1,
            'block_count':928, 'reference':'Published aligned clean sRGB reference',
            'note':'Scores shown are for this single held-out block after the workbench finishing path; they are not the 928-block benchmark average.'}
        result['processing_note']='Published sRGB test block → workbench classical / learned / hybrid paths; not Bayer RAW processing.'
        return jsonify(result)
    except Exception as e:
        app.logger.exception('Benchmark demo failed')
        return jsonify(error=f'Benchmark demo failed: {e}'),500

if __name__=='__main__': app.run(host=os.environ.get('HOST','127.0.0.1'),port=int(os.environ.get('PORT','5050')),debug=False)
