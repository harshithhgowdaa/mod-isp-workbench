"""Bayer handling and classical ISP primitives; sensor mosaics may be unclipped."""
from pathlib import Path
import io
import numpy as np
from PIL import Image, ImageFilter

RAW_EXT={'.dng','.nef','.cr2','.cr3','.arw','.raf','.rw2'}

def pack(m):
    h,w=m.shape; m=m[:h-h%2,:w-w%2]
    return np.stack([m[0::2,0::2],m[0::2,1::2],m[1::2,0::2],m[1::2,1::2]],-1)

def mosaic_rgb(rgb):
    a=np.asarray(rgb,dtype=np.float32)/255 if isinstance(rgb,Image.Image) else rgb
    h,w=a.shape[:2]; a=a[:h-h%2,:w-w%2]; m=np.empty(a.shape[:2],np.float32)
    m[0::2,0::2]=a[0::2,0::2,0];m[0::2,1::2]=a[0::2,1::2,1];m[1::2,0::2]=a[1::2,0::2,1];m[1::2,1::2]=a[1::2,1::2,2]
    return m

def decode_bayer(data, filename):
    ext=Path(filename).suffix.lower()
    if ext=='.npy':
        m=np.load(io.BytesIO(data),allow_pickle=False).astype(np.float32)
        if m.ndim!=2 or not np.isfinite(m).all() or m.min() < -0.5 or m.max() > 8: raise ValueError('NPY must be a finite 2-D RGGB Bayer mosaic in black-subtracted sensor-normalized units with AsShotNeutral white balance already applied (accepted range -0.5 to 8).')
        return m, {'kind':'Black-subtracted, white-balanced RGGB Bayer array','gains':[1.,1.,1.,1.],'ccm':np.eye(3).tolist(),'preserve_bayer_range':True}
    if ext not in RAW_EXT: raise ValueError('Unsupported Bayer format.')
    try: import rawpy
    except ImportError: raise ValueError('Install rawpy to decode camera RAW files.')
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            colors=raw.raw_colors_visible.copy(); m=raw.raw_image_visible.astype(np.float32)
            desc=raw.color_desc.decode('ascii').rstrip('\x00'); pattern=colors[:2,:2]
            labels=[[desc[int(i)] for i in row] for row in pattern]
            # Normalize a crop to RGGB, retaining the original camera color indexes.
            shifts={'RGGB':(0,0),'GRBG':(0,1),'GBRG':(1,0),'BGGR':(1,1)}
            pat=''.join(''.join(row) for row in labels)
            if pat not in shifts: raise ValueError(f'Unsupported Bayer pattern {pat}; only 2×2 RGB Bayer sensors are supported.')
            sy,sx=shifts[pat]; m=m[sy:,sx:]; colors=colors[sy:,sx:]
            black=np.array(raw.black_level_per_channel,dtype=np.float32); white=float(raw.white_level)
            levels=black[colors]
            sensor_values=m
            low_clipped=float(np.mean(sensor_values<=levels+1.0)*100)
            high_clipped=float(np.mean(sensor_values>=white-1.0)*100)
            m=np.clip((sensor_values-levels)/np.maximum(white-levels,1),0,1)
            wb=np.array(raw.camera_whitebalance,dtype=np.float32); g=wb[1] if wb[1]>0 else 1
            gains=[float(wb[int(colors[y,x])]/g) if wb[int(colors[y,x])]>0 else 1. for y,x in [(0,0),(0,1),(1,0),(1,1)]]
            # Keep the DNG color matrix as sensor metadata. `color_matrix` maps
            # reference XYZ into camera space; it is not a camera-to-sRGB CCM,
            # so the UI can report its presence but must not apply it directly.
            color_matrix=np.asarray(raw.color_matrix,dtype=np.float32)[:,:3]
            color_matrix_available=bool(color_matrix.shape==(3,3) and np.isfinite(color_matrix).all() and np.linalg.matrix_rank(color_matrix)==3)
            # LibRaw's own render applies DNG matrices, camera WB, demosaic and
            # output color conversion in the correct order. Use it as a color-
            # managed baseline for unknown cameras, not the generic Bayer path.
            rendered=raw.postprocess(use_camera_wb=True,no_auto_bright=True,output_color=rawpy.ColorSpace.sRGB,output_bps=8,half_size=False,bright=1.0,demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)
            from PIL import Image
            rendered=Image.fromarray(rendered,'RGB')
            # Bayer network input and output are aligned to even sensor extents.
            m=m[:m.shape[0]-m.shape[0]%2,:m.shape[1]-m.shape[1]%2]
            if rendered.size != (m.shape[1],m.shape[0]):
                rendered=rendered.resize((m.shape[1],m.shape[0]),Image.Resampling.LANCZOS)
            make=str(getattr(raw,'camera_make','') or '').strip()
            camera=str(getattr(raw,'camera_model','') or '').strip()
            # rawpy versions built against LibRaw do not expose camera make/model
            # as attributes. DNG/TIFF EXIF tags are a safe metadata-only fallback.
            if not make or not camera:
                try:
                    with Image.open(io.BytesIO(data)) as source:
                        exif=source.getexif()
                        make=make or str(exif.get(271,'') or '').strip()
                        camera=camera or str(exif.get(272,'') or '').strip()
                except Exception:
                    pass
            camera_name=' '.join(part for part in (make,camera) if part) or 'Camera model not identified'
            black_rggb=[float(black[int(colors[y,x])]) for y,x in [(0,0),(0,1),(1,0),(1,1)]]
            record={
                'camera':camera_name,
                'cfa':pat,
                'black_level_rggb':black_rggb,
                'white_level':white,
                'white_balance_rggb':[round(v,4) for v in gains],
                'color_matrix':'Embedded camera matrix used by LibRaw' if color_matrix_available else 'No usable embedded matrix reported',
                'renderer':'LibRaw AHD · camera WB · sRGB',
                'shadow_clip_percent':round(low_clipped,4),
                'highlight_clip_percent':round(high_clipped,4),
                'noise_assessment':'Not calibrated for this camera',
                'metadata_status':'RAW metadata parsed'
            }
            return m,{'kind':f'Camera RAW · {pat} normalized to RGGB','gains':gains,'ccm':None,
                      'camera_matrix_available':color_matrix_available,'raw_render':rendered,'raw_integrity':record}
    except ValueError: raise
    except Exception as e: raise ValueError(f'Could not decode RAW: {e}')

def white_balance(m,gains,clip=True):
    out=m.copy()
    for gain,(y,x) in zip(gains,[(0,0),(0,1),(1,0),(1,1)]): out[y::2,x::2]*=gain
    return np.clip(out,0,1) if clip else out

def demosaic(m):
    """Demosaic an even RGGB mosaic with Malvar-He-Cutler 5x5 filters.

    The center pixel's measured CFA value is always restored exactly after
    filtering. The green-at-red/blue and red/blue-at-green kernels use the
    Bayer phase, while the opposite-color samples use the diagonal kernel.
    """
    from scipy.ndimage import convolve

    m=np.asarray(m,dtype=np.float32)
    h,w=m.shape
    if h<2 or w<2:
        raise ValueError('Bayer mosaic must be at least 2x2.')
    h-=h%2;w-=w%2;m=m[:h,:w]
    green_at_rb=np.array([
        [0,0,-1,0,0],
        [0,0,2,0,0],
        [-1,2,4,2,-1],
        [0,0,2,0,0],
        [0,0,-1,0,0],
    ],dtype=np.float32)/8
    rb_at_g_h=np.array([
        [0,0,.5,0,0],
        [0,-1,0,-1,0],
        [-1,4,5,4,-1],
        [0,-1,0,-1,0],
        [0,0,.5,0,0],
    ],dtype=np.float32)/8
    rb_at_g_v=rb_at_g_h.T.copy()
    rb_at_opposite=np.array([
        [0,0,-1.5,0,0],
        [0,2,0,2,0],
        [-1.5,0,6,0,-1.5],
        [0,2,0,2,0],
        [0,0,-1.5,0,0],
    ],dtype=np.float32)/8

    out=np.empty((h,w,3),dtype=np.float32)
    red=m[0::2,0::2];green_r=m[0::2,1::2]
    green_b=m[1::2,0::2];blue=m[1::2,1::2]
    rmask=np.zeros((h,w),dtype=bool);rmask[0::2,0::2]=True
    bmask=np.zeros((h,w),dtype=bool);bmask[1::2,1::2]=True

    g=convolve(m,green_at_rb,mode='reflect')
    out[...,1]=g
    out[0::2,1::2,1]=green_r;out[1::2,0::2,1]=green_b

    opposite=convolve(m,rb_at_opposite,mode='reflect')
    out[...,0]=opposite;out[...,2]=opposite
    out[0::2,0::2,0]=red;out[1::2,1::2,2]=blue

    horizontal=convolve(m,rb_at_g_h,mode='reflect')
    vertical=convolve(m,rb_at_g_v,mode='reflect')
    # Green samples beside a red site have red horizontally and blue
    # vertically; the orientation is reversed beside a blue site.
    out[0::2,1::2,0]=horizontal[0::2,1::2]
    out[0::2,1::2,2]=vertical[0::2,1::2]
    out[1::2,0::2,0]=vertical[1::2,0::2]
    out[1::2,0::2,2]=horizontal[1::2,0::2]
    out[...,0][rmask]=red.ravel()
    out[...,2][bmask]=blue.ravel()
    return np.clip(out,0,1)

def rgb_image(a): return Image.fromarray(np.uint8(np.clip(a,0,1)*255))
