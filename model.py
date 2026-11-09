"""Small joint Bayer demosaic/restoration network with Haar attention and color MLP."""
import torch
from torch import nn
import torch.nn.functional as F

class ModISP(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(16,width,3,padding=1),nn.ReLU(),nn.Conv2d(width,width,3,padding=1),nn.ReLU())
        self.attn=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(width,width//4,1),nn.ReLU(),nn.Conv2d(width//4,width,1),nn.Sigmoid())
        self.spatial=nn.Conv2d(width,1,3,padding=1)
        self.dec=nn.Conv2d(width,48,3,padding=1)
        self.color=nn.Sequential(nn.Conv2d(6,32,1),nn.ReLU(),nn.Conv2d(32,32,1),nn.ReLU(),nn.Conv2d(32,3,1))
    def forward(self,x):
        # x: four RGGB channels at half resolution; spatial dimensions must be even.
        h,w=x.shape[-2:]; x=F.pad(x,(0,w%2,0,h%2),mode='replicate')
        p=F.pixel_unshuffle(x,2); a,b,c,d=p[:,0::4],p[:,1::4],p[:,2::4],p[:,3::4]
        z=torch.cat(((a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2),1)
        f=self.enc(z); f=f*self.attn(f)*torch.sigmoid(self.spatial(f)); z=self.dec(f)
        rgb=F.pixel_shuffle(F.pixel_shuffle(z,2),2)
        context=rgb.mean(dim=(2,3),keepdim=True).expand_as(rgb)
        rgb=rgb+self.color(torch.cat((rgb,context),dim=1))
        return rgb[:,:,:h*2,:w*2]

def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('architecture')!='bayer-v1': raise ValueError('Incompatible checkpoint; train a bayer-v1 model.')
    if not checkpoint.get('trained_steps',0): raise ValueError('Checkpoint has no recorded training steps')
    model=ModISP(); model.load_state_dict(checkpoint['model']);model.eval()
    return model,{'trained_steps':checkpoint['trained_steps'],'dataset':checkpoint.get('dataset','not recorded'),'architecture':'bayer-v1','teacher_format':checkpoint.get('teacher_format',''),'teacher':checkpoint.get('teacher',''),'init_checkpoint':checkpoint.get('init_checkpoint'),'init_trained_steps':checkpoint.get('init_trained_steps',0),'epochs':checkpoint.get('epochs'),'best_epoch':checkpoint.get('best_epoch'),'validation_l1':checkpoint.get('validation_l1'),'patch_size':checkpoint.get('patch_size'),'train_pairs':checkpoint.get('train_pairs'),'device':checkpoint.get('device','not recorded')}
