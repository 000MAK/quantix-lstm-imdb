import argparse,tarfile,urllib.request,re,json,random
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
URL='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
class Reviews(Dataset):
 def __init__(self,rows,vocab): self.rows=rows; self.vocab=vocab
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  text,y=self.rows[i]; ids=[self.vocab.get(w,1) for w in re.findall(r"[a-z]+",text.lower())[:240]] or [1]; ids=ids+[0]*max(0,240-len(ids)); return torch.tensor(ids[:240]),torch.tensor(y)
def load(folder,split):
 rows=[]
 for label,y in [('neg',0),('pos',1)]:
  for f in sorted((folder/'aclImdb'/split/label).glob('*.txt')): rows.append((f.read_text(encoding='utf8',errors='ignore'),y))
 return rows
def main():
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=2); p.add_argument('--limit',type=int,default=6000); a=p.parse_args(); r=Path(__file__).parent; d=r/'data'; (r/'outputs').mkdir(exist_ok=True); d.mkdir(exist_ok=True); archive=d/'aclImdb.tar.gz'
 if not archive.exists(): urllib.request.urlretrieve(URL,archive)
 if not (d/'aclImdb').exists():
  with tarfile.open(archive,'r:gz') as t:
   base=d.resolve()
   for m in t.getmembers():
    if not (base/m.name).resolve().is_relative_to(base): raise ValueError('Unsafe archive path')
   t.extractall(d,filter='data')
 tr=load(d,'train'); te=load(d,'test'); random.Random(42).shuffle(tr); tr=tr[:min(a.limit,len(tr))]; cut=int(.85*len(tr)); train,val=tr[:cut],tr[cut:]; te=te[:min(max(500,len(val)),len(te))]
 counts=Counter(w for text,_ in train for w in re.findall(r'[a-z]+',text.lower())); vocab={'<pad>':0,'<unk>':1}; vocab.update({w:i+2 for i,(w,c) in enumerate(counts.most_common(12000))}); device='cuda' if torch.cuda.is_available() else 'cpu'
 class Net(nn.Module):
  def __init__(self): super().__init__(); self.emb=nn.Embedding(len(vocab),64,padding_idx=0); self.rnn=nn.LSTM(64,64,batch_first=True,bidirectional=True); self.fc=nn.Linear(128,2)
  def forward(self,x):
   z=self.emb(x); lengths=(x!=0).sum(1).clamp(min=1); out,_=self.rnn(z); h=torch.cat([out[torch.arange(len(x),device=x.device),lengths-1,:64],out[:,0,64:]],1); return self.fc(h)
 model=Net().to(device); opt=torch.optim.Adam(model.parameters(),lr=.001); lossf=nn.CrossEntropyLoss()
 def run(rows,train_mode=False):
  model.train(train_mode); ls=[]; ok=n=0
  for x,y in DataLoader(Reviews(rows,vocab),batch_size=32,shuffle=train_mode):
   x,y=x.to(device),y.to(device); opt.zero_grad(); o=model(x); loss=lossf(o,y)
   if train_mode: loss.backward(); opt.step()
   ls.append(float(loss)); ok+=(o.argmax(1)==y).sum().item(); n+=len(y)
  return {'loss':round(sum(ls)/len(ls),4),'accuracy':round(ok/n,4)}
 result={}
 for e in range(a.epochs): result[f'epoch_{e+1}']={'train':run(train,True),'validation':run(val)}
 result['test']=run(te); torch.save(model.state_dict(),r/'outputs/lstm.pt'); (r/'outputs/metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
