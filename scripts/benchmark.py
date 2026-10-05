import torch,time,numpy as np
print('Torch',torch.__version__,'MPS',torch.backends.mps.is_available(),flush=True)
torch.set_num_threads(4)
for device,sparse in [('cpu',True),('mps',False)]:
 if device=='mps' and not torch.backends.mps.is_available():continue
 for batch in [2048,8192]:
  v=60000;d=100;k=3
  a=torch.nn.Embedding(v,d,sparse=sparse,device=device)
  b=torch.nn.Embedding(v,d,sparse=sparse,device=device)
  opt=torch.optim.SGD(list(a.parameters())+list(b.parameters()),lr=.02)
  ids=torch.randint(v,(batch,),device=device); ctx=torch.randint(v,(batch,),device=device); neg=torch.randint(v,(batch,k),device=device)
  start=time.perf_counter()
  for _ in range(30):
   opt.zero_grad(set_to_none=True); x=a(ids);p=(x*b(ctx)).sum(-1); n=(x[:,None,:]*b(neg)).sum(-1)
   loss=-(torch.nn.functional.logsigmoid(p)+torch.nn.functional.logsigmoid(-n).sum(-1)).sum()
   loss.backward();opt.step()
  if device=='mps':torch.mps.synchronize()
  dt=time.perf_counter()-start
  print(device,sparse,batch,'pairs/s',int(batch*30/dt),'seconds',dt,flush=True)
