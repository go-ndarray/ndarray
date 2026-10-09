# Generates testdata/promotion.json: `uv run --with numpy python testdata/promotion.py > testdata/promotion.json`
import numpy as np, json
dts=['bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float32','float64','complex64','complex128']
out={"numpy":np.__version__,"pair":{},"weak":{},"unary":{},"reduce":{}}
for a in dts:
  for b in dts:
    out["pair"][a+","+b]=str(np.result_type(np.dtype(a),np.dtype(b)))
for a in dts:
  x=np.zeros(2,dtype=a)
  out["weak"][a]={"int":str((x+1).dtype) if a!='bool' else str((x+1).dtype),"float":str((x+1.5).dtype),"complex":str((x+1j).dtype),
                  "truediv":str(np.true_divide(x,x).dtype),}
  u={}
  for f in ['sqrt','exp','sin','abs','negative','floor','square']:
    try: u[f]=str(getattr(np,f)(x).dtype)
    except Exception as e: u[f]="ERR:"+type(e).__name__
  out["unary"][a]=u
  r={}
  for f in ['sum','prod','mean','max','cumsum','argmax']:
    try: r[f]=str(np.asarray(getattr(np,f)(x)).dtype)
    except Exception as e: r[f]="ERR:"+type(e).__name__
  out["reduce"][a]=r
print(json.dumps(out,indent=0))
