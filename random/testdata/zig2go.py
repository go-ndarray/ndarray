# Generates random/zz_ziggurat.go from NumPy's ziggurat_constants.h:
#   python3 random/testdata/zig2go.py <numpy>/numpy/random/src/distributions/ziggurat_constants.h random/zz_ziggurat.go
# The C literals are copied verbatim (suffixes dropped): Go rounds an exact
# decimal constant to float32/float64 the way a C compiler rounds the literal.
import re, sys

src = open(sys.argv[1]).read()
gotype = {'uint64_t': 'uint64', 'double': 'float64', 'uint32_t': 'uint32', 'float': 'float32'}
scalars = {'ziggurat_nor_r': 'zigNorR', 'ziggurat_nor_inv_r': 'zigNorInvR', 'ziggurat_exp_r': 'zigExpR',
           'ziggurat_nor_r_f': 'zigNorRF', 'ziggurat_nor_inv_r_f': 'zigNorInvRF', 'ziggurat_exp_r_f': 'zigExpRF'}

def goname(c):
    a, b = c.split('_')
    return a + b.capitalize()

out = ["// Code generated from numpy/random/src/distributions/ziggurat_constants.h\n"
       "// (NumPy, BSD-3-Clause; see the NumPy license) by testdata/zig2go.py. DO NOT EDIT.\n\npackage random\n"]
for m in re.finditer(r'static const (\w+) (\w+)\[\] = \{(.*?)\};', src, re.S):
    t, name, body = m.groups()
    vals = [v.strip() for v in body.replace('\n', ' ').split(',') if v.strip()]
    vals = [re.sub(r'(ULL|UL|U)$', '', v) if t.startswith('uint') else re.sub(r'[fF]$', '', v) for v in vals]
    out.append(f"var {goname(name)} = [{len(vals)}]{gotype[t]}{{")
    for i in range(0, len(vals), 4):
        out.append("\t" + ", ".join(vals[i:i + 4]) + ",")
    out.append("}\n")
for m in re.finditer(r'static const (double|float) (\w+) =\s*([0-9.eE+-]+)f?;', src):
    t, name, v = m.groups()
    out.append(f"const {scalars[name]} {gotype[t]} = {v}")
open(sys.argv[2], 'w').write("\n".join(out) + "\n")
