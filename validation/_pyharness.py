"""Shared helpers for Python backend validation (scottierieh/backend)."""
import json, subprocess, sys, os
PASS=[0]; FAIL=[0]; LOG=[]
def run_script(script, payload):
    """Run a backend CLI script with a JSON payload on stdin, return parsed JSON stdout."""
    root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p=subprocess.run([sys.executable, os.path.join(root,script)],
                     input=json.dumps(payload), capture_output=True, text=True)
    if p.returncode!=0:
        raise RuntimeError(f"{script} failed: {p.stderr[:500]}")
    return json.loads(p.stdout)
def chk(name, got, exp, tol=1e-6):
    if isinstance(exp,(str,bool)):
        ok = got==exp
    else:
        ok = got is not None and exp is not None and abs(float(got)-float(exp))<=tol
    if ok: PASS[0]+=1; LOG.append(f"PASS | {name} = {got}")
    else:  FAIL[0]+=1; LOG.append(f"FAIL | {name} got={got} exp={exp}")
def report(title):
    print("\n".join(LOG)); print(f"\n==== {title}: {PASS[0]} PASS, {FAIL[0]} FAIL ====")

def find_key(d, key):
    """Depth-first search for the first occurrence of `key` in nested dicts/lists."""
    if isinstance(d, dict):
        if key in d: return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None: return r
    elif isinstance(d, list):
        for v in d:
            r = find_key(v, key)
            if r is not None: return r
    return None
