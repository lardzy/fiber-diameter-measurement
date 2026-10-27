"""Generate deterministic synthetic inputs once, then exercise independent model.

Run from repository: .venv/bin/python <this-file>. No vendor code execution.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
import cv2
import reference_prototype as r

ROOT = Path(__file__).resolve().parent
SEED = 20261023
FIXTURE = ROOT / "fixtures" / "synthetic-inputs.npz"
GRID_FIXTURE = ROOT / "fixtures" / "grid-inputs.npz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def fixture():
    if not FIXTURE.exists():
        FIXTURE.parent.mkdir(exist_ok=True)
        rng = np.random.default_rng(SEED)
        scene = rng.integers(0, 65536, (128, 160), np.uint16)
        smooth = cv2.GaussianBlur(rng.normal(size=(128, 128)), (0, 0), 2)
        smooth = (smooth-smooth.min())/(smooth.max()-smooth.min())*50000+2000
        phase_shift = (3.3, -2.4)
        wy, wx = np.fft.fftfreq(128)[:, None], np.fft.fftfreq(128)[None, :]
        shifted = np.fft.ifft2(np.fft.fft2(smooth)*np.exp(-2j*np.pi*(wx*phase_shift[0]+wy*phase_shift[1]))).real
        yy, xx = np.mgrid[:128, :160]
        periodic = ((xx % 8 < 4)*50000+5000).astype(np.uint16)
        height = .02*xx+.03*yy+1.0
        points = np.array([(x, y) for y in range(3) for x in range(3)], float)*100
        params = np.array([2, -3, 1.02, .98, .025])
        target = r.transform_five(points, params)
        corrupt_target = target.copy(); corrupt_target[4] += [30, -20]
        np.savez_compressed(FIXTURE, scene=scene, smooth=smooth, shifted=shifted,
                            phase_shift=phase_shift, periodic=periodic, height=height,
                            points=points, transform_parameters=params, target=target,
                            corrupt_target=corrupt_target, seed=np.array(SEED))
    return np.load(FIXTURE)


def main():
    f = fixture()
    results = []
    def case(name, fn):
        try:
            details = fn()
            results.append({"name": name, "pass": True, "details": details})
        except Exception as ex:
            results.append({"name": name, "pass": False, "error": f"{type(ex).__name__}: {ex}"})
    def check(condition, details=None):
        assert condition, str(details)
        return details
    def expect_error(fn):
        try: fn()
        except ValueError as ex: return {"expected_error": str(ex)}
        raise AssertionError("expected ValueError")
    scene = f["scene"]
    template = scene[12:52, 16:56]
    search = lambda s, t: r.composite_search(s, t, (0, 26), (0, 22))
    def known_match():
        m = search(scene, template)
        return check((m["x"], m["y"]) == (16, 12) and m["cost"] < 1e-12, m)
    case("composite textured even-pixel ground truth", known_match)
    case("cosine invariance to positive exposure multiplier", lambda: check(r.cosine_error(template, template.astype(float)*.7) < 1e-12))
    case("cosine is sensitive to additive offset", lambda: check(r.cosine_error(template, template.astype(float)+20000) > .005,
                                                                 {"cost": r.cosine_error(template, template.astype(float)+20000)}))
    case("zero-energy ROI reports maximal cost", lambda: check(r.cosine_error(np.zeros((16,16)), np.ones((16,16))) == 1))
    def odd_shift():
        m = search(scene, scene[13:53, 17:57])
        return check((m["x"],m["y"]) != (17,13), {"expected": [17,13], "actual":m, "limitation":"2-pixel grid cannot represent odd displacement"})
    case("odd displacement limitation is exposed", odd_shift)
    def periodic():
        m=search(f["periodic"], f["periodic"][12:52,16:56])
        return check(m["equal_minima"]>1 and r.texture_accept(m["min_vote_value"]), m)
    case("periodic wrong matches pass texture gate", periodic)
    def unrelated():
        m=search(scene, np.flipud(template))
        return check(r.texture_accept(m["min_vote_value"]) and m["cost"]>.02, m)
    case("unrelated high contrast can pass MinVote", unrelated)
    def low_contrast():
        s=scene.astype(float)/100+25000
        m=search(s,s[12:52,16:56])
        return check((m["x"],m["y"])==(16,12) and not r.texture_accept(m["min_vote_value"]),m)
    case("correct low-contrast match rejected by MinVote", low_contrast)
    case("MinVote exact boundary is rejected",lambda:check(not r.texture_accept(.1)))
    case("out-of-range ROI explicit gap",lambda:expect_error(lambda:r.composite_search(scene,template,(-2,2),(0,2))))
    case("shared normalization preserves tile brightness difference", lambda: check(
        (lambda z: z[0][0].max()<z[0][1].max())(r.global_normalize_gray([scene//2, scene]))))
    case("flat normalization explicit numerical guard",lambda:check(r.global_normalize_gray([np.full((16,16),5,np.uint16)])[1]["analyst_guard"]))
    colors=np.array([[10,20,30],[210,220,230]],np.uint8)
    case("feather reduces overlap discontinuity",lambda:check(np.all(r.blend_pixel(colors,[2,2])==[110,120,130])))
    case("paint-order direct override",lambda:check(np.all(r.blend_pixel(colors,[2,2],enabled=False)==colors[-1])))
    case("fourth-power weights favor interior",lambda:check(r.blend_pixel(colors,[1,3],power=4)[0]>200))
    case("count truncation uses latest covering tiles",lambda:check(np.all(r.blend_pixel(colors,[2,2],count=1)==colors[-1])))
    case("weight threshold shortcut",lambda:check(np.all(r.blend_pixel(colors,[2,4],threshold=3)==colors[-1])))
    case("border exclusive extent asymmetry",lambda:check([r.border_distance(x,5,0,0,10,10) for x in [0,1,5,9,10]]==[1,1,5,1,0]))
    case("blend handles uint16 above byte range",lambda:check(r.blend_pixel(np.array([[10000],[60000]],np.uint16),[2,2])[0]==35000))
    height=f["height"];mask=np.zeros_like(height,dtype=bool);mask[:, :20]=True
    def height_offset():
        out, offset=r.align_height(height,height+3.5,mask)
        return check(np.max(abs(out-height))<1e-12,{"offset":offset})
    case("height additive seam recovered",height_offset)
    def height_tilt():
        child=height+3.5+np.arange(height.shape[1])[None,:]*.02
        out, offset=r.align_height(height,child,mask)
        return check(np.max(abs(out-height))>2,{"remaining_max_error":float(np.max(abs(out-height))),"offset":offset})
    case("height offset cannot repair tilt",height_tilt)
    case("empty height overlap fails explicitly",lambda:expect_error(lambda:r.align_height(height,height,np.zeros_like(mask))))
    case("invalid height requires mask",lambda:expect_error(lambda:r.align_height(height,np.full_like(height,np.nan),mask)))
    edges=[(1,0,10,1),(2,1,10,1),(2,0,21,1)]
    def graph_loop():
        g=r.global_positions([0,10,20],edges)
        return check(g["objective"]<1 and len(g["components"])==1 and abs(g["positions"].mean()-10)<1e-10,g)
    case("loop closure distributes inconsistency",graph_loop)
    def zero_bad():
        good=r.global_positions([0,10,20],[(1,0,10,1),(2,1,10,1)])
        bad=r.global_positions([0,10,20],[(1,0,10,1),(2,1,10,1),(2,0,50,0)])
        return check(np.max(abs(bad["positions"]-good["positions"]))>1,{"good":good,"bad":bad,"limitation":"zero score is downweighted, never deleted"})
    case("zero-score bad edge still biases result",zero_bad)
    def disconnected():
        g=r.global_positions([0,10,100,110,500],[(1,0,9,1),(3,2,11,1)])
        return check(len(g["components"])==3 and abs(g["positions"][-1]-500)<1e-9,g)
    case("disconnected graph and isolated tile explicit gauges",disconnected)
    case("invalid edge identity fails",lambda:expect_error(lambda:r.global_positions([0,1],[(1,1,0,1)])))
    case("hardware exact identity when no edges",lambda:check(np.allclose(r.global_positions([3,4,5],[])["positions"],[3,4,5])))
    def phase(lpf):
        m=r.phase_ols(f["smooth"],f["shifted"],lpf=lpf,hann=False)
        error=float(np.linalg.norm(np.array(m["shift"])-f["phase_shift"]))
        return check(error<.15 if lpf else .1<error<.6,{**m,"ground_truth":f["phase_shift"],"error":error})
    case("OLS centroid exposes measurable fractional bias",lambda:phase(False))
    case("OLS amplitude-fit fractional Fourier-shift image",lambda:phase(True))
    def phase_integer():
        m=r.phase_ols(scene[:128,:128],np.roll(scene[:128,:128],(4,-6),(0,1)),lpf=True,hann=False)
        return check(np.linalg.norm(np.array(m["shift"])-[-6,4])<.15,m)
    case("OLS LPF integer translation",phase_integer)
    def phase_blank():
        m=r.phase_ols(np.zeros((64,64)),np.zeros((64,64)),lpf=False)
        return check(not r.ols_gate(m["score"]),m)
    case("OLS dark image has rejected score",phase_blank)
    def phase_limit():
        m=r.phase_ols(scene[:128,:128],np.roll(scene[:128,:128],(0,30),(0,1)),lpf=False,hann=False,search_size=(10,10))
        return check(abs(m["shift"][0]-30)>10,{**m,"limitation":"search window excludes true displacement"})
    case("phase restricted search cannot recover large motion",phase_limit)
    case("OLS gate rejects both lower and upper boundaries",lambda:check(not r.ols_gate(.05) and not r.ols_gate(1.2)))
    def transform():
        fit=r.fit_hardware_transform(f["points"],f["corrupt_target"],SEED,.1,100)
        return check(fit["inliers"].sum()==8 and fit["rms"]<1e-6,fit)
    case("hardware transform excludes synthetic outlier",transform)
    case("hardware collinear degeneracy explicit analyst guard",lambda:expect_error(lambda:r.fit_hardware_transform(np.array([[0,0],[1,0],[2,0]]),np.array([[0,0],[1,0],[2,0]]),SEED,.1)))
    case("hardware too few points not falsely recovered",lambda:expect_error(lambda:r.fit_hardware_transform(f["points"][:2],f["target"][:2],SEED,.1)))
    if not GRID_FIXTURE.exists():
        rng=np.random.default_rng(SEED+1)
        raw=cv2.GaussianBlur(rng.normal(size=(224,224)),(0,0),1.1)
        world=np.rint((raw-raw.min())/(raw.max()-raw.min())*55000+5000).astype(np.uint16)
        origins=np.array([(x*64,y*64) for y in range(3) for x in range(3)])
        tiles=np.array([world[y:y+96,x:x+96] for x,y in origins])
        jitter=np.array([[1,-1],[-1,2],[2,1],[-2,1],[1,-2],[0,2],[2,-1],[-1,0],[1,1]])
        np.savez_compressed(GRID_FIXTURE,world=world,origins=origins,tiles=tiles,priors=origins+jitter,seed=SEED+1)
    grid=np.load(GRID_FIXTURE)
    def pipeline():
        q=r.reconstruct_grid(grid["tiles"],grid["priors"],3,3)
        error=np.linalg.norm(q["positions"]-grid["origins"],axis=1)
        rounded=np.array_equal(q["integer_positions"],grid["origins"])
        mae=float(np.mean(abs(q["mosaic"]-grid["world"]))) if q["mosaic"].shape==grid["world"].shape else None
        np.savez_compressed(ROOT/"fixtures"/"grid-output.npz",positions=q["positions"],integer_positions=q["integer_positions"],mosaic=q["mosaic"],valid=q["valid"])
        cv2.imwrite(str(ROOT/"fixtures"/"grid-output.png"),np.clip(q["mosaic"]/256,0,255).astype(np.uint8))
        details={"position_errors_px":error,"rounded_exact":rounded,"mosaic_mae":mae,
                 "accepted_pairs":sum(p["accepted"] for p in q["pairs"]),"total_pairs":len(q["pairs"]),
                 "positions":q["positions"],"scores":[p["score"] for p in q["pairs"]]}
        return check(error.max()<.5 and rounded and mae<1e-8,details)
    case("3x3 end-to-end overlap phase graph and feather",pipeline)
    def pipeline_dark():
        tiles=grid["tiles"].copy();tiles[4]=0
        q=r.reconstruct_grid(tiles,grid["priors"],3,3,reject_bad_edges=True)
        return check(len(q["graph_x"]["components"])==2 and sum(not p["accepted"] for p in q["pairs"])>=4,
                     {"components":q["graph_x"]["components"],"rejected":[[p["i"],p["j"]] for p in q["pairs"] if not p["accepted"]],"mode":"explicit improvement: reject failed edges"})
    case("3x3 dark tile failure retains explicit disconnected component",pipeline_dark)
    output={"schema":"dsx-stitching-independent-validation-v1","utc":datetime.now(timezone.utc).isoformat(),
            "seed":SEED,"python":sys.version,"platform":platform.platform(),
            "dependencies":{"numpy":np.__version__,"opencv":cv2.__version__,"scipy":scipy.__version__},
            "sources":{p.name:sha(p) for p in [Path(__file__),ROOT/"reference_prototype.py"]},
            "input":{"path":str(FIXTURE.relative_to(ROOT)),"sha256":sha(FIXTURE)},
            "grid_input":{"path":str(GRID_FIXTURE.relative_to(ROOT)),"sha256":sha(GRID_FIXTURE),"seed":SEED+1},
            "scope":"independent synthetic algorithms only; no original executable or DLL invoked",
            "not_vendor_equivalence":True,"passed":sum(t["pass"] for t in results),"total":len(results),"tests":results}
    (ROOT/"test-results.json").write_text(json.dumps(output,ensure_ascii=False,indent=2,default=serial)+"\n")
    print(f"{output['passed']}/{output['total']} passed; {ROOT/'test-results.json'}")
    for t in results:
        if not t["pass"]: print(t["name"],t["error"])
    return int(output["passed"] != output["total"])


if __name__=="__main__": raise SystemExit(main())
