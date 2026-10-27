"""Generate reproducible synthetic NMV evidence; no vendor DLL/device interaction."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import warnings

import cv2
import numpy as np
import scipy

from nmv_reference import Settings, LiveDisplayModel, block_convert, estimate, filter_gate, preferred_size

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation"
SEED = 20261023


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k:clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)):
        return [clean(x) for x in v]
    return v


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    h,w = 512,640
    base = np.clip(cv2.GaussianBlur(rng.normal(127,40,(h,w)).astype(np.float32),
                                  (0,0),1)*1.4-50,0,255).astype(np.uint8)
    def shift(a,x,y):
        return cv2.warpAffine(a,np.float32([[1,0,x],[0,1,y]]),(w,h),
                             flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    yy,xx = np.mgrid[:h,:w]
    periodic = np.rint(127+50*np.cos(2*np.pi*xx/32)+50*np.cos(2*np.pi*yy/32)).astype(np.uint8)
    noise = rng.integers(0,256,(h,w),dtype=np.uint8)
    cases = [
        ("identity",base,base.copy(),[0,0],"same pixels; finite-ratio caveat"),
        ("integer_shift",base,shift(base,5,-3),[5,-3],"translation positive"),
        ("fractional_shift",base,shift(base,1.5,2.5),[1.5,2.5],"subpixel positive"),
        ("brightness_gain",base,np.clip(shift(base,5,-3).astype(float)*1.2+10,0,255).astype(np.uint8),[5,-3],"gain and offset"),
        ("threshold_large_16",base,shift(base,16,0),[16,0],"managed >15 reset"),
        ("threshold_large_24",base,shift(base,24,3),[24,3],"managed >20 reset"),
        ("black_zero",np.zeros_like(base),np.zeros_like(base),None,"no texture; zero correlation"),
        ("flat_nonzero",np.full_like(base,100),np.full_like(base,100),None,"window-induced confidence; no recoverable motion"),
        ("periodic_hidden_32",periodic,np.roll(periodic,32,axis=1),[32,0],"period shift indistinguishable from identity"),
        ("unrelated_frames",base,noise,None,"wrong frame / unrelated texture"),
        ("rotation_2deg",base,cv2.warpAffine(base,cv2.getRotationMatrix2D((w/2,h/2),2.,1.),(w,h),borderMode=cv2.BORDER_REFLECT_101),None,"translation model cannot represent rotation"),
        ("exposure_blur",base,cv2.filter2D(shift(base,5,-3),-1,np.ones((1,15),np.float32)/15),[5,-3],"shift estimate does not deblur pixels"),
    ]
    fixtures = {}
    records = []
    settings = Settings()
    for name,a,b,truth,note in cases:
        fixtures[name+"__previous"] = a
        fixtures[name+"__current"] = b
        r = estimate(a,b,settings)
        stable = estimate(a,b,replace(settings,stable_ratio=True))
        cvshift,cvresponse = cv2.phaseCorrelate(block_convert(a,8),block_convert(b,8))
        records.append({"name":name,"known_synthetic_translation":truth,"note":note,
                        "result":r,"gate":filter_gate(r),
                        "stable_ratio_improvement_result":stable,
                        "opencv_builtin_comparison_only":{"shift_downsampled":list(cvshift),
                            "response":cvresponse,"not_the_vendor_metric":True},
                        "absolute_xy_error":None if truth is None else
                            np.abs(np.array(r["native_shift"])-truth).tolist()})
    np.savez_compressed(OUT/"fixtures.npz",**fixtures)
    by_name = {r["name"]:r for r in records}
    checks = []
    def check(name,condition,detail):
        checks.append({"name":name,"passed":bool(condition),"detail":detail})
        if not condition:
            raise AssertionError(name)
    for name in ("integer_shift","fractional_shift","brightness_gain"):
        r = by_name[name]
        check(name+"_subpixel_recovery",max(r["absolute_xy_error"])<1 and not r["gate"]["reset"],
              "Synthetic error <1 original pixel, accepted; not a device tolerance")
    check("large_16_reset",by_name["threshold_large_16"]["gate"]["reason"]=="blur_shift_gt15","Gate after successful estimate")
    check("large_24_reset",by_name["threshold_large_24"]["gate"]["reason"]=="stage_shift_gt20","Gate after successful estimate")
    check("black_rejected",by_name["black_zero"]["gate"]["reset"],"Zero peak leads boundary/low-reliability failure")
    check("flat_false_confidence_demonstrated",not by_name["flat_nonzero"]["gate"]["reset"],"Counterexample intentionally ACCEPTED despite no texture")
    check("periodic_alias_demonstrated",not by_name["periodic_hidden_32"]["gate"]["reset"] and
          by_name["periodic_hidden_32"]["absolute_xy_error"][0]>30,"Counterexample intentionally ACCEPTED despite 32-pixel ambiguity")
    check("unrelated_rejected",by_name["unrelated_frames"]["gate"]["reset"],"Specific seeded unrelated pair; not universal detection")
    # Gate boundaries and negative floor behavior are exactly defined independent of FFT error.
    gate_records=[]
    for x,r in [(15,.36),(15.001,.36),(20,.36),(20.001,.36),(.0012,.35),(-.0012,.36)]:
        o=filter_gate({"managed_shift":[x,0.],"reliability":r})
        gate_records.append({"x":x,"reliability":r,"result":o})
    check("thresholds_strict",not gate_records[0]["result"]["reset"] and gate_records[1]["result"]["reset"] and
          gate_records[2]["result"]["reason"]=="blur_shift_gt15" and gate_records[3]["result"]["reason"]=="stage_shift_gt20",
          "15 accepted; 15.001 rejected; exactly20 reaches >15 branch")
    check("reliability_equal_rejected",gate_records[4]["result"]["reset"],"Threshold requires > .35")
    check("negative_floor",gate_records[5]["result"]["shift"][0]==-.002,"Negative values floor, not round to nearest")
    # Validate conversion edges (alpha ignored, unsigned16, partial block discarded).
    sample=np.arange(6*10*4,dtype=np.uint16).reshape(6,10,4)
    sample[:,:,3]=65535
    converted=block_convert(sample,4)
    expected=np.array([[sample[:4,:4,:3].sum()/16,sample[:4,4:8,:3].sum()/16]],np.float32)
    check("block_conversion",np.array_equal(converted,expected),"4ch/16bit; alpha ignored; trailing rows/columns dropped")
    check("even_fft_table",preferred_size(63)==64 and preferred_size(81)==90,"Exact shipped table, not generic FFT size helper")
    # Accumulation and explicit base reset. All source buffers remain unchanged.
    model=LiveDisplayModel(settings)
    timeline=[]
    for i in range(7):
        frame=shift(base,i*6,0)
        before=hashlib.sha256(frame.tobytes()).hexdigest()
        o=model.feed(frame,display_size=(320,256))
        check("frame_"+str(i)+"_unchanged",before==hashlib.sha256(frame.tobytes()).hexdigest(),"No pixel resampling")
        timeline.append({"frame":i,"truth_from_first":[i*6,0],"state":o})
    check("first_frame_reset",timeline[0]["state"]["gate"]["reason"]=="new_base","Reference creation resets display")
    check("accumulation_boundary_reset",abs(timeline[4]["state"]["accumulated_pixels"][0])==0,
          "Four ~6px incremental motions overflow 20px display buffer")
    model.request_base()
    explicit=model.feed(base)
    check("explicit_base_reset",explicit["accumulated_pixels"]==[0.,0.] and explicit["gate"]["reason"]=="new_base","Base request consumed")
    resized=model.feed(base[:400,:600])
    check("size_change_reset",resized["gate"]["reason"]=="new_base","Dimension change recreates reference")
    # A rejected big move still becomes reference: next +2px motion should be accepted.
    model2=LiveDisplayModel(settings)
    m0=model2.feed(base)
    m1=model2.feed(shift(base,24,3))
    m2=model2.feed(shift(base,26,4))
    check("rejected_gate_advances_reference",m1["gate"]["reset"] and not m2["gate"]["reset"] and
          abs(m2["estimate"]["native_shift"][0]-2)<1,"Rejection does not freeze previous frame")
    fixture_hash=digest(OUT/"fixtures.npz")
    result={"purpose":"Independent synthetic reconstruction; no vendor/device execution", "seed":SEED,
            "settings":asdict(settings), "versions":{"python":platform.python_version(),"numpy":np.__version__,
              "scipy":scipy.__version__,"opencv":cv2.__version__,"platform":platform.platform()},
            "source_sha256":{p:digest(ROOT/p) for p in ["nmv_reference.py","run_validation.py","fft-sizes.json"]},
            "fixtures_sha256":fixture_hash,"cases":records,"gate_boundaries":gate_records,
            "state_timeline":timeline,"explicit_base_reset":explicit,"size_change_reset":resized,
            "rejected_reference_sequence":[m0,m1,m2],"checks":checks,
            "counts":{"cases":len(cases),"checks":len(checks),"passed":sum(c["passed"] for c in checks)},
            "limits":["No vendor numeric oracle","Synthetic successes do not validate hardware/real-frame freshness",
                      "SciPy solver is not proved bit-equivalent","Stable ratio is an analyst change",
                      "Nonfinite numbers are JSON null; residual_finite remains explicit"]}
    (OUT/"results.json").write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 合成验证结果", "",f"固定 seed `{SEED}`，{len(cases)} 对输入，{len(checks)} 项研究实现检查通过。", "",
           "| 场景 | native X/Y（原图像素） | reliability | gate |", "|---|---|---:|---|"]
    for r in records:
        x,y=r["result"]["native_shift"]
        lines.append(f"| {r['name']} | {x:.4f}, {y:.4f} | {r['result']['reliability']:.6f} | {r['gate']['reason']} |")
    lines.extend(["", "非零平坦图与完整周期平移故意保留为高置信反例；它们‘通过’的是反例被复现，并非配准正确。",
                  "", "identity/平坦场景可出现直接 sin/sin 的非有限残差，详见 JSON residual_finite。改用解析极限的结果单独保存。",
                  "", "输入 fixture、源码 SHA、库版本及所有状态均在 results.json。未运行厂商 DLL、设备或 FDM 产品代码。"])
    (OUT/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
