#!/usr/bin/env python3
"""Stage-3 DURING-inference (M1->M4 confidence reviewer, prompt v3) on data2.
Iterates directly over M1 records (alignment guaranteed). Checkpoint-based/restartable.
M4 = MedGemma in bfloat16, greedy decoding. Scope: k in {0,1,4,8} x seeds 0-4."""
import sys, os, json, time, re
from pathlib import Path
import pandas as pd, torch
from PIL import Image
ROOT=Path("/data/arta/thoraxai_research/1_equity"); RERUN=ROOT/"final_rerun_paper_revision"; DATA2=ROOT/"data2"
sys.path.insert(0, str(RERUN)); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from during_ensemble import (load_m1_results, build_reviewer_user_text,
                             build_few_shot_context_v2, load_checkpoint, append_jsonl, USER_QUERY_REVIEWER)
from src.example_selector import FewShotSelector

K_SHOTS=[0,1,4,8]; SEEDS=[0,1,2,3,4]; MAX_NEW_TOKENS=1024
OUT=RERUN/"results/during_sd_v3"; OUT.mkdir(parents=True, exist_ok=True)
CKPT=OUT/"checkpoint.jsonl"

df_m1=load_m1_results(RERUN/"results/exp1v2_nv_reason/all_results.jsonl")
df_m1=df_m1[df_m1.k_shot.isin(K_SHOTS)&df_m1.seed.isin(SEEDS)].copy()
print("M1 subset:",len(df_m1),"records")

selector=FewShotSelector(positive_csv=str(DATA2/"chexpert_ptx_positive.csv"),
                         negative_csv=str(DATA2/"chexpert_ptx_negative.csv"))

def path_for(ts, image_id, gt):
    if "Internal" in ts or "CheXpert" in ts:
        sub="internal_test_pneumothorax" if gt==1 else "internal_test_normal"
        return DATA2/"chexpert_plus"/"images"/sub/f"{image_id}.png"
    sub="pneumothorax" if gt==1 else "normal"
    return DATA2/"rs_unair"/"images"/sub/f"{image_id}.png"

# ── M4 model (bfloat16) ──
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
qc=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
                      bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
proc=AutoProcessor.from_pretrained(str(ROOT/"models/MedGemma-4B-IT"))
model=Gemma3ForConditionalGeneration.from_pretrained(str(ROOT/"models/MedGemma-4B-IT"),
                                                     quantization_config=qc,device_map="auto").eval()
print("M4 loaded, VRAM",round(torch.cuda.memory_allocated()/1e9,2),"GB")

SYS_V3=("You are an expert radiologist assistant performing a second reading of a chest X-ray (CXR).\n\n"
 "A first reader (AI model) has already analyzed this image. Their full analysis is provided below.\n\n"
 "Your role:\n1. REVIEW the first reader's analysis — check if their observations match the radiographic evidence\n"
 "2. VERIFY specific findings — if the first reader identifies pneumothorax signs (e.g., visceral pleural line, "
 "absent lung markings, hyperlucency), look carefully at those areas\n"
 "3. PROVIDE YOUR OWN ASSESSMENT based on the radiographic evidence\n\n"
 "Format your response EXACTLY as follows:\n\n"
 "FIRST READER EVALUATION: [2-3 sentences evaluating the first reader's claims]\n"
 "MY OBSERVATIONS: [2-3 sentences describing what you see in the image]\n"
 "CLASSIFICATION: [PNEUMOTHORAX DETECTED or NO PNEUMOTHORAX DETECTED]\n"
 "CONFIDENCE: [Your confidence as a percentage, e.g. 85%]\n"
 "KEY EVIDENCE: [1-2 sentences with specific anatomical findings]\n\n"
 "Confidence scale:\n- 90-100%: Very confident, clear and unambiguous findings\n"
 "- 70-89%: Confident, findings present but may be subtle\n- 50-69%: Moderate confidence, some suggestive signs but uncertain\n"
 "- 30-49%: Low confidence, ambiguous findings\n- 0-29%: Very low confidence, minimal evidence\n\n"
 "Important:\n- Base your classification on radiographic evidence visible in the image.\n"
 "- If you see signs consistent with pneumothorax (even subtle), classify as PNEUMOTHORAX DETECTED.\n"
 "- If the lung fields are clear with no evidence of pneumothorax, classify as NO PNEUMOTHORAX DETECTED.\n"
 "- Be honest about your confidence — low confidence is valuable information.")

def m4_review(image_path, m1_output, ctx):
    img=Image.open(image_path); img=img.convert("RGB") if img.mode!="RGB" else img
    user_text=build_reviewer_user_text(m1_output, ctx)
    content=[{"type":"text","text":user_text},{"type":"image","image":img},{"type":"text","text":USER_QUERY_REVIEWER}]
    msgs=[{"role":"system","content":[{"type":"text","text":SYS_V3}]},{"role":"user","content":content}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inputs=proc(text=text,images=[img],return_tensors="pt").to(model.device)
    il=inputs["input_ids"].shape[1]; t0=time.time()
    with torch.no_grad():
        out=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False)
    res=proc.batch_decode(out[:,il:],skip_special_tokens=True)[0]
    del inputs,out; torch.cuda.empty_cache(); img.close()
    return res, time.time()-t0

done,_=load_checkpoint(CKPT)
total=len(df_m1); print("target",total,"| done",len(done))
buf=[]; t_start=time.time(); n=0
for k in K_SHOTS:
    for seed in SEEDS:
        ctx=""
        if k>0:
            try: ctx=build_few_shot_context_v2(selector.select(strategy="random",k=k,seed=seed),k)
            except Exception as e: print("ctx warn",k,seed,e)
        sub=df_m1[(df_m1.k_shot==k)&(df_m1.seed==seed)]
        for _,r in sub.iterrows():
            n+=1; key=(k,seed,r.test_set,str(r.image_id))
            if key in done: continue
            p=path_for(r.test_set,str(r.image_id),int(r.ground_truth))
            if not p.exists(): print("MISSING IMG",p); continue
            try: txt,dt=m4_review(str(p),r.generated_text,ctx)
            except Exception as e: txt,dt="",0.0; print("ERR",key,e)
            buf.append({"k_shot":k,"strategy":"random","seed":seed,"test_set":r.test_set,
                        "image_id":str(r.image_id),"ground_truth":int(r.ground_truth),
                        "m1_generated_text":r.generated_text,"m4_generated_text":txt,
                        "m4_inference_time":round(dt,3),"timestamp":pd.Timestamp.now().isoformat()})
            done.add(key)
            if len(buf)>=10: append_jsonl(CKPT,buf); buf=[]
            if n%200==0: print(f"  {len(done)}/{total} | {(time.time()-t_start)/max(1,n-len(done)|1):.1f}s/img avg", flush=True)
if buf: append_jsonl(CKPT,buf)
# finalize
recs=[json.loads(l) for l in open(CKPT)]
seen=set(); uniq=[]
for rr in recs:
    kk=(rr["k_shot"],rr["seed"],rr["test_set"],rr["image_id"])
    if kk not in seen: seen.add(kk); uniq.append(rr)
open(OUT/"all_results.jsonl","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in uniq)+"\n")
print(f"DONE during-inference: {len(uniq)} records -> {OUT/'all_results.jsonl'}")
