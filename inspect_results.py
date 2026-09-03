import json

with open("outputs/phase13_9/calibration_results.json") as f:
    r = json.load(f)

print("=== DATASET ===")
for k, v in r["dataset"].items():
    print(f"  {k}: {v}")

print()
print("=== IMAGE LEVEL ===")
il = r["image_level"]
print(f"  ROC-AUC: {il['roc_auc']}")
print(f"  EER: {il['eer']}")
print(f"  Global max impostor: {il['global_max_impostor']}")
gs = il["genuine_stats"]
ims = il["impostor_stats"]
print(f"  Genuine: mean={gs['mean']:.4f}, min={gs['min']:.4f}, max={gs['max']:.4f}")
print(f"  Impostor: mean={ims['mean']:.4f}, min={ims['min']:.4f}, max={ims['max']:.4f}")

print()
print("=== IDENTITY LEVEL ===")
idl = r["identity_level"]
print(f"  ROC-AUC: {idl['roc_auc']}")
print(f"  EER: {idl['eer']}")
print(f"  Global max impostor: {idl['global_max_impostor']}")
print(f"  Genuine mean: {idl['genuine_stats']['mean']:.4f}")
print(f"  Impostor mean: {idl['impostor_stats']['mean']:.4f}")

print()
print("=== FIXED THRESHOLDS ===")
for name, data in r["fixed_thresholds"].items():
    print(f"  {name:12s} t={data['threshold']:.4f}: FAR={data['far']:.4f} FRR={data['frr']:.4f} F1={data['f1']:.4f} TP={data['tp']} FP={data['fp']} TN={data['tn']} FN={data['fn']}")

print()
print("=== GALLERY SIZES ===")
for gs, data in r["gallery_sizes"].items():
    print(f"  Size {gs}: ref={data['reference_vectors']}, AUC={data['image_roc_auc']:.4f}, EER={data['image_eer']:.4f}, max_imp={data['global_max_impostor']:.4f}")

print()
print("=== HARD NEGATIVES (image) ===")
for hn in r["hard_negatives"]["image_level_top5"][:3]:
    print(f"  {hn['query_person_id']} -> {hn['impostor_person_id']}: sim={hn['similarity']:.4f}")

print()
print("=== HARD NEGATIVES (identity) ===")
for hn in r["hard_negatives"]["identity_level_top5"][:3]:
    print(f"  {hn['query_person_id']} -> {hn['impostor_person_id']}: sim={hn['identity_score']:.4f}")
