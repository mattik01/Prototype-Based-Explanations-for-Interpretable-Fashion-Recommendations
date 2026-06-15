import torch

root = "Master/experiments/reg_sensitivity"
print(f"{'variant':40s} {'item_protos':>14s} {'user_protos':>14s}")
for ds in ("ml-1m", "hm_1_month"):
    for v in ("reg_off", "reg_x0.01", "reg_x0.1", "reg_x1", "reg_x10", "reg_x100"):
        d = f"{root}/user_item_proto_{ds}_s38210573_{v}/best_model.pth"
        sd = torch.load(d, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        shapes = {k: tuple(t.shape) for k, t in sd.items() if k.endswith("prototypes")}
        item = next((s for k, s in shapes.items() if "item" in k), None)
        user = next((s for k, s in shapes.items() if "user" in k), None)
        print(f"{ds + '/' + v:40s} {str(item):>14s} {str(user):>14s}")
