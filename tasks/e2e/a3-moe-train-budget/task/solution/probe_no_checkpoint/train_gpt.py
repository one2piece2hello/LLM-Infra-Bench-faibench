# REVIEWER-ONLY zeroing witness: a recipe that never writes a checkpoint inside the budget.
# Must score exactly 0.0 with reason `build_or_entry_contract_failed`. 🔴 NOT SHIPPED.
import os, sys, time
def load_model_for_verification(checkpoint_path, device):
    raise RuntimeError("no checkpoint was produced")
if __name__ == "__main__":
    print("[probe] sleeping past the whole budget without ever saving", flush=True)
    while True:
        time.sleep(10)
