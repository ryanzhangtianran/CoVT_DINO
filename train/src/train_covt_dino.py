import os
import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaProcessor, AutoImageProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image

from covt_dino_model import COVT_DINO

CONFIG = {
    # Training parameters
    "lr": 1e-4,
    "batch_size": 32,
    "grad_accumulation_steps": 2,
    "max_steps": 10000,
    "save_steps": 1000,
    "num_workers": 8,
    # Models
    "paligemma_model_path": "/home/zhongzd/trzhang/models/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c",
    "dino_model_path": "/home/zhongzd/trzhang/models/models--facebook--dinov2-giant/snapshots/611a9d42f2335e0f921f1e313ad3c1b7178d206d",
    "num_vis_tokens": 4,
    # Data
    "dataset_id": "lerobot/droid_100",
    "image_key": "observation.images.image",
    # Logging
    "project_name": "covt-training",
    "save_dir": "./output_covt_paligemma"
}

def collate_fn_droid(batch, paligemma_proc, dino_proc, special_tokens_str, target_ids_tensor):
    pil_images = []
    texts = []

    for item in batch:
        img_tensor = item[CONFIG["image_key"]]
        img_np = img_tensor.permute(1, 2, 0).byte().cpu().numpy()
        image = Image.fromarray(img_np)
        pil_images.append(image)

        texts.append(f"Extract features from the image: {special_tokens_str}")

    paligemma_inputs = paligemma_proc(images=pil_images, text=texts, return_tensors="pt", padding=True)
    dino_inputs = dino_proc(images=pil_images, return_tensors="pt")

    return {
        "input_ids": paligemma_inputs["input_ids"],
        "attention_mask": paligemma_inputs["attention_mask"],
        "pixel_values_paligemma": paligemma_inputs.pixel_values.to(torch.bfloat16),
        "pixel_values_dino": dino_inputs.pixel_values.to(torch.bfloat16),
        "target_token_ids": target_ids_tensor
    }

def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["grad_accumulation_steps"],
        mixed_precision="bf16",
        log_with="wandb"
    )

    if accelerator.is_main_process:
        print(f"--- Training on {accelerator.num_processes} GPUs ---")
        os.makedirs(CONFIG["save_dir"], exist_ok=True)
        accelerator.init_trackers(
            project_name=CONFIG["project_name"], 
            config=CONFIG
        )

    paligemma_processor = PaliGemmaProcessor.from_pretrained(CONFIG["paligemma_model_path"])
    dino_processor = AutoImageProcessor.from_pretrained(CONFIG["dino_model_path"])

    special_tokens = [f"<vis_{i}>" for i in range(CONFIG["num_vis_tokens"])]
    num_added = paligemma_processor.tokenizer.add_tokens(special_tokens)
    special_tokens_str = "".join(special_tokens)
    
    special_token_ids = paligemma_processor.tokenizer.convert_tokens_to_ids(special_tokens)
    target_ids_tensor = torch.tensor(special_token_ids, dtype=torch.long)
    
    if accelerator.is_main_process:
        print(f"Added {num_added} special tokens: {special_tokens_str}")
        print(f"Target IDs: {target_ids_tensor}")
    
    with accelerator.main_process_first():
        model = COVT_DINO(
            paligemma_model_path=CONFIG["paligemma_model_path"],
            dino_model_path=CONFIG["dino_model_path"],
            num_vis_tokens=CONFIG["num_vis_tokens"]
        )
        model.resize_token_embeddings(len(paligemma_processor.tokenizer))

    if accelerator.is_main_process:
        print(f"Loading LeRobot Dataset: {CONFIG['dataset_id']}...")
        
    dataset = LeRobotDataset(CONFIG["dataset_id"], split="train")
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=lambda b: collate_fn_droid(
            b, 
            paligemma_processor, 
            dino_processor, 
            special_tokens_str, 
            target_ids_tensor
        )
    )

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=CONFIG["lr"])

    if accelerator.is_main_process:
        print(f"Num trainable params: {sum(p.numel() for p in params_to_optimize)}")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    global_step = 0
    progress_bar = tqdm(total=CONFIG["max_steps"], disable=not accelerator.is_local_main_process)
    data_iter = iter(dataloader)
    
    while global_step < CONFIG["max_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        with accelerator.accumulate(model):
            loss = model(**batch)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            
            if accelerator.is_main_process:
                accelerator.log({"train_loss": loss.item()}, step=global_step)
                
                if global_step % 10 == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if global_step % CONFIG["save_steps"] == 0:
                    save_path = os.path.join(CONFIG["save_dir"], f"adapter_step_{global_step}.bin")
                    unwrapped = accelerator.unwrap_model(model)
                    state_dict = unwrapped.state_dict()
                    
                    adapter_weights = {
                        k: v for k, v in state_dict.items() 
                        if not k.startswith("vlm.") and not k.startswith("dino.")
                    }
                    
                    torch.save(adapter_weights, save_path)
                    print(f"Saved adapter to {save_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.state_dict()
        adapter_weights = {
            k: v for k, v in state_dict.items() 
            if not k.startswith("vlm.") and not k.startswith("dino.")
        }
        torch.save(adapter_weights, os.path.join(CONFIG["save_dir"], "final_adapter.bin"))
        paligemma_processor.save_pretrained(CONFIG["save_dir"])
        accelerator.end_training()
        print("Training Finished.")

if __name__ == "__main__":
    main()
    