import argparse
import os
import torch

from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from transformers import PaliGemmaProcessor, AutoImageProcessor


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
    "dataset_path": "/home/zhongzd/trzhang/dataset/datasets--lerobot--droid_100/snapshots/87301a2d2e99340e2010c9ef0f1d8e780b08aaf9",
    "image_key": "observation.images.image",
    # Logging
    "project_name": "covt-training",
    "save_dir": "./output_covt_paligemma"
}

def setup_processors_and_tokens():
    paligemma_processor = PaliGemmaProcessor.from_pretrained(CONFIG["paligemma_model_path"], use_fast=True)
    dino_processor = AutoImageProcessor.from_pretrained(CONFIG["dino_model_path"], use_fast=True)

    special_tokens = [f"<vis_{i}>" for i in range(CONFIG["num_vis_tokens"])]
    num_added = paligemma_processor.tokenizer.add_tokens(special_tokens)
    special_tokens_str = "".join(special_tokens)
    special_token_ids = paligemma_processor.tokenizer.convert_tokens_to_ids(special_tokens)
    target_ids_tensor = torch.tensor(special_token_ids, dtype=torch.long)

    return paligemma_processor, dino_processor, special_tokens_str, target_ids_tensor, num_added

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
    batch_size = len(batch)
    target_ids = target_ids_tensor.expand(batch_size, -1).clone()

    return {
        "input_ids": paligemma_inputs["input_ids"],
        "attention_mask": paligemma_inputs["attention_mask"],
        "pixel_values_paligemma": paligemma_inputs.pixel_values.to(torch.bfloat16),
        "pixel_values_dino": dino_inputs.pixel_values.to(torch.bfloat16),
        "target_token_ids": target_ids
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

    (
        paligemma_processor,
        dino_processor,
        special_tokens_str,
        target_ids_tensor,
        num_added,
    ) = setup_processors_and_tokens()
    
    if accelerator.is_main_process:
        print(f"Added {num_added} special tokens: {special_tokens_str}")
        print(f"Target IDs: {target_ids_tensor}")
    
    with accelerator.main_process_first():
        model = COVT_DINO(
            PALIGEMMA_MODEL_PATH=CONFIG["paligemma_model_path"],
            DINO_MODEL_PATH=CONFIG["dino_model_path"],
            DTYPE=torch.bfloat16,
            num_vis_tokens=CONFIG["num_vis_tokens"]
        )
        model.resize_token_embeddings(len(paligemma_processor.tokenizer))

    if accelerator.is_main_process:
        print(f"Loading LeRobot Dataset: {CONFIG['dataset_path']}...")

    dataset = LeRobotDataset(CONFIG["dataset_path"])
    
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

            if accelerator.sync_gradients:
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

def smoke_test_four_gpu(max_batches=2, test_batch_size=4):
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with=None
    )

    if accelerator.is_main_process:
        print(f"--- Smoke test on {accelerator.num_processes} processes ---")
        if accelerator.num_processes != 4:
            print("警告：当前进程数不是4，请确认使用四卡运行。")

    (
        paligemma_processor,
        dino_processor,
        special_tokens_str,
        target_ids_tensor,
        num_added,
    ) = setup_processors_and_tokens()

    if accelerator.is_main_process:
        print(f"已添加 {num_added} 个可视化特征token: {special_tokens_str}")
        print(f"目标token ID: {target_ids_tensor}")

    with accelerator.main_process_first():
        model = COVT_DINO(
            PALIGEMMA_MODEL_PATH=CONFIG["paligemma_model_path"],
            DINO_MODEL_PATH=CONFIG["dino_model_path"],
            DTYPE=torch.bfloat16,
            num_vis_tokens=CONFIG["num_vis_tokens"]
        )
        model.resize_token_embeddings(len(paligemma_processor.tokenizer))

    dataset = LeRobotDataset(CONFIG["dataset_path"])
    per_device_bs = max(1, min(test_batch_size, CONFIG["batch_size"]))
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_bs,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=lambda b: collate_fn_droid(
            b,
            paligemma_processor,
            dino_processor,
            special_tokens_str,
            target_ids_tensor
        )
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            loss = model(**batch)
            accelerator.print(f"[step {step}] loss={loss.item():.4f}")
            if step + 1 >= max_batches:
                break

    accelerator.wait_for_everyone()
    accelerator.print("四卡数据加载与前向计算烟雾测试完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COVT DINO Train/Test Entry")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="运行四卡数据加载与模型前向烟雾测试。"
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=2,
        help="烟雾测试中执行的批次数。"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="烟雾测试时每个设备的batch size。"
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test_four_gpu(
            max_batches=args.max_test_batches,
            test_batch_size=args.test_batch_size
        )
    else:
        main()
    