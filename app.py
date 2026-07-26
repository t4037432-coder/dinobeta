# ==============================================================================
# [CHẠY FLUX VỚI TOKEN CỦA BRO]
# ==============================================================================
from IPython.display import display
import torch
import gc

# Token của bro
hf_token = "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"

print("⏳ Đang dọn dẹp bộ nhớ...")
if 'pipe' in locals():
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

# Dùng bản schnell công khai, mượt trên T4
model_id = "black-forest-labs/FLUX.1-schnell"

print(f"⏳ Đang tải mô hình bằng token của bro...")

try:
    from diffusers import FluxPipeline

    # Tải pipeline có xác thực token
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        low_cpu_mem_usage=True
    )

    # Tắt kiểm tra an toàn (tránh lỗi ảnh đen)
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    class DummySafetyChecker(StableDiffusionSafetyChecker):
        def __init__(self, **kwargs): pass
        def forward(self, images, clip_input): return images, [False] * len(images)
    pipe.safety_checker = DummySafetyChecker()

    # Phân bổ CPU tiết kiệm VRAM cho T4
    pipe.enable_model_cpu_offload()

    print("✅ Tải thành công! Đang tiến hành vẽ...")

except Exception as e:
    print(f"❌ Lỗi: {e}")
    raise SystemExit("Dừng chương trình.")

# Ý tưởng của bro (bro có thể sửa lại tiếng Việt theo ý thích ở đây)
prompt_text = "A close-up portrait of a beautiful Vietnamese woman in traditional white ao dai, natural smile, natural lighting, highly detailed face, sharp focus, 8k"

# Tạo ảnh
with torch.inference_mode():
    image = pipe(
        prompt=prompt_text,
        height=768,
        width=768,
        guidance_scale=0.0,
        num_inference_steps=4,
    ).images[0]

print("✅ Đã vẽ xong! Thành quả:")
display(image)# ==============================================================================
# [CHẠY FLUX VỚI TOKEN CỦA BRO]
# ==============================================================================
from IPython.display import display
import torch
import gc

# Token của bro
hf_token = "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"

print("⏳ Đang dọn dẹp bộ nhớ...")
if 'pipe' in locals():
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

# Dùng bản schnell công khai, mượt trên T4
model_id = "black-forest-labs/FLUX.1-schnell"

print(f"⏳ Đang tải mô hình bằng token của bro...")

try:
    from diffusers import FluxPipeline

    # Tải pipeline có xác thực token
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        low_cpu_mem_usage=True
    )

    # Tắt kiểm tra an toàn (tránh lỗi ảnh đen)
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    class DummySafetyChecker(StableDiffusionSafetyChecker):
        def __init__(self, **kwargs): pass
        def forward(self, images, clip_input): return images, [False] * len(images)
    pipe.safety_checker = DummySafetyChecker()

    # Phân bổ CPU tiết kiệm VRAM cho T4
    pipe.enable_model_cpu_offload()

    print("✅ Tải thành công! Đang tiến hành vẽ...")

except Exception as e:
    print(f"❌ Lỗi: {e}")
    raise SystemExit("Dừng chương trình.")

# Ý tưởng của bro (bro có thể sửa lại tiếng Việt theo ý thích ở đây)
prompt_text = "A close-up portrait of a beautiful Vietnamese woman in traditional white ao dai, natural smile, natural lighting, highly detailed face, sharp focus, 8k"

# Tạo ảnh
with torch.inference_mode():
    image = pipe(
        prompt=prompt_text,
        height=768,
        width=768,
        guidance_scale=0.0,
        num_inference_steps=4,
    ).images[0]

print("✅ Đã vẽ xong! Thành quả:")
display(image)
