from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModel
from PIL import Image
import torch
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import os
from typing import List 

class IPAdapterProjectionLayer(torch.nn.Module):
    def __init__(self, clip_dim=512, embedd_dim=768, context_size=4):
        super().__init__()
        self.projection = nn.Linear(clip_dim, context_size * embedd_dim, bias=False)
        self.norm = nn.LayerNorm(embedd_dim)
        self.clip_dim = clip_dim
        self.embedd_dim = embedd_dim
        self.context_size = context_size

    def forward(self, x):
        x = self.projection(x)
        x = x.view(x.size(0), self.context_size, self.embedd_dim)
        x = self.norm(x)
        return x
    

class IPAdapterAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, cross_attention_dim, hidden_size, context_size=4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()
        self.ip_to_k = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.ip_to_v = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.contet_size = context_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # deprecate("scale", "1.0.0", deprecation_message)
        
        encoder_hidden_states_image =  encoder_hidden_states[:, -self.contet_size:]
        encoder_hidden_states_text =  encoder_hidden_states[:, :-self.contet_size]
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states_text is None else encoder_hidden_states_text.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states_text is None:
            encoder_hidden_states_text = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states_text = attn.norm_encoder_hidden_states(encoder_hidden_states_text)
            encoder_hidden_states_image = attn.norm_encoder_hidden_states(encoder_hidden_states_image)

        key = attn.to_k(encoder_hidden_states_text)
        key_ip = self.ip_to_k(encoder_hidden_states_image)
        value = attn.to_v(encoder_hidden_states_text)
        value_ip = self.ip_to_v(encoder_hidden_states_image)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_ip = key_ip.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_ip = value_ip.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            key_ip = attn.norm_k(key_ip)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states + F.scaled_dot_product_attention(query, key_ip, value_ip, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def replace_cross_attention_processors(unet):
    cross_attn_processors = {}    
    unet_state_dict = unet.state_dict()
    for name, processor in unet.attn_processors.items():
        if name.endswith('attn2.processor'): # these are the cross-attention processors
            # Retrieve size of the hidden states for the cross-attention processor
            # unet.config.block_out_channels is [320, 640, 1280, 1280], corresponding to resolutions 32,16,8,4 (I think)
            # but remember to reverse them for the up blocks
            if name.startswith('mid_block'):
                # https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/unets/unet_2d_condition.py#L388
                hidden_size = unet.config.block_out_channels[-1]
            else:
                block_id = int(name.split(".")[1])
                if name.startswith('down_blocks'):
                    hidden_size = unet.config.block_out_channels[block_id]
                elif name.startswith('up_blocks'):
                    hidden_size = unet.config.block_out_channels[-block_id-1] # reverse order
                    # unet.config.cross_attention_dim is the input size of the to_k and to_v linear layers
                    # hidden size the the input/output size of the to_q linear layer

            layer_name = name.split(".processor")[0]
            cross_attn_processors[name] = IPAdapterAttnProcessor2_0(
                cross_attention_dim=unet.config.cross_attention_dim, 
                hidden_size=hidden_size,
                context_size=4,
            )
            load_dict = {
                "ip_to_k.weight": unet_state_dict[f"{layer_name}.to_k.weight"],
                "ip_to_v.weight": unet_state_dict[f"{layer_name}.to_v.weight"],
            }
            cross_attn_processors[name].load_state_dict(load_dict)
        else:
            cross_attn_processors[name] = processor
    unet.set_attn_processor(cross_attn_processors)


def print_trainable_parameters(unet):
    # Print the names of trainable weights of the UNet
    total_params = 0
    print("="*105)
    print(f"{'# Parameters':<15} {'Shape':<15} {'Parameter Name':<60}")
    print("="*105)
    for name, param in unet.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            shape_str = str(tuple(param.shape)).replace("torch.Size", "")
            print(f"{num_params:<15} {shape_str:<15} {name:<60}")
    print("="*105)
    print(f"Total Trainable Parameters: {total_params} ({total_params / 1e6:.2f}M)")
    print("="*105)


class NarutoDataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset("lambdalabs/naruto-blip-captions", cache_dir="F:/nviolant/hugging_face_cache")["train"]
        self.transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        # Add other transformations as required
])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        caption = item['text']
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "images": image,
            "prompt": caption
        }
    
class IPAdapterPipeline:
    def __init__(self, sd_pipe, ip_adapter_layer, clip_image_encoder):
        self.sd_pipe = sd_pipe
        self.ip_adapter_layer = ip_adapter_layer
        self.clip_image_encoder = clip_image_encoder

    def generate(
        self,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=50,
        **kwargs,
    ):
        num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(clip_image_embeds)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.sd_pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                device="cuda"
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)


        images = self.sd_pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **kwargs,
        ).images

        return images  
    
    @torch.inference_mode()
    def get_image_embeds(self, clip_image_embeds=None):
        image_prompt_embeds = self.ip_adapter_layer(clip_image_embeds)
        uncond_image_prompt_embeds = self.ip_adapter_layer(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the Stable Diffusion model from pretrained
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                    cache_dir="F:/nviolant/hugging_face_cache")

    # Separate the components
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    unet = pipeline.unet
    scheduler = pipeline.scheduler
    safety_checker = pipeline.safety_checker
    image_processor = pipeline.feature_extractor

    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

    dataset = NarutoDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Freeze the UNet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Replace the cross-attention processors with the custom one
    replace_cross_attention_processors(unet)

    print_trainable_parameters(unet)

    trainable_parameters = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-4)

    num_steps = 2000
    step_count = 0

    # Move models to the GPU
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    safety_checker.to(device)

    unet.train()

    progress_bar = tqdm(total=num_steps)
    ip_adapter_layer = IPAdapterProjectionLayer(512, 768, 4).to(device)

    ip_pipeline = IPAdapterPipeline(pipeline, ip_adapter_layer, clip_image_encoder)



    while step_count < num_steps:
        for batch in dataloader:
            if step_count >= num_steps:
                break

            # Encode the clean images into latents
            clean_images = batch["images"].cuda()
            with torch.no_grad():
                clean_latents = vae.encode(clean_images).latent_dist.sample() * vae.config.scaling_factor
                image_tokens = clip_image_encoder(image_processor(clean_images*0.5 + 0.5, return_tensors="pt").pixel_values).image_embeds
                image_embeds = ip_adapter_layer(image_tokens.to(device))
            # Sample gaussian noise
            noise = torch.randn(clean_latents.shape, device=clean_latents.device)
            
            # Generate random timesteps for noise sampling
            bs = clean_latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=clean_latents.device, dtype=torch.int64)

            # Add noise to the clean latents
            noisy_images = scheduler.add_noise(clean_latents, noise, timesteps)

            # Encode the text prompt
            prompt = batch["prompt"]
            text_embeds = text_encoder(tokenizer(prompt, 
                                                           max_length=tokenizer.model_max_length,
                                                           padding="max_length",
                                                           truncation=True,return_tensors="pt"
                                                           ).input_ids.to(device)
                                                )[0]
            
            # Encode the image prompt
            encoder_hidden_states = torch.cat([text_embeds, image_embeds], dim=1)

            # Predict the noise                
            noise_pred = unet(noisy_images, timesteps, encoder_hidden_states, return_dict=False)[0]
            
            # Loss computation
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            step_count += 1
            progress_bar.update(1)

    progress_bar.close()
    
    torch.save(unet.state_dict(), os.path.join("F:/nviolant/playground", "unet_ip_adapter.pth"))
    prompt = ["a man. high quality"]
    generated_image = ip_pipeline.generate(prompt=prompt, clip_image_embeds=image_tokens.to(device)[0:1])[0]
    generated_image.show()


        