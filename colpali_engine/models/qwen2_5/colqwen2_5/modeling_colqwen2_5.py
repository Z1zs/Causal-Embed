from typing import ClassVar
from tqdm import tqdm

import torch
from torch import nn
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLModel


class ColQwen2_5(Qwen2_5_VLModel):  # noqa: N801
    """
    ColQwen2.5 model implementation, following the achitecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen2.5-VL backbone.

    Args:
        config (Qwen2.5VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2_5_VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size



class CausalQwen2_5(Qwen2_5_VLModel):  # noqa: N801
    """
    ColQwen2.5 model implementation, following the achitecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen2.5-VL backbone.

    Args:
        config (Qwen2.5VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2_5_VLConfig,doc_token_num: int=32,query_token_num:int=16, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.latent_token_id = None
        self._checkpoint_conversion_mapping=super()._checkpoint_conversion_mapping
        print(self._checkpoint_conversion_mapping)
        self.doc_token_num=doc_token_num
        self.query_token_num=query_token_num
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )
            num_latent_tokens=self.doc_token_num
        else:
            num_latent_tokens=self.query_token_num

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        # Get initial hidden states
        outputs = super().forward(*args, **kwargs, use_cache=True, output_hidden_states=True, return_dict=True)
        past_key_values = outputs.past_key_values
        last_hidden_states = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        latent_token_embeds=[last_hidden_states[:, -1:, :]]

        batch_size, seq_length, hidden_size = last_hidden_states.shape
        input_ids=torch.full((batch_size, 1), self.latent_token_id, device=last_hidden_states.device, dtype=torch.long) # (batch_size, 1)
            
        # Autoregressive generation of latent tokens
        for i in range(num_latent_tokens-1):
            # Forward pass with latent tokens
            outputs = super().forward(
                    *args,
                    input_ids=input_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                    past_key_values=past_key_values
                )
                
            past_key_values = outputs.past_key_values
            last_hidden_states = outputs.last_hidden_state  # (batch_size, 1, hidden_size)

            latent_token_embeds.append(last_hidden_states[:, -1:, :])  

        latent_token_embeds = torch.cat(latent_token_embeds, dim=1)  # (batch_size, num_latent_tokens, hidden_size)
        proj = latent_token_embeds

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, num_latent_tokens, dim)
        
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
