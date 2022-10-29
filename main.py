import os
from typing import Tuple, Union

import pytorch_lightning as pl

import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import GPT2Tokenizer

import wandb
from dataset import *
from image_caption_style_token import ImageCaptionStyleToken

EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 8
MAX_TEXT_LENGTH = 128
TOP_K = 1000
TOP_P = 0.95
DATASET_PATH = os.getenv("DATASET_PATH", "/home/ray1422/data/ins_dataset/Influencer_brand_dataset")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    wandb.init(project="ins-feed-gen", entity="ray1422")
    wandb.config = {
        "learning_rate": LR,
        "reference_caption": False
    }
    wandb_logger = WandbLogger()
    lightning_module = LightningModule()

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gpus=torch.cuda.device_count(),
        gradient_clip_val=1.0,
        precision=16,
        num_sanity_val_steps=0,
        logger=wandb_logger
    )
    train_ds, val_ds, test_ds = generate_dataset(DATASET_PATH, [.6, .2], debug=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    valid_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    trainer.fit(lightning_module, train_dl, valid_dl)
    wandb.finish()


def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens


class LightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ImageCaptionStyleToken.from_encoder_decoder_pretrained("google/vit-base-patch16-224-in21k",
                                                                            "distilgpt2")
        self.model = self.model.to(device)
        self.lr = LR
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = -1

    def common_step(self, batch: Tuple[any, any, any]) -> torch.FloatTensor:
        pixel_values, ref_cap, cap = batch
        bs, _, ch, h, w = pixel_values.size()
        pixel_values = torch.reshape(pixel_values, (bs, ch, h, w)).to(device)
        ref_cap = self.tokenizer(ref_cap, max_length=MAX_TEXT_LENGTH, truncation=True, padding="max_length",
                                 return_tensors="pt")
        cap = self.tokenizer(cap, max_length=MAX_TEXT_LENGTH, truncation=True, padding="max_length",
                             return_tensors="pt")

        encoder_outputs = self.model.encoder(pixel_values=pixel_values)
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=cap["input_ids"].to(device),
            decoder_attention_mask=cap["attention_mask"].to(device),
            labels=cap["input_ids"].to(device),
            return_dict=True,
        )

        return outputs["loss"]

    def training_step(self, batch: Tuple[any, any, any], batch_idx: int) -> torch.FloatTensor:
        loss = self.common_step(batch)
        self.log(name="Training loss", value=loss, on_step=False, on_epoch=True, batch_size=BATCH_SIZE)
        return loss

    def validation_step(self, batch: Tuple[any, any, any], batch_idx: int):
        if batch_idx == 0:
            pixel_values, ref_cap, cap = batch
            bs, _, ch, h, w = pixel_values.size()
            pixel_values = torch.reshape(pixel_values, (bs, ch, h, w)).to(device)
            encoder_outputs = self.model.encoder(pixel_values=pixel_values)
            generated_sentences = self.generate_sentence_from_image(
                self.model,
                encoder_outputs,
                self.tokenizer,
                MAX_TEXT_LENGTH,
                self.device
            )
            if generated_sentences is not None:
                images = [wandb.Image(transforms.ToPILImage()(image)) for image in pixel_values]
                data = list(map(list, zip(images, cap, generated_sentences)))
                table = wandb.Table(data=data, columns=["Images", "Actual Sentence", "Generated Sentence"])
                self.logger.experiment.log({f"epoch {self.current_epoch} results": table})
            else:
                wandb.log(f"epoch {self.current_epoch} failed to generate")

        loss = self.common_step(batch)
        self.log(name="Validation loss", value=loss, on_step=False, on_epoch=True, batch_size=BATCH_SIZE)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @staticmethod
    def generate_sentence_from_image(model, encoder_outputs, tokenizer, max_text_length: int, device) -> List[str]:
        try:
            generated_so_far = torch.LongTensor([[tokenizer.bos_token_id]] * len(encoder_outputs.last_hidden_state)).to(
                device)
            with torch.no_grad():
                for _ in tqdm(range(max_text_length)):
                    attention_mask = torch.ones_like(generated_so_far)
                    decoder_out = model(
                        decoder_input_ids=generated_so_far,
                        decoder_attention_mask=attention_mask,
                        encoder_outputs=encoder_outputs
                    )

                    next_token_logits = decoder_out["logits"][:, -1, :]
                    filtered_p = top_k_top_p_filtering(next_token_logits, top_k=TOP_K, top_p=TOP_P, device=device)
                    next_token = torch.multinomial(filtered_p, num_samples=1)
                    generated_so_far = torch.cat((generated_so_far, next_token), dim=1)

            return [tokenizer.decode(coded_sentence) for coded_sentence in generated_so_far]
        except Exception as e:
            print("generate sentence failed!", e)


def top_k_top_p_filtering(
        next_token_logits: torch.FloatTensor,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
) -> torch.FloatTensor:
    if top_k is None:
        top_k = next_token_logits.shape[-1]
    if top_p is None:
        top_p = 1.0

    p, largest_p_idx = torch.softmax(next_token_logits, dim=-1).topk(top_k, dim=-1)
    cumulative_p = p.cumsum(dim=-1)
    threshold_repeated = top_p + torch.zeros((len(p), 1)).to(device)
    idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k - 1).squeeze()
    cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
    censored_p = (cumulative_p <= cutoffs[:, None]) * p
    renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)

    final_p = torch.zeros_like(next_token_logits)
    row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1, top_k).to(device)
    final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

    return final_p


if __name__ == '__main__':
    main()
