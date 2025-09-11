import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
import numpy as np
from ldm.modules.ema import LitEma
from torch.optim.lr_scheduler import LambdaLR
from packaging import version

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import torch.nn.functional as F
# from piq import NLPD

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        # PL 2.x : manual optimization for multi-optimizer training
        self.automatic_optimization = False
        self.embed_dim = embed_dim
        self.n_embed = n_embed

        # track which codebook entries were used in the current val epoch
        self.register_buffer("val_code_used", torch.zeros(self.n_embed, dtype=torch.bool), persistent=False)

        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        # metrics (expect inputs in [0,1])
        self.psnr  = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim  = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')  # or 'alex'
        self.fid   = FrechetInceptionDistance(feature=2048)

        # NLPD from PIQ (module, not torchmetrics)
        # self.nlpd = NLPD()  # moved to correct device in setup()


    # def on_fit_start(self):
    #     # ensure PIQ module is on the same device
    #     self.nlpd = self.nlpd.to(self.device)

    def _to_01(self, x):
        return (x + 1.0) / 2.0

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx):
        # manual optimization: step AE (gen) then D
        opt_list = self.optimizers()
        if isinstance(opt_list, (list, tuple)):
            opt_ae = opt_list[0]
            opt_disc = opt_list[1] if len(opt_list) > 1 else None
        else:
            opt_ae, opt_disc = opt_list, None

        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        #---------------AE/generator step -------------------
        if hasattr(self, "toggle_optimizer"):
            self.toggle_optimizer(opt_ae)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
                                            # predicted_indices=ind)
        opt_ae.zero_grad(set_to_none=True)
        self.manual_backward(aeloss)
        opt_ae.step()
        if hasattr(self, "untoggle_optimizer"):
            self.untoggle_optimizer(opt_ae)
        if isinstance(log_dict_ae, dict):
            self.log_dict(
                log_dict_ae, 
                prog_bar=False, 
                logger=True, 
                on_step=True, 
                on_epoch=True,
                batch_size=x.size(0)
                )
         
        #---------------Discriminator step (after disc_start) -------------
        disc_start = getattr(self.loss, "disc_start", 0)
        if opt_disc is not None and self.global_step >= disc_start:
            if hasattr(self, "toggle_optimizer"):
                self.toggle_optimizer(opt_disc)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            opt_disc.zero_grad(set_to_none=True)
            self.manual_backward(discloss)
            opt_disc.step()
            if hasattr(self, "untoggle_optimizer"):
                self.untoggle_optimizer(opt_disc)
            if isinstance(log_dict_disc, dict):
                self.log_dict(
                    log_dict_disc, 
                    prog_bar=False, 
                    logger=True, 
                    on_step=True, 
                    on_epoch=True,
                    batch_size=x.size(0))
            return aeloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        # predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            # predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        # self.log(f"val{suffix}/rec_loss", rec_loss,
        #            prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log(f"val{suffix}/aeloss", aeloss,
        #            prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # if version.parse(pl.__version__) >= version.parse('1.4.0'):
        #     del log_dict_ae[f"val{suffix}/rec_loss"]
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        bs = x.size(0)
        self.log(f"val{suffix}/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=bs)
        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=bs)
        # avoid duplicate logging of the same key
        log_dict_ae.pop(f"val{suffix}/rec_loss", None)
        self.log_dict(log_dict_ae, on_step=False, on_epoch=True, batch_size=bs)
        self.log_dict(log_dict_disc, on_step=False, on_epoch=True, batch_size=bs)
        # ... you already have:
        # ---- Metrics block (add this) ----
        bs = x.size(0)
        x01    = self._to_01(x).clamp(0,1)
        xrec01 = self._to_01(xrec).clamp(0,1)

        # PSNR / SSIM / LPIPS (batched, averaged internally)
        psnr_val  = self.psnr(xrec01, x01)
        ssim_val  = self.ssim(xrec01, x01)
        lpips_val = self.lpips(xrec01, x01)

        # NLPD (PIQ) expects float in [0,1]
        # nlpd_val = self.nlpd(xrec01, x01)

        # rFID accumulator (real vs recon)
        # torchmetrics FID accepts uint8 in [0,255] with shape NCHW
        x_fid    = (x01    * 255).to(torch.uint8)
        xrec_fid = (xrec01 * 255).to(torch.uint8)
        self.fid.update(x_fid,    real=True)
        self.fid.update(xrec_fid, real=False)

        # log per-step, aggregate to epoch
        self.log("val/psnr",  psnr_val,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bs)
        self.log("val/ssim",  ssim_val,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bs)
        self.log("val/lpips", lpips_val, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True, batch_size=bs)
        # self.log("val/nlpd",  nlpd_val,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bs)
        # ---- end metrics block ----

        # --- accumulate code usage ---
        # 'ind' is expected shape [B, H, W] with code indices in [0, n_embed-1]
        used = ind.detach().reshape(-1)
        used_unique = torch.unique(used)
        # update bitmap on CPU to avoid GPU host syncs when large
        self.val_code_used[used_unique.cpu()] = True

        return self.log_dict
    
    def on_validation_epoch_start(self):
        # reset usage bitmap
        if hasattr(self, "val_code_used"):
            self.val_code_used.zero_()

    def on_validation_epoch_end(self):
        # finalize FID over whatever was accumulated this val run
        try:
            rfid_val = self.fid.compute()
            self.log("val/rfid", rfid_val, prog_bar=True, sync_dist=True)
        finally:
            self.fid.reset()

        # --- DDP-safe merge of usage bitmaps ---
        local_mask = self.val_code_used.to(self.device).float()  # 0/1 as float for all_gather
        try:
            gathered = self.all_gather(local_mask)  # [world_size, n_embed] or [n_embed] if single proc
            if gathered.ndim == 2:
                merged = (gathered.sum(dim=0) > 0)
            else:
                merged = (gathered > 0)
        except Exception:
            # fallback single-process
            merged = (local_mask > 0)

        used_count = merged.sum().item()
        usage_pct = used_count / float(self.n_embed)

        # log both
        self.log("val/codebook_used_count", used_count, prog_bar=False, sync_dist=False)
        self.log("val/codebook_usage_pct", usage_pct, prog_bar=True, sync_dist=False)

        # (optional) keep the merged view around for debugging
        self.val_code_used = merged.to(self.val_code_used.device)

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        # PL 2.x : manual optimization for multi-optimizer training
        self.automatic_optimization = False
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        # manual optimization: step AE (gen) then D
        opt_list = self.optimizers()
        if isinstance(opt_list, (list, tuple)):
            opt_ae = opt_list[0]
            opt_disc = opt_list[1] if len(opt_list) > 1 else None
        else:
            opt_ae, opt_disc = opt_list, None

        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        #---------------AE/generator step -------------------
        if hasattr(self, "toggle_optimizer"):
            self.toggle_optimizer(opt_ae)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        opt_ae.zero_grad(set_to_none=True)
        self.manual_backward(aeloss)
        opt_ae.step()
        if hasattr(self, "untoggle_optimizer"):
            self.untoggle_optimizer(opt_ae)

        bs = inputs.size(0)
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=bs)
        # self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if isinstance(log_dict_ae, dict):
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=bs)
            # self.log_dict(
            #     log_dict_ae, 
            #     prog_bar=False, 
            #     logger=True, 
            #     on_step=True, 
            #     on_epoch=False,
            #     batch_size=inputs.size(0)
            #     )
        
        #---------------Discriminator step (after disc_start) -------------
        disc_start = getattr(self.loss, "disc_start", 0)
        if opt_disc is not None and self.global_step >= disc_start:
            if hasattr(self, "toggle_optimizer"):
                self.toggle_optimizer(opt_disc)
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            opt_disc.zero_grad(set_to_none=True)
            self.manual_backward(discloss)
            opt_disc.step()
            if hasattr(self, "untoggle_optimizer"):
                self.untoggle_optimizer(opt_disc)
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            if isinstance(log_dict_disc, dict):
                self.log_dict(
                    log_dict_disc, 
                    prog_bar=False, 
                    logger=True, 
                    on_step=True, 
                    on_epoch=True,
                    batch_size=inputs.size(0))
            return aeloss
        
    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     inputs = self.get_input(batch, self.image_key)
    #     reconstructions, posterior = self(inputs)

    #     if optimizer_idx == 0:
    #         # train encoder+decoder+logvar
    #         aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="train")
    #         self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    #         return aeloss

    #     if optimizer_idx == 1:
    #         # train the discriminator
    #         discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
    #                                             last_layer=self.get_last_layer(), split="train")

    #         self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    #         return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        # self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        bs = inputs.size(0)
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"],
                 on_step=False, on_epoch=True, batch_size=bs)
        log_dict_ae.pop("val/rec_loss", None)   # prevent duplicate metric name
        self.log_dict(log_dict_ae, on_step=False, on_epoch=True, batch_size=bs)
        self.log_dict(log_dict_disc, on_step=False, on_epoch=True, batch_size=bs)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
