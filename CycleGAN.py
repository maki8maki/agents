import itertools
import os
from typing import Dict, List, Optional

import torch as th

from .cyclegan.models.networks import GANLoss, define_D, define_G
from .cyclegan.util.image_pool import ImagePool
from .utils import get_scheduler


# Reffered to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
class CycleGAN:
    def __init__(
        self,
        input_channel: int = 3,
        output_channel: int = 3,
        ndf: int = 64,
        ngf: int = 64,
        netD: str = "basic",
        netG: str = "resnet_9blocks",
        n_layers_D: int = 3,
        norm: str = "instance",
        no_dropout: bool = True,
        init_type: str = "normal",
        init_gain: float = 0.02,
        beta1: float = 0.5,
        lr: float = 0.0002,
        gan_mode: str = "lsgan",
        pool_size: int = 50,
        lambda_A: float = 10,
        lambda_B: float = 10,
        lambda_identity: float = 0.5,
        direction: str = "A2B",
        lr_policy: str = "linear",
        lr_decay_iters: int = 50,
        is_train: bool = True,
        device: str = "cpu",
    ) -> None:
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity
        self.direction = direction
        self.device = device

        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]

        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if is_train and lambda_identity > 0.0:
            assert input_channel == output_channel
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")
        self.visual_names = visual_names_A + visual_names_B

        if is_train:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:
            self.model_names = ["G_A", "G_B"]

        gpu_ids = "0" if device == "cuda" else "-1"
        self.netG_A: th.nn.Module = define_G(
            input_nc=input_channel,
            output_nc=output_channel,
            ngf=ngf,
            netG=netG,
            norm=norm,
            no_dropout=not no_dropout,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
        )
        self.netG_B: th.nn.Module = define_G(
            input_nc=input_channel,
            output_nc=output_channel,
            ngf=ngf,
            netG=netG,
            norm=norm,
            no_dropout=not no_dropout,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
        )

        if is_train:
            self.netD_A: th.nn.Module = define_D(
                output_nc=output_channel,
                ndf=ndf,
                netD=netD,
                n_layers_D=n_layers_D,
                norm=norm,
                init_type=init_type,
                init_gain=init_gain,
                gpu_ids=gpu_ids,
            )
            self.netD_B: th.nn.Module = define_D(
                output_nc=output_channel,
                ndf=ndf,
                netD=netD,
                n_layers_D=n_layers_D,
                norm=norm,
                init_type=init_type,
                init_gain=init_gain,
                gpu_ids=gpu_ids,
            )

            self.fake_A_pool = ImagePool(pool_size)
            self.fake_B_pool = ImagePool(pool_size)

            self.criterionGAN = GANLoss(gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = th.nn.L1Loss()
            self.criterionIdt = th.nn.L1Loss()

            self.optimizers = []
            self.optimizer_G = th.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=(beta1, 0.999)
            )
            self.optimizer_D = th.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=(beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.policy_kwargs = {
            "lr_policy": lr_policy,
            "lr_decay_iters": lr_decay_iters,
        }

    def setup(self, n_epochs: int = 100, n_epochs_decay: Optional[int] = None):
        if n_epochs_decay is None:
            n_epochs_decay = n_epochs
        self.policy_kwargs.update([("n_epochs", n_epochs), ("n_epochs_decay", n_epochs_decay)])

        self.schedulers = [
            get_scheduler(optimizer, lr_policy=self.policy_kwargs["lr_policy"], policy_kwargs=self.policy_kwargs)
            for optimizer in self.optimizers
        ]

    def set_input(self, input: Dict[str, th.Tensor]):
        self.real_A = input["A" if self.direction == "A2B" else "B"].to(self.device)
        self.real_B = input["B" if self.direction == "A2B" else "A"].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_A)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD: th.nn.Module, real: th.Tensor, fake: th.Tensor) -> th.Tensor:
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = (
            self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        )

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def set_requires_grad(self, nets: List[th.nn.Module], requires_grad=False):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net: th.nn.Module = getattr(self, "net" + name)
                net.eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net: th.nn.Module = getattr(self, "net" + name)
                net.train()

    def save(self, path):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{name}.pth"
                save_path = os.path.join(path, save_filename)
                net: th.nn.Module = getattr(self, "net" + name)

                th.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict: dict, module: th.nn.Module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (key == "num_batches_tracked"):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load(self, path):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{name}.pth"
                load_path = os.path.join(path, load_filename)
                net: th.nn.Module = getattr(self, "net" + name)
                if isinstance(net, th.nn.DataParallel):
                    net = net.module
                state_dict: dict = th.load(load_path, map_location=self.device)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

    def to(self, device):
        for name in self.model_names:
            if isinstance(name, str):
                net: th.nn.Module = getattr(self, "net" + name)

                net.to(device)
        self.criterionGAN.to(device)

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self._get_name() + "()"
