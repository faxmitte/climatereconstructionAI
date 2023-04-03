import torch
from torch import nn
from climatereconstructionai import config as cfg
import numpy as np

class DM(nn.Module):


    def __init__(self, n_steps: int, gmodel:torch.Tensor=None, min_val:float=1e-4, max_val=1e-3, use_sigma=True, use_mu=True):
        super().__init__()
        self.n_steps = n_steps

        self.register_buffer("beta", torch.linspace(min_val, max_val, n_steps))
        alpha =  1 - self.beta
        self.register_buffer("alpha", alpha.to(cfg.device))
        self.register_buffer("alpha_bar", alpha.cumprod(0).to(device=cfg.device))

       # alphas_cumprod = np.cumprod(self.alpha, axis=0)
       # alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
       # self.sqrt_alphas_cumprod_prev = np.sqrt(
       #     np.append(1., alphas_cumprod))
        
        self.device = cfg.device
        self.gmodel = gmodel
        self.use_sigma = use_sigma
        self.use_mu = use_mu
  
    @torch.no_grad()
    def generate_moments(self, image, mask):
        
        if self.gmodel is not None:
            output = self.gmodel(image, mask)
            if self.use_mu:
                mu = output[:,:,0].unsqueeze(dim=2)
            else:
                mu = image
            if self.use_sigma:
                sigma = output[:,:,1].unsqueeze(dim=2)
       #         sigma /= sigma.mean()
            else:
                sigma = torch.ones_like(image, device=self.device)
        else:
            mu = image
            sigma = torch.ones_like(image, device=self.device)

        return mu, sigma

    def diffusion_loss(self, model: nn.Module, inp) -> torch.Tensor:

        image, mask, gt = inp

        mu, sigma = self.generate_moments(image, mask)

        #inserted
        batch_size = mu.shape[0]

        t = torch.randint(0,self.n_steps-1,(batch_size,), device=self.device)

       # shape = gt.shape

        #alpha_bar_t = torch.zeros(tuple([sh if k==0 else 1 for k,sh in enumerate(gt.shape)]))

        alpha_bar_t = self.alpha_bar[t][:,None,None,None,None]

        eps = sigma*torch.randn_like((mu), device=self.device)

        x_noisy = torch.sqrt(alpha_bar_t) * gt + torch.sqrt(1 - alpha_bar_t) * eps

        input = torch.concat((mu,x_noisy),dim=2)
        mask = torch.concat((mask,mask),dim=2)

        eps_pred = model(input ,mask)

        loss = {'total': nn.MSELoss()(eps_pred, eps)} 

        return loss, eps, x_noisy
    

    def sample(self, model: nn.Module, inp):

        image, mask = inp

        mask_3 = torch.concat((mask, mask),dim=2)

        mu, sigma = self.generate_moments(image, mask)

        n_samples=mu.shape[0]

        with torch.no_grad():

            # start off with an intial random ensemble of particles
            x = sigma*torch.randn_like((mu), device=self.device)

            # the number of steps is fixed before beginning training. unfortunately.
            for t in reversed(range(self.n_steps)):
                # apply the same variance to all particles in the ensemble equally.
                a = self.alpha[t].repeat(n_samples)[:,None,None,None,None]
                abar = self.alpha_bar[t].repeat(n_samples)[:,None,None,None,None]

                input = torch.concat((mu, x),dim=2)
                
                # deterministic trajectory. eps_theta is similar to the Force on the particle
                eps_theta = model(input, mask_3)

                x_mean = (x - eps_theta * (1 - a) / torch.sqrt(1 - abar)) / torch.sqrt(a)

                sigma_t = torch.sqrt(1 - self.alpha[t])

                z = sigma * torch.randn_like(x, device=self.device)
                x = x_mean + sigma_t * z 

            return x_mean  
