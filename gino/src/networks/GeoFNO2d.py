import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(
            u_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        factor2 = self.compl_mul2d(
            u_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                s1,
                s2 // 2 + 1,
                dtype=torch.cfloat,
                device=u.device,
            )
            out_ft[:, :, : self.modes1, : self.modes2] = factor1
            out_ft[:, :, -self.modes1 :, : self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes1, step=1),
                    torch.arange(start=-(self.modes1), end=0, step=1),
                ),
                0,
            )
            .reshape(m1, 1)
            .repeat(1, m2)
            .to(device)
        )
        k_x2 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes2, step=1),
                    torch.arange(start=-(self.modes2 - 1), end=0, step=1),
                ),
                0,
            )
            .reshape(1, m2)
            .repeat(m1, 1)
            .to(device)
        )

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[..., 0].view(-1), k_x1.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K2 = torch.outer(x[..., 1].view(-1), k_x2.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes1, step=1),
                    torch.arange(start=-(self.modes1), end=0, step=1),
                ),
                0,
            )
            .reshape(m1, 1)
            .repeat(1, m2)
            .to(device)
        )
        k_x2 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes2, step=1),
                    torch.arange(start=-(self.modes2 - 1), end=0, step=1),
                ),
                0,
            )
            .reshape(1, m2)
            .repeat(m1, 1)
            .to(device)
        )

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:, :, 0].view(-1), k_x1.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K2 = torch.outer(x[:, :, 1].view(-1), k_x2.view(-1)).reshape(
            batchsize, N, m1, m2
        )
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y


class IPHI(nn.Module):
    def __init__(self, width=36):
        super(IPHI, self).__init__()

        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc_sphere = nn.Linear(3, self.width)
        self.fc0 = nn.Linear(3, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(4 * self.width, 4 * self.width)
        self.fc1 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc3 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc4 = nn.Linear(4 * self.width, 2)
        self.activation = F.gelu

        self.B = np.pi * torch.pow(
            2, torch.arange(0, self.width // 3, dtype=torch.float, device="cuda")
        ).reshape(1, 1, 1, self.width // 3)

    def forward(self, x, code=None, is_identity=False):
        if is_identity:
            return x

        # x (batch, N_grid, 3)
        # code (batch, N_features)
        b, n, d = x.shape[0], x.shape[1], x.shape[2]

        # re-center
        x_mean = torch.mean(x, dim=1).reshape(b, 1, d)
        x_std = torch.std(x, dim=1).reshape(b, 1, d)  #
        x_centered = x

        # spherical coordinate systems
        radius = torch.norm(x_centered, dim=-1)
        phi = torch.arccos(x_centered[..., 2] / radius)
        psi = torch.sign(x_centered[..., 1]) * torch.arccos(
            x[..., 0] / torch.norm(x[..., :2], dim=-1)
        )
        x_sphere = torch.stack([radius, phi, psi], dim=-1)
        xd = self.fc_sphere(x_sphere)

        # sin features from SIREN
        x_sin = torch.sin(self.B * x.view(b, n, d, 1)).view(b, n, d * self.width // 3)
        x_cos = torch.cos(self.B * x.view(b, n, d, 1)).view(b, n, d * self.width // 3)
        xc = self.fc0(x)
        xc = torch.cat([xd, xc, x_sin, x_cos], dim=-1).reshape(b, n, 4 * self.width)

        xd = self.fc_no_code(xc)
        # xd = self.fc1(xd)
        # xd = self.activation(xd)
        # xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        # return x[..., 0:3:2] + x[..., 0:3:2] * xd
        return x_sphere[..., 1:3] + x_sphere[..., 1:3] * xd


class GeoFNO2d(BaseModel):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width,
        in_channels=3,
        out_channels=1,
        is_mesh=True,
        s=64,
        iphi=None,
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s
        self.s2 = s
        # self.s3 = s3

        self.fc0 = nn.Linear(
            in_channels, self.width
        )  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, s, s
        )
        self.conv1 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv2 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv3 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv4 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, s, s
        )
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(3, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.iphi = IPHI()

    def forward(self, u, code=None, x_in=None, x_out=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        # u = self.fc_3dto2d(u)

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(
            0, 3, 1, 2
        )

        u = self.fc0(u)
        u = u.permute(0, 2, 1)
        u0 = u

        uc1 = self.conv0(u, x_in=x_in, iphi=self.iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=self.iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3 + u0

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def data_dict_to_input(self, data_dict, **kwargs):
        vert = data_dict["vertices"].to(self.device)
        return vert

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None):
        vert = self.data_dict_to_input(data_dict)
        pred_out = self(vert)
        gt_out = data_dict["pressure"].to(self.device)
        out_dict = {"l2": loss_fn(pred_out, gt_out)}
        if decode_fn is not None:
            pred_out = decode_fn(pred_out)
            gt_out = decode_fn(gt_out)
            out_dict["l2_decoded"] = loss_fn(pred_out, gt_out)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        vert = self.data_dict_to_input(data_dict)
        pressure = self(vert)
        if loss_fn is None:
            loss_fn = self.loss
        return {"loss": loss_fn(pressure, data_dict["pressure"].to(self.device))}
