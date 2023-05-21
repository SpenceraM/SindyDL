import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.input_dim  = cfg['input_dim']
        self.latent_dim = cfg['latent_dim']
        self.widths     = cfg['widths']
        self.activation = cfg['activation']
        self.poly_order = cfg['poly_order']
        if 'include_sine' in cfg.keys():
            self.include_sine = cfg['include_sine']
        else:
            self.include_sine = False
        self.library_dim = cfg['library_dim']  # number of terms in library
        self.model_order = cfg['model_order']  # 1 or 2, related to z derivative
        self.sequential_thresholding = cfg['sequential_thresholding']

        if self.activation == 'linear':
            pass
        else:  # non-linear activation
            self.encoder = XcoderHalf(self.input_dim, self.latent_dim, self.widths, self.activation)
            self.decoder = XcoderHalf(self.latent_dim, self.input_dim, self.widths[::-1], self.activation)

        if self.model_order == 1:
            self.z_derivative_func = ZDerivativeOrder1(self.activation)
            self.sindy_library = SINDyLibraryOrder1(self.poly_order, self.include_sine)
        else:
            raise ValueError('Invalid model order')
            # self.z_derivative_func  =
            # self.sindy_library =

        # TODO: initialize with random values
        self.sindy_coefficients = torch.empty(self.library_dim, self.latent_dim, requires_grad=True)  # Tensor to hold coefficients
        torch.nn.init.constant_(self.sindy_coefficients, 1.0)

        if self.sequential_thresholding:
            # Binary tensor to mask out coefficients
            self.coefficient_mask = torch.ones((self.library_dim, self.latent_dim), requires_grad=False)

    def forward(self, x, dx, ddx=None):
        z = self.encoder(x)  # z = latent dim
        x_hat = self.decoder(z) # standard autoencoder output

        dz = self.z_derivative_func(x, dx, self.encoder)
        Theta = self.sindy_library(z)  # Theta = library dim

        if self.sequential_thresholding:
            sindy_prediction = torch.matmul(Theta, self.coefficient_mask * self.sindy_coefficients)
        else:
            sindy_prediction = torch.matmul(Theta, self.sindy_coefficients)

        if self.model_order == 1:
            dx_decoded = self.z_derivative_func(z, sindy_prediction, self.decoder)
        else:
            raise ValueError('Invalid model order')

        # Create dictionary to hold outputs
        outputs = {}
        outputs['x'] = x
        outputs['x_hat'] = x_hat
        outputs['dx'] = dx
        outputs['dx_decoded'] = dx_decoded
        outputs['z'] = z
        outputs['dz'] = dz
        outputs['sindy_coefficients'] = self.sindy_coefficients
        outputs['coefficient_mask'] = self.coefficient_mask
        outputs['Theta'] = Theta
        if self.model_order == 1:
            outputs['dz_predict'] = sindy_prediction
        else:
            raise ValueError('Invalid model order')

        return outputs




class XcoderHalf(nn.Module):  # Xcoder as in Encoder or Decoder
    def __init__(self, input_dim, output_dim, widths, activation):
        super().__init__()
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        # self.activation = activation

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'elu':
            activation = nn.ELU()
        else:
            raise ValueError('Invalid activation function')

        self.layers = nn.ModuleList()
        for i in range(len(widths)):
            if i == 0:
                self.layers.append(FcLayer(input_dim, widths[i], activation))
            else:
                self.layers.append(FcLayer(widths[i-1], widths[i], activation))
        self.layers.append(FcLayer(widths[-1], output_dim, activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FcLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        fc = nn.Linear(self.input_dim, self.output_dim)
        torch.nn.init.xavier_normal_(fc.weight)
        torch.nn.init.zeros_(fc.bias)
        self.fc = fc

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class ZDerivativeOrder1(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, dx, encoder):
        dz = dx
        if self.activation == 'elu':
            for i in range(len(encoder.layers)-1):
                x = encoder.layers[i](x)
                dz = torch.multiply(torch.min(torch.exp(x), torch.ones_like(x)), torch.matmul(dz,encoder.layers[i].fc.weight))  # TODO check transpose
                f = nn.ELU()
                x = f(x)
        if self.activation == 'relu':
            for i in range(len(encoder.layers)-1):
                x = encoder.layers[i](x)
                dz = torch.multiply(torch.max(torch.sign(x),0), torch.matmul(dz,encoder.layers[i].fc.weight))  # TODO check transpose
                f = nn.ReLU()
                x = f(x)
        if self.activation == 'sigmoid':
            for i in range(len(encoder.layers)-1):
                x = encoder.layers[i](x)
                f = nn.Sigmoid()
                x = f(x)
                dz = torch.multiply(torch.multiply(x,1-x), torch.matmul(dz,torch.transpose(encoder.layers[i].fc.weight,0,1)))
        else:
            for i in range(len(encoder.layers)-1):
                dz = torch.matmul(dz,encoder.layers[i].fc.weight)
        dz = torch.matmul(dz,torch.transpose(encoder.layers[-1].fc.weight,0,1))
        return dz


class SINDyLibraryOrder1(nn.Module):
    def __init__(self,  poly_order, include_sine):
        super().__init__()
        self.poly_order = poly_order
        self.include_sine = include_sine
    def forward(self, z):
        n_times = z.shape[0]
        latent_dim = z.shape[1]

        library = torch.ones(n_times, 1)
        library = torch.cat((library, z), 1) # order = 1

        if self.poly_order > 1: # order = 2
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    library = torch.cat((library, torch.unsqueeze(z[:, i]*z[:, j],1)), 1)

        if self.poly_order > 2:  # order = 3
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        library = torch.cat((library,torch.unsqueeze(z[:, i] * z[:, j] * z[:, k],1)),1)

        if self.poly_order > 3: # order = 4
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            library = torch.cat((library,torch.unsqueeze(z[:, i] * z[:, j] * z[:, k] * z[:, p],1)),1)

        if self.poly_order == 5:  # order = 5
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            for q in range(p, latent_dim):
                                library = torch.cat((library, torch.unsqueeze(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q],1)),1)
        if self.poly_order > 5:
            raise ValueError('poly_order > 5 not implemented')

        if self.include_sine:
            library = torch.cat((library,torch.sin(z)),1)

        return library


def compound_loss(pred, target, cfg):
    x_input = pred['x']
    x_output = pred['x_hat']
    if cfg['model_order'] == 1:
        dz = pred['dz']
        dz_predict = pred['dz_predict']
        dx = pred['dx']
        dx_decoded = pred['dx_decoded']
    else:
        raise ValueError('Invalid model order')
    sindy_coeff = pred['sindy_coefficients'] * pred['coefficient_mask']
    losses = {}
    losses['decoder'] =nn.functional.mse_loss(x_output, x_input)   # reconstruction loss
    if cfg['model_order'] == 1:
        losses['sindy_z'] = nn.functional.mse_loss(dz, dz_predict)  # SINDy loss in z
        losses['sindy_x'] = nn.functional.mse_loss(dx, dx_decoded)  # SINDy loss in x
    else:
        raise ValueError('Invalid model order')
    losses['sindy_regularization'] = torch.abs(sindy_coeff).mean()  # Sparsify SINDy coefficients

    # combine losses
    loss = cfg['loss_weight_decoder'] * losses['decoder'] \
           + cfg['loss_weight_sindy_z'] * losses['sindy_z'] \
           + cfg['loss_weight_sindy_x'] * losses['sindy_x'] \
           + cfg['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    loss_refinement = cfg['loss_weight_decoder'] * losses['decoder'] \
                      + cfg['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + cfg['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, loss_refinement, losses

