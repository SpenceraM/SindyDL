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
        self.coefficient_threshold = cfg['coefficient_threshold']
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
            self.z_derivative_func = ZDerivativeOrder2(self.activation)
            self.sindy_library = SINDyLibraryOrder2(self.poly_order, self.include_sine)


        # TODO: initialize with random values
        self.sindy_coefficients = nn.Parameter(torch.empty(self.library_dim, self.latent_dim, requires_grad=True))  # Tensor to hold coefficients
        torch.nn.init.constant_(self.sindy_coefficients, 1.0)

        if self.sequential_thresholding:
            # Binary tensor to mask out coefficients
            self.coefficient_mask = nn.Parameter(torch.ones(self.library_dim, self.latent_dim)).requires_grad_(False)

    def forward(self, x, dx, ddx=None):
        z = self.encoder(x)  # z = latent dim
        x_hat = self.decoder(z) # standard autoencoder output

        if self.model_order == 1:
            dz = self.z_derivative_func(x, dx, self.encoder)
            Theta = self.sindy_library(z).to(z.device)  # Theta = library dim
        else:
            dz, ddz = self.z_derivative_func(x, dx, ddx, self.encoder)
            Theta = self.sindy_library(z, dz).to(z.device)  # Theta = library dim

        if self.sequential_thresholding:
            sindy_prediction = torch.matmul(Theta, self.coefficient_mask * self.sindy_coefficients)
        else:
            sindy_prediction = torch.matmul(Theta, self.sindy_coefficients)

        if self.model_order == 1:
            dx_decoded = self.z_derivative_func(z, sindy_prediction, self.decoder)
        else:
            dx_decoded, ddx_decoded = self.z_derivative_func(z, dz, sindy_prediction, self.decoder)

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
            outputs['ddz'] = ddz
            outputs['ddz_predict'] = sindy_prediction
            outputs['ddx'] = ddx
            outputs['ddx_decoded'] = ddx_decoded

        return outputs

    def threshold_mask(self):
        with torch.no_grad():
            threshold = self.coefficient_threshold
            self.coefficient_mask[torch.abs(self.sindy_coefficients) < threshold] = 0
            print(self.coefficient_mask)
            print(self.sindy_coefficients)


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
        self.layers.append(FcLayer(widths[-1], output_dim, None))

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
        # torch.nn.init.constant_(fc.weight, 1.0)
        torch.nn.init.zeros_(fc.bias)
        self.fc = fc

    def forward(self, x):
        x = self.fc(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ZDerivativeOrder1(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, dx, coder):
        dz = dx
        if self.activation == 'elu':
            f = nn.ELU()
            for i in range(len(coder.layers) - 1):
                x = torch.matmul(x, torch.transpose(coder.layers[i].fc.weight, 0, 1)) + coder.layers[i].fc.bias
                dz = torch.multiply(torch.min(torch.exp(x), torch.ones_like(x)), torch.matmul(dz, coder.layers[i].fc.weight))  # TODO check transpose
                x = f(x)
        if self.activation == 'relu':
            f = nn.ReLU()
            for i in range(len(coder.layers) - 1):
                x = torch.matmul(x, torch.transpose(coder.layers[i].fc.weight, 0, 1)) + coder.layers[i].fc.bias
                dz = torch.multiply(torch.max(torch.sign(x),0), torch.matmul(dz, coder.layers[i].fc.weight))  # TODO check transpose
                x = f(x)
        if self.activation == 'sigmoid':
            f = nn.Sigmoid()
            for i in range(len(coder.layers) - 1):
                x = torch.matmul(x, torch.transpose(coder.layers[i].fc.weight, 0, 1)) + coder.layers[i].fc.bias
                x = f(x)
                dz = torch.multiply(torch.multiply(x,1-x), torch.matmul(dz, torch.transpose(coder.layers[i].fc.weight, 0, 1)))
        else:
            for i in range(len(coder.layers) - 1):
                dz = torch.matmul(dz, coder.layers[i].fc.weight)
        dz = torch.matmul(dz, torch.transpose(coder.layers[-1].fc.weight, 0, 1))
        return dz


class ZDerivativeOrder2(nn.Module):
    """
    Compute the first and second order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
        ddz - Tensorflow array, second order time derivatives of the network output.
    """
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, dx, ddx, coder):
        dz = dx
        ddz = ddx
        if self.activation == 'elu':
            f = nn.ELU()
            for i in range(len(coder.layers) - 1):
                x = torch.matmul(x, torch.transpose(coder.layers[i].fc.weight, 0, 1)) + coder.layers[i].fc.bias
                dz_prev = torch.matmul(dz, coder.layers[i].fc.weight)  # prolly will need transpose
                elu_derivative = torch.min(torch.exp(x), torch.ones_like(x))
                elu_derivative2 = torch.multiply(torch.exp(x), (input<0).to(torch.float32))
                dz = torch.multiply(elu_derivative, dz_prev)
                ddz = torch.multiply(elu_derivative2, torch.square(dz_prev)) \
                      + torch.multiply(elu_derivative, torch.matmul(ddz, coder.layers[i].fc.weight))  # prolly will need transpose
                x = f(x)
            dz = torch.matmul(dz, coder.layers[-1].fc.weight)
            ddz = torch.matmul(ddz, coder.layers[-1].fc.weight)
        elif self.activation == 'relu':
            # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
            f = nn.ReLU()
            for i in range(len(coder.layers) - 1):
                x = torch.matmul(x, torch.transpose(coder.layers[i].fc.weight, 0, 1)) + coder.layers[i].fc.bias
                relu_derivative = torch.max(torch.sign(x),0)
                dz = torch.multiply(relu_derivative, torch.matmul(dz, coder.layers[i].fc.weight))
                ddz = torch.multiply(relu_derivative, torch.matmul(ddz, coder.layers[i].fc.weight))  # 2nd deriv of activ. is 0
                x = f(x)
            dz = torch.matmul(dz,coder.layers[-1].fc.weight)
            ddz = torch.matmul(ddz,coder.layers[-1].fc.weight)
        elif self.activation == 'sigmoid':
            f = nn.Sigmoid()
            for i in range(len(coder.layers) - 1):
                x = torch.matmul(x, torch.transpose(coder.layers[i].fc.weight, 0, 1)) + coder.layers[i].fc.bias
                x = f(x)
                dz_prev = torch.matmul(dz, torch.transpose(coder.layers[i].fc.weight, 0, 1))
                sigmoid_derivative = torch.multiply(x, 1-x)
                sigmoid_derivative2 = torch.multiply(sigmoid_derivative, 1 - 2*x)
                dz = torch.multiply(sigmoid_derivative, dz_prev)
                ddz = torch.multiply(sigmoid_derivative2, torch.square(dz_prev)) \
                      + torch.multiply(sigmoid_derivative, torch.matmul(ddz, torch.transpose(coder.layers[i].fc.weight, 0, 1)))
            dz = torch.matmul(dz,torch.transpose(coder.layers[-1].fc.weight, 0, 1))
            ddz = torch.matmul(ddz, torch.transpose(coder.layers[-1].fc.weight, 0, 1))
        else:
            for i in range(len(coder.layers) - 1):
                dz = torch.matmul(dz, coder.layers[i].fc.weight)
                ddz = torch.matmul(ddz, coder.layers[i].fc.weight)
            dz = torch.matmul(dz, coder.layers[-1].fc.weight)
            ddz = torch.matmul(ddz, coder.layers[-1].fc.weight)
        return dz,ddz

class SINDyLibraryOrder1(nn.Module):
    def __init__(self,  poly_order, include_sine):
        super().__init__()
        self.poly_order = poly_order
        self.include_sine = include_sine

    def forward(self, z):
        n_times = z.shape[0]
        latent_dim = z.shape[1]

        library = torch.ones(n_times, 1, device=z.device)
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


class SINDyLibraryOrder2(nn.Module):
    def __init__(self,  poly_order, include_sine):
        super().__init__()
        self.poly_order = poly_order
        self.include_sine = include_sine

    def forward(self, z, dz):
        n_times = z.shape[0]
        latent_dim = z.shape[1]

        library = torch.ones(n_times, 1, device=z.device)
        z_combined = torch.cat((z, dz), 1)
        library = torch.cat((library, z_combined), 1)

        if self.poly_order > 1:
            for i in range(2*latent_dim):
                for j in range(i, 2*latent_dim):
                    library = torch.cat((library,torch.unsqueeze(z_combined[:, i]*z_combined[:, j],1)), 1)

        if self.poly_order > 2:
            for i in range(2*latent_dim):
                for j in range(i, 2*latent_dim):
                    for k in range(j, 2*latent_dim):
                        library = torch.cat((library, torch.unsqueeze(z_combined[:, i]*z_combined[:, j]*z_combined[:, k],1)), 1)

        if self.poly_order > 3:
            for i in range(2*latent_dim):
                for j in range(i, 2*latent_dim):
                    for k in range(j, 2*latent_dim):
                        for p in range(k, 2*latent_dim):
                            library = torch.cat((library, torch.unsqueeze(z_combined[:, i]*z_combined[:, j]*z_combined[:, k]*z_combined[:, p],1)), 1)

        if self.poly_order > 4:
            for i in range(2*latent_dim):
                for j in range(i, 2*latent_dim):
                    for k in range(j, 2*latent_dim):
                        for p in range(k, 2*latent_dim):
                            for q in range(p, 2*latent_dim):
                                library = torch.cat((library, torch.unsqueeze(z_combined[:, i]*z_combined[:, j]*z_combined[:, k]*z_combined[:, p]*z_combined[:, q],1)), 1)

        if self.poly_order > 5:
            raise ValueError('poly_order > 5 not implemented')

        if self.include_sine:
            library = torch.cat((library, torch.sin(z_combined)), 1)

        return library

def compound_loss(network_out, cfg):
    x_input = network_out['x']
    x_output = network_out['x_hat']
    if cfg['model_order'] == 1:
        dz = network_out['dz']
        dz_predict = network_out['dz_predict']
        dx = network_out['dx']
        dx_decoded = network_out['dx_decoded']
    else:
        ddz = network_out['ddz']
        ddz_predict = network_out['ddz_predict']
        ddx = network_out['ddx']
        ddx_decoded = network_out['ddx_decoded']

    sindy_coeff = network_out['sindy_coefficients'] * network_out['coefficient_mask']
    losses = {}
    losses['decoder'] =nn.functional.mse_loss(x_output, x_input, reduction='mean')   # reconstruction loss
    if cfg['model_order'] == 1:
        losses['sindy_z'] = nn.functional.mse_loss(dz, dz_predict, reduction='mean')  # SINDy loss in z
        losses['sindy_x'] = nn.functional.mse_loss(dx, dx_decoded, reduction='mean')  # SINDy loss in x
    else:
        losses['sindy_z'] = nn.functional.mse_loss(ddz, ddz_predict, reduction='mean')
        losses['sindy_x'] = nn.functional.mse_loss(ddx, ddx_decoded, reduction='mean')

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

