import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
from functools import partial
from typing import Any, Callable


def conv_kernel_init():
    return jax.nn.initializers.variance_scaling(
        scale=2.0,
        mode='fan_out',
        distribution='normal'
    )


class SqueezeExcite(nnx.Module):

    def __init__(self, in_features: int, *, se_ratio: float = 0.25, activation: Any = nnx.swish, rngs: nnx.Rngs):
        self.activation = activation
        reduced_features = max(1, int(in_features * se_ratio))
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=reduced_features,
            kernel_size=1,
            padding='SAME',
            kernel_init=conv_kernel_init(),
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=reduced_features,
            out_features=in_features,
            kernel_size=1,
            padding='SAME',
            kernel_init=conv_kernel_init(),
            rngs=rngs
        )

    def __call__(self, inputs):
        x = jnp.mean(inputs, axis=1, keepdims=True)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = nnx.sigmoid(x)

        return inputs * x
    

class ASPP(nnx.Module):

    def __init__(
            self,
            in_features: int,
            *,
            depth: int,
            activation: Callable = nnx.swish,
            atrous_rates: list[int] = [6, 12, 18],
            kernel_sizes: list[int] = [5, 5, 5],
            rngs: nnx.Rngs):
        self.activation = activation
        conv1 = nnx.Conv(
            in_features=in_features,
            out_features=depth,
            kernel_size=1,
            use_bias=False,
            kernel_init=conv_kernel_init(),
            rngs=rngs
        )
        atrous_convs = [
            nnx.Conv(
                in_features=in_features,
                out_features=depth,
                kernel_size=ksize,
                kernel_dilation=rate,
                use_bias=False,
                rngs=rngs
            )
            for rate, ksize in zip(atrous_rates, kernel_sizes)
        ]
        self.conv_list = nnx.data([conv1, *atrous_convs])
        self.bn_list = nnx.data([
            nnx.BatchNorm(num_features=depth, rngs=rngs)
            for _ in range(len(self.conv_list))
        ])
        self.conv_project = nnx.Conv(
            in_features=(len(atrous_rates) + 1) * depth,  # Sum of all conv features
            out_features=depth,
            kernel_size=1,
            use_bias=False,
            rngs=rngs
        )
        self.bn_project = nnx.BatchNorm(num_features=depth, rngs=rngs)

    def __call__(self, x):
        concatenated = jnp.concatenate([
            self.activation(bn(conv(x)))
            for conv, bn in zip(self.conv_list, self.bn_list)
        ], axis=-1)

        projected = self.activation(self.bn_project(self.conv_project(concatenated)))

        return projected
    

def pad_nodes_to_match(node1, node2):
    """Pad to same shape if needed.
    
    This may be necessary when pooling leads to fractionated output shape.
    E.g. if pool_size is 4, and input length is 10, pooling leads to
    10/4 = 2.5. In this case, pooling layer implicitly crops the output to
    length 2. This needs to be taken into account by padding the up path output
    in upsampling part.
    
    NOTE: node1 should be the potentially larger one.
    
    Credit: https://github.com/perslev/U-Time/blob/master/utime/models/utime.py
    """
    s1 = node1.shape
    s2 = node2.shape
    
    diffs = s1[1] - s2[1]
    left_pad = diffs // 2
    right_pad = diffs // 2
    right_pad = right_pad + (diffs % 2)
    pads = [[0, 0], [left_pad, right_pad], [0, 0]]
    return jnp.pad(node2, pads)


class ConvBlock(nnx.Module):

    def __init__(
            self,
            *,
            in_features: int,
            out_features: int,
            kernel_size: int,
            pool_size: int,
            strides: int = 1,
            dilation_rate: int = 1,
            activation: Callable = nnx.swish,
            se_ratio: float | None = 0.25,
            use_batchnorm: bool = True,
            rngs: nnx.Rngs):
        self.activation = activation
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        self.se_ratio = se_ratio
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            kernel_dilation=dilation_rate,
            kernel_init=conv_kernel_init(),
            use_bias=False,
            rngs=rngs
        )
        if self.use_batchnorm:
            self.bn1 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            kernel_dilation=dilation_rate,
            kernel_init=conv_kernel_init(),
            use_bias=False,
            rngs=rngs
        )
        if self.use_batchnorm:
            self.bn2 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        if self.se_ratio is not None:
            self.se = SqueezeExcite(in_features=out_features, se_ratio=se_ratio, rngs=rngs)
        self.res_conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            kernel_init=conv_kernel_init(),
            rngs=rngs
        )

    def __call__(self, x):
        residual = self.res_conv(x)
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.activation(x)
        #x = self.activation(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        #x = self.bn2(self.conv2(x))
        if self.se_ratio is not None:
            x = self.se(x)

        x += residual
        skip = self.activation(x)
        if self.pool_size > 1:
            x = nnx.max_pool(skip, window_shape=(self.pool_size,), strides=(self.pool_size,))

        return x, skip
        

class Encoder(nnx.Module):

    def __init__(
            self,
            in_features,
            *,
            block_args: list[dict[str, Any]],
            activation: Callable = nnx.swish,
            use_batchnorm: bool = True,
            rngs: nnx.Rngs):
        in_features_list = [in_features, *[args['output_filters'] for args in block_args[:-2]]]
        self.blocks = nnx.data([
            ConvBlock(
                in_features=_in_features,
                out_features=args['output_filters'],
                kernel_size=args['kernel_size'],
                pool_size=args['pool_size'],
                strides=args['strides'],
                dilation_rate=args['dilation'],
                activation=activation,
                se_ratio=args['se_ratio'],
                use_batchnorm=use_batchnorm,
                rngs=rngs
            )
            for args, _in_features in zip(block_args[:-1], in_features_list)
        ])
        
        # Separate bottom block that will discard residuals
        args = block_args[-1]
        self.block_out = ConvBlock(
            in_features=block_args[-2]['output_filters'],
            out_features=args['output_filters'],
            kernel_size=args['kernel_size'],
            pool_size=args['pool_size'],
            strides=args['strides'],
            dilation_rate=args['dilation'],
            activation=activation,
            se_ratio=args['se_ratio'],
            use_batchnorm=use_batchnorm,
            rngs=rngs
        )

    def __call__(self, x):
        skips = ()
        for block in self.blocks:
            x, skip = block(x)
            skips += (skip,)

        x, _ = self.block_out(x)

        return x, skips
        

@partial(jax.jit, static_argnums=1)
def chwise_linear_upsample(arr: jnp.array, repeat: int):
    def _linear_upsample(arr):
        new_x = jnp.linspace(0, arr.shape[0] - 1, repeat * arr.shape[0])
        x = jnp.arange(arr.shape[0])
        return jnp.interp(new_x, x, arr)

    return jnp.apply_along_axis(_linear_upsample, 0, arr)


class UpsamplingBlock(nnx.Module):

    def __init__(
            self,
            *,
            in_features: int,
            out_features: int,
            kernel_size: int,
            activation: Callable = nnx.swish,
            pool_size: int = 1,
            se_ratio: float | None = 0.25,
            use_batchnorm: bool = True,
            rngs: nnx.Rngs):
        self.pool_size = pool_size
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.se_ratio = se_ratio
        self.conv1 = nnx.Conv(
            in_features=in_features + out_features,
            out_features=out_features,
            kernel_size=kernel_size,
            kernel_init=conv_kernel_init(),
            rngs=rngs
        )
        if self.use_batchnorm:
            self.bn1 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=kernel_size,
            kernel_init=conv_kernel_init(),
            rngs=rngs
        )
        if self.use_batchnorm:
            self.bn2 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        
        if self.se_ratio is not None:
            self.se = SqueezeExcite(in_features=out_features, se_ratio=se_ratio, activation=activation, rngs=rngs)
        self.res_conv = nnx.Conv(
            in_features=in_features + out_features,
            out_features=out_features,
            kernel_size=1,
            kernel_init=conv_kernel_init(),
            rngs=rngs
        )

    def __call__(self, x, skip):
        x = jax.vmap(partial(chwise_linear_upsample, repeat=self.pool_size))(x)
        x = pad_nodes_to_match(skip, x)
        x = jnp.concatenate([x, skip], axis=-1)

        residual = self.res_conv(x)

        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        if self.se_ratio is not None:
            x = self.se(x)
        x += residual
        x = self.activation(x)

        return x


class Decoder(nnx.Module):

    def __init__(
            self,
            in_features,
            *,
            block_args: list[dict[str, Any]],
            activation: Callable = nnx.swish,
            use_batchnorm: bool = True,
            rngs: nnx.Rngs):
        
        in_features_list = [in_features, *[args['output_filters'] for args in block_args[1:-1][::-1]]]
        self.blocks = nnx.data([
            UpsamplingBlock(
                in_features=_in_features,
                out_features=args['output_filters'],
                kernel_size=args['kernel_size'],
                activation=activation,
                pool_size=args['pool_size'],
                se_ratio=args['se_ratio'],
                use_batchnorm=use_batchnorm,
                rngs=rngs
            )
            for args, _in_features in zip(block_args[:-1][::-1], in_features_list)  # Exclude bottom block and reverse iterate block args
        ])

    def __call__(self, x, skips):
        for block, skip in zip(self.blocks, skips[::-1]):
            x = block(x, skip)
        
        return x


class UTime(nnx.Module):

    def __init__(
            self,
            *,
            input_names: list[str],
            block_args: list[dict[str, Any]],
            output_args: list[dict[str, Any]],
            activation: Callable = nnx.swish,
            aspp_depth: int | None = 128,
            return_bottom: bool = False,
            use_batchnorm: bool = True,
            dense_activation: Callable = nnx.tanh,
            rngs: nnx.Rngs):
        self.input_names = input_names
        self.return_bottom = return_bottom
        self.aspp_depth = aspp_depth
        self.dense_activation = dense_activation
        self.encoder = Encoder(
            in_features=len(input_names),
            block_args=block_args,
            activation=activation,
            use_batchnorm=use_batchnorm,
            rngs=rngs
        )
        if self.aspp_depth is not None:
            self.aspp = ASPP(
                in_features=block_args[-1]['output_filters'],
                depth=aspp_depth,
                activation=activation,
                rngs=rngs
            )
        self.decoder = Decoder(
            in_features=aspp_depth if aspp_depth is not None else block_args[-1]['output_filters'],
            block_args=block_args,
            activation=activation,
            use_batchnorm=use_batchnorm,
            rngs=rngs
        )
        self.output_blocks = nnx.data({
            args['output_name']: {
                'dense_conv': nnx.Conv(
                    in_features=block_args[0]['output_filters'],
                    out_features=args['n_classes'],
                    kernel_size=args['dense_ksize'],
                    rngs=rngs
                ),
                'samples_per_segment': args['samples_per_segment'],
                'segment_conv': nnx.Conv(
                    in_features=args['n_classes'],
                    out_features=args['n_classes'],
                    kernel_size=args['segment_ksize'],
                    rngs=rngs
                )
            }
            for args in output_args 
        })

    def __call__(self, inputs: dict[str, np.array]):
        x = jnp.concatenate([inputs[k] for k in self.input_names], axis=-1)
        encoded, skips = self.encoder(x)
        if self.aspp_depth is not None:
            encoded = self.aspp(encoded)
        decoded = self.decoder(encoded, skips)

        outputs = {}
        for oname, oblock in self.output_blocks.items():
            output = oblock['dense_conv'](decoded)
            output = self.dense_activation(output)
            output = nnx.avg_pool(
                output,
                window_shape=(oblock['samples_per_segment'],),
                strides=(oblock['samples_per_segment'],)
            )
            output = oblock['segment_conv'](output)
            outputs[oname] = output

        if self.return_bottom:
            return outputs, encoded

        return outputs
        