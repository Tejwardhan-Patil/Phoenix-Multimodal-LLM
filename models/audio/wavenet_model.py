import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        padding = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (padding, 0))
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, dilation, residual_channels, skip_channels):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.conv_filter = CausalConv1d(residual_channels, residual_channels, kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_channels, residual_channels, kernel_size=2, dilation=dilation)
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        out = filter_out * gate_out
        residual = self.conv_residual(out) + x
        skip = self.conv_skip(out)
        return residual, skip

class WaveNet(nn.Module):
    def __init__(self, residual_channels=64, skip_channels=256, n_blocks=3, n_layers=10):
        super(WaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.dilations = [2 ** i for i in range(n_layers)]
        
        self.causal_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(dilation, residual_channels, skip_channels) 
            for _ in range(n_blocks) 
            for dilation in self.dilations
        ])
        
        self.conv_post_skip = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.causal_conv(x)
        skip_connections = []
        
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        skip_sum = F.relu(self.conv_post_skip(skip_sum))
        out = F.relu(self.conv_out1(skip_sum))
        out = self.conv_out2(out)
        return out

class DilatedResidualBlock(nn.Module):
    def __init__(self, dilation, channels):
        super(DilatedResidualBlock, self).__init__()
        self.conv_filter = CausalConv1d(channels, channels, kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(channels, channels, kernel_size=2, dilation=dilation)
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        out = filter_out * gate_out
        residual = self.residual_conv(out) + x
        return residual

class ExtendedWaveNet(nn.Module):
    def __init__(self, layers, residual_channels, skip_channels, out_channels=1):
        super(ExtendedWaveNet, self).__init__()
        self.layers = layers
        self.dilations = [2 ** i for i in range(layers)]
        self.initial_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        
        self.residual_blocks = nn.ModuleList([
            DilatedResidualBlock(dilation, residual_channels)
            for dilation in self.dilations
        ])
        
        self.skip_connections = nn.ModuleList([
            nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
            for _ in range(layers)
        ])
        
        self.final_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.final_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)
        skip_outs = []

        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            skip_out = self.skip_connections[i](x)
            skip_outs.append(skip_out)

        total_skip_out = sum(skip_outs)
        total_skip_out = F.relu(self.final_conv1(total_skip_out))
        out = self.final_conv2(total_skip_out)
        return out

class ConditionedWaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, condition_channels):
        super(ConditionedWaveNet, self).__init__()
        self.initial_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        
        self.condition_conv = nn.Conv1d(condition_channels, residual_channels, kernel_size=1)
        
        self.residual_block = ResidualBlock(dilation=2, residual_channels=residual_channels, skip_channels=skip_channels)
        
        self.output_conv = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x, condition):
        condition = self.condition_conv(condition)
        x = self.initial_conv(x) + condition
        x, skip = self.residual_block(x)
        out = self.output_conv(skip)
        return out

class SkipResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size=2, dilation=1):
        super(SkipResidualBlock, self).__init__()
        self.conv_filter = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.conv_gate = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        out = filter_out * gate_out
        residual = self.conv_residual(out) + x
        skip = self.conv_skip(out)
        return residual, skip

class StackedWaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, n_blocks, n_layers):
        super(StackedWaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        
        self.causal_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        self.blocks = nn.ModuleList([
            SkipResidualBlock(residual_channels, skip_channels, kernel_size=2, dilation=2**i)
            for i in range(n_layers)
        ] for _ in range(n_blocks))
        
        self.conv_post_skip = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.causal_conv(x)
        skip_connections = []

        for block_set in self.blocks:
            for block in block_set:
                x, skip = block(x)
                skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        skip_sum = F.relu(self.conv_post_skip(skip_sum))
        out = F.relu(self.conv_out1(skip_sum))
        out = self.conv_out2(out)
        return out

class MultiConditionedWaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, condition_channels):
        super(MultiConditionedWaveNet, self).__init__()
        self.initial_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        self.condition_conv = nn.Conv1d(condition_channels, residual_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            SkipResidualBlock(residual_channels, skip_channels, kernel_size=2, dilation=2**i)
            for i in range(10)  # 10 layers with exponentially increasing dilation
        ])
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x, condition):
        condition = self.condition_conv(condition)
        x = self.initial_conv(x) + condition
        skip_connections = []

        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        skip_sum = F.relu(self.conv_out1(skip_sum))
        out = self.conv_out2(skip_sum)
        return out

class DynamicWaveNet(nn.Module):
    def __init__(self, residual_channels=64, skip_channels=256, condition_channels=128, n_blocks=4, n_layers=8):
        super(DynamicWaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.dilations = [2 ** i for i in range(n_layers)]
        
        self.input_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        self.condition_conv = nn.Conv1d(condition_channels, residual_channels, kernel_size=1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(dilation, residual_channels, skip_channels) 
            for _ in range(n_blocks) 
            for dilation in self.dilations
        ])
        
        self.conv_post_skip = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x, condition):
        condition = self.condition_conv(condition)
        x = self.input_conv(x) + condition
        skip_connections = []
        
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        skip_sum = F.relu(self.conv_post_skip(skip_sum))
        out = F.relu(self.conv_out1(skip_sum))
        out = self.conv_out2(out)
        return out

class ResidualWaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, n_blocks=4, n_layers=10):
        super(ResidualWaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.dilations = [2 ** i for i in range(n_layers)]
        
        self.causal_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(dilation, residual_channels, skip_channels) 
            for _ in range(n_blocks) 
            for dilation in self.dilations
        ])
        
        self.conv_post_skip = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.causal_conv(x)
        skip_connections = []
        
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        skip_sum = F.relu(self.conv_post_skip(skip_sum))
        out = F.relu(self.conv_out1(skip_sum))
        out = self.conv_out2(out)
        return out

class ConditionedResidualWaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, condition_channels):
        super(ConditionedResidualWaveNet, self).__init__()
        self.initial_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        
        self.condition_conv = nn.Conv1d(condition_channels, residual_channels, kernel_size=1)
        
        self.residual_block = ResidualBlock(dilation=2, residual_channels=residual_channels, skip_channels=skip_channels)
        
        self.output_conv = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x, condition):
        condition = self.condition_conv(condition)
        x = self.initial_conv(x) + condition
        x, skip = self.residual_block(x)
        out = self.output_conv(skip)
        return out

class AdvancedSkipResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size=2, dilation=1):
        super(AdvancedSkipResidualBlock, self).__init__()
        self.conv_filter = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.conv_gate = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.conv_condition = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

    def forward(self, x, condition=None):
        if condition is not None:
            x = x + self.conv_condition(condition)
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        out = filter_out * gate_out
        residual = self.conv_residual(out) + x
        skip = self.conv_skip(out)
        return residual, skip

class DeepStackedWaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, condition_channels=None, n_blocks=4, n_layers=10):
        super(DeepStackedWaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        
        self.input_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        
        self.blocks = nn.ModuleList([
            AdvancedSkipResidualBlock(residual_channels, skip_channels, kernel_size=2, dilation=2**i)
            for i in range(n_layers)
        ] for _ in range(n_blocks))
        
        self.condition_conv = nn.Conv1d(condition_channels, residual_channels, kernel_size=1) if condition_channels else None
        
        self.conv_post_skip = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x, condition=None):
        x = self.input_conv(x)
        if condition is not None and self.condition_conv is not None:
            condition = self.condition_conv(condition)
        
        skip_connections = []

        for block_set in self.blocks:
            for block in block_set:
                x, skip = block(x, condition)
                skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        skip_sum = F.relu(self.conv_post_skip(skip_sum))
        out = F.relu(self.conv_out1(skip_sum))
        out = self.conv_out2(out)
        return out

class AttentionWaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, attention_heads, condition_channels=None, n_blocks=4, n_layers=10):
        super(AttentionWaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.attention_heads = attention_heads
        
        self.input_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        
        self.blocks = nn.ModuleList([
            AdvancedSkipResidualBlock(residual_channels, skip_channels, kernel_size=2, dilation=2**i)
            for i in range(n_layers)
        ] for _ in range(n_blocks))
        
        self.attention = nn.MultiheadAttention(embed_dim=residual_channels, num_heads=attention_heads)
        self.condition_conv = nn.Conv1d(condition_channels, residual_channels, kernel_size=1) if condition_channels else None
        
        self.conv_post_skip = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x, condition=None):
        x = self.input_conv(x)
        if condition is not None and self.condition_conv is not None:
            condition = self.condition_conv(condition)
        
        skip_connections = []

        for block_set in self.blocks:
            for block in block_set:
                x, skip = block(x, condition)
                skip_connections.append(skip)
        
        x = x.permute(2, 0, 1)  # Change shape for attention [sequence_length, batch, embedding]
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)  # Back to [batch, channels, sequence_length]

        skip_sum = sum(skip_connections)
        skip_sum = F.relu(self.conv_post_skip(skip_sum))
        out = F.relu(self.conv_out1(skip_sum))
        out = self.conv_out2(out)
        return out

class WaveNetVAE(nn.Module):
    def __init__(self, residual_channels, skip_channels, latent_dim, n_blocks=4, n_layers=10):
        super(WaveNetVAE, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        
        self.encoder_input_conv = CausalConv1d(1, residual_channels, kernel_size=2, dilation=1)
        self.encoder_blocks = nn.ModuleList([
            AdvancedSkipResidualBlock(residual_channels, skip_channels, kernel_size=2, dilation=2**i)
            for i in range(n_layers)
        ] for _ in range(n_blocks))
        
        self.fc_mu = nn.Linear(skip_channels, latent_dim)
        self.fc_logvar = nn.Linear(skip_channels, latent_dim)
        
        self.decoder_input_conv = nn.Conv1d(latent_dim, skip_channels, kernel_size=1)
        self.decoder_blocks = nn.ModuleList([
            AdvancedSkipResidualBlock(skip_channels, skip_channels, kernel_size=2, dilation=2**i)
            for i in range(n_layers)
        ])
        
        self.conv_out = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def encode(self, x):
        x = self.encoder_input_conv(x)
        skip_connections = []

        for block_set in self.encoder_blocks:
            for block in block_set:
                x, skip = block(x)
                skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        mu = self.fc_mu(skip_sum.mean(dim=-1))
        logvar = self.fc_logvar(skip_sum.mean(dim=-1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input_conv(z.unsqueeze(-1))
        for block in self.decoder_blocks:
            z, _ = block(z)
        out = self.conv_out(z)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar