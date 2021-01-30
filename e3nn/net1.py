import os
from e3nn import non_linearities
from e3nn.networks import GatedConvNetwork
from e3nn.point.data_helpers import DataNeighbors
from e3nn.point.message_passing import Convolution as MessageConv
from utils import get_data_loader, mask_by_len, N_FEATURES
import torch
from setproctitle import setproctitle
import argparse
from torch.utils.tensorboard import SummaryWriter


class MaskedAverageNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = GatedConvNetwork(*args, **kwargs)

    def forward(self, lengths, *args, **kwargs):
        output = self.network(*args, **kwargs)
        output = mask_by_len(output, lengths)
        return output.sum(1)[:, 0] / lengths

class GatedConvNeighbors(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs['convolution'] = MessageConv
        self.network = GatedConvNetwork(*args, **kwargs)

    def forward(self, nbrs):
        """ nbrs is a neighbors list generated by e3nn.point.data_helpers.DataNeighbors
        """
        output = self.network(nbrs.x, nbrs.edge_index, nbrs.edge_attr)
        return output

# class GatedConvNeighbors(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         kwargs['convolution'] = MessageConv
#         self.network = GatedConvNetwork(*args, **kwargs)

#     def forward(self, x, edge_index, attr):
#         """ nbrs is a neighbors list generated by e3nn.point.data_helpers.DataNeighbors
#         """
#         output = self.network(x, edge_index, attr)
#         return output

def main1(d=''):
    """ Load the data and train a GatedConvNet on sections of proteins. Convolutions are computed
        over the entire backbone which makes the learning task slightly infeasible, since the memory
        required rto compute the convolutions increases exponentially with sequence length. 
    """
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = get_data_loader(d, batch_size=4, max_length=50)
    Rs_in = [(1, 0)]
    Rs_hidden = [(16, 0), (16, 1), (16, 2)]
    Rs_out = [(1, 0)]
    lmax = 3

    f = MaskedAverageNet(Rs_in, Rs_hidden, Rs_out, lmax)
    f = f.to(device)

    optimizer = torch.optim.Adam(f.parameters(), lr=1e-2)

    epoch = 0
    while True:
        epoch += 1
        for i, batch in enumerate(data_loader):
            coords = batch['coords']
            feature = batch['feature']
            labels = batch['drmsd']
            lengths = batch['length']
            for tens in [coords, feature, labels, lengths]:
                tens.to(device)
            print(f'Max Length = {lengths.max()}', end='\t\t')
            out = f(lengths, feature, coords)
            loss = torch.nn.functional.mse_loss(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"step={i} loss={loss:.2f}")

def main2(
    name: str = 'GatedConvDefault',
    data_path: str='',
    save_dir: str='',
    save_every: int=50,
    tblog: str = '',
    tbinterval: int = 10,
    gpu: int = -1,
    lmax: int = 3,
    rmax: float = 1000,
    layers: int = 3,
    lr: float = 1e-2
    ):
    """ Try to do the same as main1, but with the conbolution calculated only over a fixed-radius
        neighborhood around a certain point.
    """

    if tbinterval != 0:
        ldir = os.path.join(tblog, name)
        print(f'Logging in {ldir}...')
        tbwriter = SummaryWriter(log_dir=ldir, comment=f'pid_{os.getpid()}')
    else:
        tbwriter = None

    torch.set_default_dtype(torch.float64)
    if gpu == -1:
        device = 'cpu'
    else:
        device = f'cuda:{gpu}'
    device = torch.device(device)

    data_loader = get_data_loader(data_path, batch_size=1, max_length=-1)

    Rs_in = [(N_FEATURES, 0)]
    Rs_hidden = [(16, 0), (16, 1), (8, 2)]
    Rs_out = [(1, 0)]

    net = GatedConvNeighbors(Rs_in, Rs_hidden, Rs_out, lmax=lmax, max_radius=rmax, layers=layers)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    epoch = 0
    i = 0
    metrics = {
        'Loss': [],
        'Avg_Neighbors': [],
        'Out/range': [],
        'Out/var': []
    }
    while True:
        epoch += 1
        for batch in data_loader:
            i += 1
            coords = batch['coords'][0]
            feature = batch['features'][0]
            labels = batch['drmsd'][0]
            lengths = batch['length'][0]
            for tens in [coords, feature, labels, lengths]:
                tens.to(device)
            
            nbrs = DataNeighbors(feature, coords, rmax)
            out = net(nbrs)
            if i == 1 and tbwriter != None:
                # tbwriter.add_graph(net, [nbrs.x, nbrs.edge_index, nbrs.edge_attr])
                pass

            loss = (out.mean() - labels) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update metrics
            metrics['Loss'].append(loss)
            metrics['Avg_Neighbors'].append(nbrs.edge_attr.shape[0] / nbrs.x.shape[0])
            metrics['Out/range'].append(out.max() - out.min())
            metrics['Out/var'].append(out.var())
            if i % save_every == 0 and save_dir != '' and save_every != 0:
                sdir = f'{name}-{epoch}:{i}'
                if not os.path.isdir(sdir):
                    os.mkdir(f'{name}-{epoch}:{i}')
                torch.save(net.state_dict(), os.path.join(save_dir, sdir))
                if tbwriter is not None:
                    tbwriter.add_hparams({'lr': lr, 'lmax': lmax, 'rmax': rmax, 'layers': layers, 'epochs': epoch, 'steps': i},
                             {'Loss': 0., 'Avg_Neighbors': 0., 'Out/var': 0.},
                            )
            if i % tbinterval == 0:
                if tbwriter is None:
                    print(f"epoch:step={epoch}:{i} loss={loss:.2f}")
                for m in metrics:
                    val = sum(metrics[m]) / max(1, len(metrics[m]))
                    if tbwriter is not None:
                        tbwriter.add_scalar(m, val, i)
                    else:
                        print(f'\t{m}: {val:.2f}')
                metrics = {m: [] for m in metrics}



if __name__ == '__main__':
    # command line interface
    # Parse inputs
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str,
                        help=f'Direcotry to find the representations dataset outputted by protein_geometry/src/scripts/rgn_data.py')
    parser.add_argument('--name', type=str, default='',
                        help='Override the name of the run. Affects the save file name and the name of the tensorboard run. '
                             'Will default to depend on the model specs.')
    parser.add_argument('--load', type=None, default=None,
                        help='Not Implemented')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default=os.path.join('.', 'save'),
                        help='Where to save the model.')
    parser.add_argument('--save-interval', dest='save_interval', type=int, default=100,
                        help='How often to save the model. Setting to 0 will disable saving.')
    parser.add_argument('--tb', type=str, default=os.path.join('~', 'eqnetworks', 'runs'),
                        help='Where to log tensorboard output. Will create a directory inside of the tb directory for this run.')
    parser.add_argument('--tb-interval', dest='tb_interval', type=int, default=10,
                        help='How often to log to tensorboard. Setting to 0 will disable logging.')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Which gpu to use. -1 means cpu.')
    parser.add_argument('--lmax', type=int, default=3,
                        help='Maximum l to use for spherical hermonics.')
    parser.add_argument('--rmax', type=float, default=1000,
                        help='Radius (Picometer) for convolution scope.')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of gated equivariant convolutional layers.')
    parser.add_argument('--reps', type=None, default=None,
                        help='Not Implemented')
    parser.add_argument('--features', type=None, default=None,
                        help='Not Implemented')
    opts = parser.parse_args()

    data_path = opts.data  # '~/protein_geometry/data/representations/rgn'
    save_dir = opts.save_dir
    if opts.name == '':
        name = f'lmax:{opts.lmax}-rmax:{opts.rmax}-layers:{opts.layers}'
    else:
        name = opts.name

    # Considering parameters to a file that can be loaded.
    kwargs = dict(
        data_path=data_path,
        save_every=opts.save_interval,
        save_dir=save_dir,
        tblog=opts.tb,
        gpu=opts.gpu,
        lmax=opts.lmax,
        rmax=opts.rmax,
        name=name,
        layers=opts.layers,
        tbinterval=opts.tb_interval,
    )

    setproctitle(name)
    print(f'Starting {name} with pid {os.getpid()}')
    main2(**kwargs)