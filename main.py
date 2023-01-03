import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


# todo
# report loss via osc
# read data from file


# OSC SERVER and ENDPOINTS
def point_handler(address, *args):
    arg_list = list(args)
    program_state.data.add_point(arg_list[:program_state.data.input_size], arg_list[program_state.data.input_size:])
    print(len(program_state.data.input_data))


def train_handler(address, *args):
    program_state.data.scale(program_state.normalize_input)
    input_tensor = torch.from_numpy(program_state.data.input_data_array)
    target_tensor = torch.from_numpy(program_state.data.output_data_array)
    ds = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=program_state.batch_size, shuffle=True)
    for epoch in range(int(args[0])):
        loss = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            # print(f'{epoch} {batch_idx} {features} {targets}')
            program_state.optimizer.zero_grad()  # zero the gradient buffers
            output = program_state.net(features)
            loss = program_state.criterion(output, targets)
            loss.backward()
            program_state.optimizer.step()
        if (epoch % program_state.loss_report) == 0:
            client.send_message("/loss", loss.item())


def predict_handler(address, *args):
    with torch.no_grad():
        input_data = np.array(list(args), dtype='float32')
        if program_state.normalize_input:
            input_data = program_state.data.input_scaler.transform(input_data.reshape(1, -1))
        input_tensor = torch.from_numpy(input_data.reshape(1, -1))
        pred = program_state.net(input_tensor)
        pred_scaled = program_state.data.output_scaler.inverse_transform(pred.numpy())
        client.send_message(program_state.client_path, *(pred_scaled.tolist()))


def save_handler(address, *args):
    program_state.save(args[0])


def default_handler(address, *args):
    print(f"Unknown endpoint {address}: {args}")


dispatcher = Dispatcher()
dispatcher.map("/nn/point", point_handler)
dispatcher.map("/nn/train", train_handler)
dispatcher.map("/nn/pred", predict_handler)
dispatcher.map("/save", save_handler)
dispatcher.set_default_handler(default_handler)

# argument parsing
parser = argparse.ArgumentParser(
    prog='DeepOsc',
    description='Deep MLP Regression via OSC',
    epilog='luc.doebereiner@gmail.com, (c) 2022')

parser.add_argument('-l', '--load-file', help='Load model and data from a file')
parser.add_argument('-s', '--layers-size', type=int, help='Number of perceptrons in hidden layers', default=10)
parser.add_argument('-i', '--input', type=int, help='Number of input values', default=4)
parser.add_argument('-o', '--output', type=int, help='Number of output values', default=2)
parser.add_argument('-d', '--depth', type=int, help='Network depth', default=3)
parser.add_argument('-lr', '--loss-report', type=int, help='Report loss every n epochs', default=10)
parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=5)
parser.add_argument('-pi', '--port-in', type=int, help='OSC receiving port', default=1337)
parser.add_argument('-ipo', '--ip-out', help='IP of the client', default="127.0.0.1")
parser.add_argument('-po', '--port-out', help='Port of the client', default=57120)
parser.add_argument('-p', '--pred-path', help='OSC path of the client for predictions', default="/pred")
parser.add_argument('-ui', '--unscaled-input', help='Do not rescale prediction input', action='store_false')

program_state = None


class ProgramState:
    def __init__(self):
        self.normalize_input = True
        self.batch_size = 5
        self.loss_report = None
        self.net = None
        self.data = None
        self.optimizer = None
        self.criterion = None
        self.client = None
        self.ip = "127.0.0.1"
        self.port = 1337
        self.client_path = "/pred"

    def save(self, fname):
        with open(fname + '.pkl', 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as fp:
            return pickle.load(fp)


# Neural Net

class Data:
    def __init__(self, in_size, out_size):
        self.input_size = in_size
        self.output_size = out_size
        self.input_data = []
        self.output_data = []
        self.input_data_array = None
        self.output_data_array = None
        self.input_scaler = MinMaxScaler()
        self.input_min = None
        self.input_max = None
        self.output_scaler = MinMaxScaler()
        self.output_min = None
        self.output_max = None

    def add_point(self, input_data, output_data):
        # todo check sizes
        self.input_data.append(input_data)
        self.output_data.append(output_data)

    def scale(self, scale_input=True):
        self.input_data_array = np.array(self.input_data, dtype='float32')
        self.output_data_array = np.array(self.output_data, dtype='float32')
        if scale_input:
            self.input_data_array = self.input_scaler.fit_transform(self.input_data_array)
        self.output_data_array = self.output_scaler.fit_transform(self.output_data_array)


class Net(nn.Module):

    def __init__(self, layers_size, depth, n_in, n_out):
        super(Net, self).__init__()

        # layers_size = (n_in + n_out) // 2
        self.layers = nn.ModuleList([nn.Linear(n_in, layers_size)])
        self.layers.extend([nn.Linear(layers_size, layers_size) for i in range(1, depth - 1)])
        self.layers.append(nn.Linear(layers_size, n_out))

    def forward(self, x):
        for f in self.layers:
            x = torch.sigmoid(f(x))
        return x


if __name__ == '__main__':
    args = parser.parse_args()

    if args.load_file is not None:
        program_state = ProgramState.load(args.load_file)
    else:
        program_state = ProgramState()
        program_state.port = args.port_in
        program_state.net = Net(args.layers_size, args.depth, args.input, args.output)
        program_state.data = Data(args.input, args.output)
        program_state.loss_report = args.loss_report
        program_state.criterion = nn.MSELoss()
        program_state.optimizer = optim.Adam(program_state.net.parameters())
        program_state.client_path = args.pred_path
        program_state.normalize_input = args.unscaled_input
        program_state.batch_size = args.batch_size

    client = udp_client.SimpleUDPClient(args.ip_out, args.port_out)
    server = BlockingOSCUDPServer((program_state.ip, program_state.port), dispatcher)
    server.serve_forever()
