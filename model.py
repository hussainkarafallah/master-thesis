import torch.nn as nn
import torch
from torchvision import models
from torch.hub import load_state_dict_from_url


class EnglishCNNBackBone(nn.Module):

    def __init__(self, num_class=37, map_to_seq_hidden=64, rnn_hidden=256):
        super(EnglishCNNBackBone, self).__init__()
        self.cnn, (output_channel, output_height, output_width) = self.build_cnn()
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)
        state_dict = load_state_dict_from_url(
            'https://github.com/GitYCC/crnn-pytorch/raw/master/checkpoints/crnn_synth90k.pt',
            progress=True
        )

        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
    def build_cnn(self, img_height=32, img_width=100):
        channels = [1, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)
        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)


class OCRModel(nn.Module):

    def __init__(self, num_classes=1, num_lstms=1) -> None:
        super(OCRModel, self).__init__()
        self.num_classes = num_classes
        self.num_lstms = num_lstms

        # pre-trained RCNN English model
        self.cnn_backbone = EnglishCNNBackBone()

        # first freeze all layers
        self.cnn_backbone.cnn.requires_grad_(False)
        # unfreeze layers from conv3 to the end
        self.cnn_backbone.cnn.conv3.requires_grad_(True)
        self.cnn_backbone.cnn.conv4.requires_grad_(True)
        self.cnn_backbone.cnn.batchnorm4.requires_grad_(True)
        self.cnn_backbone.cnn.conv5.requires_grad_(True)
        self.cnn_backbone.cnn.batchnorm5.requires_grad_(True)
        self.cnn_backbone.cnn.conv6.requires_grad_(True)

        self.dropout = nn.Dropout(0.3)


        self.lstm = nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=self.num_lstms,
            dropout=0.3,
            batch_first=True, bidirectional=True
        )
        self.dense = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256*2, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes + 1)
        )

    def forward(self, x):
        # x has shape (B-size, h, w, c)
        # extract visual features
        conv_feat = torch.permute(x, (0, 3, 1, 2))
        # now conv_feat has shape (B-size, c, h, w)
        conv_feat = self.cnn_backbone(conv_feat)
        # remove h from shape
        conv_feat = nn.MaxPool2d((conv_feat.shape[2], 1))(conv_feat)
        # now conv_feat has shape (B-size, c, 1, w)
        conv_feat = torch.permute(conv_feat, (0, 3, 2, 1))
        # now conv_feat has shape (B-size, w, 1, c)

        # reshape
        new_shape = (
            conv_feat.shape[0],
            conv_feat.shape[1],
            -1 # merge the last 2 dims
        )
        conv_feat = torch.reshape(conv_feat, (new_shape))
        # now conv_feat has shape (B-size, w, c)

        conv_feat = self.dropout(conv_feat)

        # extract time related features
        lstm_out, _ = self.lstm(conv_feat)

        logits = self.dense(lstm_out)
        # logits have shape (B-size, w, num_classes + 1)

        #logits = torch.permute(logits, (1, 0, 2))

        log_probs = nn.functional.log_softmax(logits, dim=-1)

        return log_probs
