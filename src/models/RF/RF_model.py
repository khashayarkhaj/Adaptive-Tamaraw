import torch.nn as nn
import math
import torch
from utils.file_operations import predict_model_size
# The robust fingerprinting model (obtained from the source code https://github.com/robust-fingerprinting/RF/blob/master/RF/models/RF.py)

class RF(nn.Module):

    def __init__(self, feature_configs, num_classes=95, init_weights=True, num_classes_2=None):
        super(RF, self).__init__()
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        self.first_layer = make_first_layers()
        self.feature_configs = feature_configs
        self.features = make_layers(feature_configs['N'])
        self.class_num = num_classes
        #self.classifier = nn.AdaptiveAvgPool1d(1)

        self.main_classifier = make_classifier(input_channels= feature_configs['N'][-1], num_classes= num_classes)
        self.num_classes_2 = num_classes_2
        self.multi_task_enabled = False
        
        if num_classes_2 is not None: # we want to do dual learning
            self.multi_task_enabled = True
            self.classifier_y2 = make_classifier(
                input_channels=feature_configs['N'][-1], 
                num_classes= num_classes_2
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        
        # Shared features
        features = self.features(x)
        
        # Task-specific predictions
        if self.multi_task_enabled is False:
            y1_pred = self.main_classifier(features)
            y1_pred = y1_pred.view(y1_pred.size(0), -1)
            return y1_pred
        else:
            y1_pred = self.main_classifier(features)
            y2_pred = self.classifier_y2(features)
            
            y1_pred = y1_pred.view(y1_pred.size(0), -1)
            y2_pred = y2_pred.view(y2_pred.size(0), -1)
            
            return y1_pred, y2_pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def enable_multi_task(self):
        self.multi_task_enabled = True
    def disable_multi_task(self):
        self.multi_task_enabled = False




def make_layers(cfg, in_channels=32):
    layers = []

    for i, v in enumerate(cfg):
        # print(f'v is {v} ')
        if v == 'M':
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)

def make_classifier(input_channels = 512, num_classes = 95):
    layers = []
    conv1d = nn.Conv1d(input_channels, num_classes, kernel_size=3, stride=1, padding=1)
    layers += [conv1d, nn.BatchNorm1d(num_classes, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
    layers += [nn.AdaptiveAvgPool1d(1)]
    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)





def getRF(num, cfg = None, num2 = None):

    if cfg is None:
        cfg = {
            'N': [128, 128, 'M', 256, 256, 'M', 512]
                } # play with this part to prevent overfitting?

    
    model = RF(cfg, num_classes=num, num_classes_2= num2)
    

    return model


if __name__ == '__main__':
    net = getRF(95)
    print(net)
    model_size = predict_model_size(net, unit= 'mb')

    print(f'model is approximately {model_size} mb')
