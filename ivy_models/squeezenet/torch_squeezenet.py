import torch
import ivy
if __name__=='__main__':
    model = torch.hub.load('pytorch/vision', 'squeezenet1_0', pretrained=True)
    weights_raw = ivy.to_numpy(ivy.Container(model.state_dict()))
    print(weights_raw)