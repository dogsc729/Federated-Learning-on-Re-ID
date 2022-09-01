import torch
import torchreid
import resnet_vallina as resnets

datamanager = torchreid.data.ImageDataManager(
    root="../datasets", 
    sources="msmt17",
    targets="msmt17",
)

model = resnets.__dict__['resnet50']()
save_path = "/home/b07611033/decentralized_reproduce/checkpoint/2022-06-13-05-18-11-normalresnet50_fedavg_laststride/MSMT17_specific.ckpt"
model.load_state_dict(torch.load(save_path ,map_location=torch.device('cpu')))

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer = optimizer,
    scheduler = scheduler
)

engine.run(
    save_dir='../checkpoint/prid',
    test_only=True
)
