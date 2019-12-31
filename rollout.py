from gym.envs.box2d.car_racing import *
import torchvision.transforms as transforms
import cv2
import torch
import numpy as np

device = torch.device('cuda')
tensorize = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

def Rollout(encoder, rnn, cont, render=False, return_frames=False):
    assert cont.__name__ == 'global'
    
    env = CarRacing()
    env.render()
    obs = env.reset()
    total_reward = 0
    hidden = rnn._init_hidden(1)
    h, c = hidden[0].to(device).unsqueeze(0), hidden[1].to(device).unsqueeze(0)
    limit = 1000
    time = 0
    frames = []
    while True:
        if render:
            env.render()
        obs = cv2.resize(obs, (64, 64))
        obs = tensorize(obs).to(device)
        z = encoder(obs.unsqueeze(0))
        z = z.unsqueeze(0)
        action = cont(torch.cat([z, h], dim=-1))
        obs, reward, done, _ = env.step(action.squeeze().detach().cpu().numpy())
        if return_frames:
            frames.append(obs)# clipping??
        total_reward += reward
        _, _, _, h, c = rnn(torch.cat([z, action], dim=-1), h, c)

        if time%10==0:
            h, c = h.detach(), c.detach()
        
        if done or time > limit:
            env.close()
            break
        time += 1
    print("Reward:", total_reward)
    if return_frames:
        return total_reward, frames
    return total_reward