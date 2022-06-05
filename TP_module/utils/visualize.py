
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import datetime
import os 
import random

def visualize_reward(matrix, time_folder, img_name, opt, gpu=True):
   
   if gpu:
      matrix = matrix.cpu().detach().numpy()
   fig = plt.figure(figsize=(8,6))
   plt.imshow(matrix)
   plt.colorbar()
   plt.title("Reward Map")
   reward_folder = "reward_map/{}".format(time_folder)
   if opt.save_img:
      if not os.path.exists(reward_folder):
         os.mkdir(reward_folder)
      plt.savefig("{}/{}".format(reward_folder, img_name))
   if opt.visualize:
      plt.show()
    # plt.imshow(matrix[0][0], interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5,10.5,0.5,10.5))
    # plt.show()

def visualize_svf(matrix, img_name, opt, gpu=False):
   if gpu:
      matrix = matrix.cpu().detach().numpy()
   fig = plt.figure(figsize=(8,6))
   plt.imshow(matrix)
   plt.colorbar()
   plt.title("SVF Map")
   svf_path = "svfs/{}".format(img_name)
   if True:
      plt.savefig("{}".format(svf_path))
   if opt.visualize:
      plt.show()
    # plt.imshow(matrix[0][0], interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5,10.5,0.5,10.5))
    # plt.show()