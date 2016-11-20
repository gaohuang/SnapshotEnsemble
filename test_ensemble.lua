--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
--  Code modified for Snapshot Ensembles by Gao Huang
--
local optim = require 'optim'

local M = {}
local Tester = torch.class('resnet.Tester', M)

function Tester:__init(model, opt)
   self.model = model
   self.opt = opt
end


function Tester:test(dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0
   local nModels = #self.model

   local top1All, top5All = torch.zeros(nModels), torch.zeros(nModels) -- single model results plus ensemble results
   local top1Evolve, top5Evolve = torch.zeros(nModels), torch.zeros(nModels) -- ensemble results with growing number of models

   for i = 1, nModels do
      self.model[i]:cuda()     
      self.model[i]:evaluate()
   end

   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1)

      local softmax = torch.Tensor() --store the average softmax
      if self.opt.dataset == 'cifar10' or self.opt.dataset == 'svhn' then
         softmax:resize(batchSize,10):zero()
      elseif self.opt.dataset == 'cifar100' then
         softmax:resize(batchSize,100):zero()  
      elseif self.opt.dataset == 'tiny-imagenet' then
         softmax:resize(batchSize,200):zero() 
      elseif self.opt.dataset == 'imagenet' then
         softmax:resize(batchSize,1000):zero() 
      else 
         print('Unkonwn dataset!') 
      end 

      batchSize = batchSize / nCrops

      local top1, top5 = 0, 0
      for i = 1, nModels do
         local output = self.model[i]:forward(self.input:cuda()):float()
         top1, top5 = self:computeScore(output, sample.target, nCrops)
         top1All[i] = top1All[i] + top1*batchSize
         top5All[i] = top5All[i] + top5*batchSize
         softmax:add(nn.SoftMax():forward(output))
         top1, top5 = self:computeScore(softmax, sample.target, nCrops)
         top1Evolve[i] = top1Evolve[i] + top1*batchSize
         top5Evolve[i] = top5Evolve[i] + top5*batchSize
      end

      N = N + batchSize

      print((' | Test: [%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         n, size, timer:time().real, dataTime, top1, top1Evolve[nModels] / N, top5, top5Evolve[nModels] / N))

      timer:reset()
      dataTimer:reset()
   end

   for i = 1, nModels do
      top1All[i] = top1All[i] / N
      top5All[i] = top5All[i] / N
      top1Evolve[i] = top1Evolve[i] / N
      top5Evolve[i] = top5Evolve[i] / N
      print((' * Single model top1: %7.3f  top5: %7.3f * Ensemble %d model(s) top1: %7.3f  top5: %7.3f  \n'):format(top1All[i], top5All[i], i, top1Evolve[i], top5Evolve[i]))
   end

   return top1All, top5All, top1Evolve, top5Evolve
end


function Tester:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Tester:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end


return M.Tester
