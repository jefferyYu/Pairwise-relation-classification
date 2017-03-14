require 'nn'
require 'sys'
require 'torch'

local Trainer = torch.class('Trainer')

-- Perform one epoch of training.
function Trainer:train(train_data, train_e1, train_e2, train_dep, train_dep_e1, train_dep_e2, train_labels, mi_labels, model, criterion, micriterion, optim_method, layers, state, params, grads)
  model:training()

  local train_size = train_data:size(1)

  local timer = torch.Timer()
  local time = timer:time().real
  local total_err = 0

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local config -- for optim
  local config2
  if opt.optim_method == 'adadelta' then
    config = { rho = 0.95, eps = 1e-6 } 
    config2 = { rho = 0.95, eps = 1e-1 } 
  elseif opt.optim_method == 'adam' then
    config = {}
  end

  -- shuffle batches
  local num_batches = math.floor(train_size / opt.batch_size)
  local shuffle = torch.randperm(num_batches)
  local final_weight = torch.ones(opt.num_classes)
  local grad2 = torch.zeros(opt.num_classes)
  for i = 1, shuffle:size(1) do
    local t = (shuffle[i] - 1) * opt.batch_size + 1
    local batch_size = math.min(opt.batch_size, train_size - t + 1)

    -- data samples and labels, in mini batches.
    local inputs = train_data:narrow(1, t, batch_size) --narrow(dim,index, size)
	  local input_p1s = train_e1:narrow(1, t, batch_size)
	  local input_p2s = train_e2:narrow(1, t, batch_size)
    local indep = train_dep:narrow(1, t, batch_size)
    local indep_p1s = train_dep_e1:narrow(1, t, batch_size)
    local indep_p2s = train_dep_e2:narrow(1, t, batch_size)
    local targets = train_labels:narrow(1, t, batch_size)
    local mitargets = mi_labels:narrow(1, t, batch_size)

    inputs = inputs:double()
	  input_p1s = input_p1s:double()
	  input_p2s = input_p2s:double()
    indep = indep:double()
    indep_p1s = indep_p1s:double()
    indep_p2s = indep_p2s:double()
    targets = targets:double()
    mitargets = mitargets:double()

    -- closure to return err, df/dx
    local func = function(x)
      -- get new parameters
      if x ~= params then
        params:copy(x)
      end
      -- reset gradients
      grads:zero()

      -- forward pass
      local output = model:forward({inputs, input_p1s, input_p2s, indep, indep_p1s, indep_p2s, inputs, input_p2s, input_p1s, indep, indep_p2s, indep_p1s})
      local outputs, mioutputs, sm_outputs, sm_mioutputs = unpack(output)
      local finaloutputs = torch.zeros(batch_size, opt.num_classes)

      local err = criterion:forward(outputs, targets)
      local mierr = micriterion:forward(mioutputs, mitargets)
      -- track errors and confusion
      local final_err = 0
      local N = batch_size
      for i=1, N do
        label = targets[i]
        local ori_pb = 0
        local mi_pb = 0
        if label == 1 then
          ori_pb = sm_outputs[i][label]
          mi_pb = sm_mioutputs[i][label]
        elseif label < 11 then 
          ori_pb = sm_outputs[i][label]
          mi_pb = sm_mioutputs[i][label+9]
        elseif label > 10 then
          ori_pb =  sm_outputs[i][label]
          mi_pb = sm_mioutputs[i][label-9]
        end
        local weight = torch.sigmoid(final_weight[label])
        local prb = weight * ori_pb + (1-weight) * mi_pb
        if prb > 1e-300 then
          final_err = final_err + torch.log(prb)
        else
          final_err = final_err + torch.log(1e-300)
        end
      end
      final_err = -final_err/N
      --print(final_err)
      
      --local finalerr = finalcriterion:forward(finaloutputs, targets)

      -- track errors and confusion
      total_err = total_err + err * batch_size
      total_err = total_err + mierr * batch_size
      total_err = total_err + final_err * batch_size

      for j = 1, batch_size do
        confusion:add(outputs[j], targets[j])
      end

      -- compute gradients
      local M = opt.num_classes
      local warr_gd = torch.zeros(M)
      local sm_df_do = torch.zeros(batch_size, opt.num_classes)
      local sm_midf_do = torch.zeros(batch_size, opt.num_classes)
      for i = 1, N do
        label = targets[i]
        local ori_pb = 0
        local mi_pb = 0
        local weight = torch.sigmoid(final_weight[label])
        if label == 1 then
          ori_pb = sm_outputs[i][label]
          mi_pb = sm_mioutputs[i][label]
          sm_df_do[i][label] = sm_df_do[i][label]+ weight/(mi_pb+(ori_pb-mi_pb)*weight)
          sm_midf_do[i][label] = sm_midf_do[i][label]+ (1-weight)/(mi_pb+(ori_pb-mi_pb)*weight)
        elseif label < 11 then 
          ori_pb = sm_outputs[i][label]
          mi_pb = sm_mioutputs[i][label+9]
          sm_df_do[i][label] = sm_df_do[i][label] + weight/(mi_pb+(ori_pb-mi_pb)*weight)
          sm_midf_do[i][label+9] = sm_midf_do[i][label+9] + (1-weight)/(mi_pb+(ori_pb-mi_pb)*weight)
        elseif label > 10 then
          ori_pb =  sm_outputs[i][label]
          mi_pb = sm_mioutputs[i][label-9]
          sm_df_do[i][label] = sm_df_do[i][label] + weight/(mi_pb+(ori_pb-mi_pb)*weight)
          sm_midf_do[i][label-9] = sm_midf_do[i][label-9] + (1-weight)/(mi_pb+(ori_pb-mi_pb)*weight)
        end
        warr_gd[label] = warr_gd[label] + (ori_pb-mi_pb)*weight*(1-weight)/(mi_pb+(ori_pb-mi_pb)*weight)
      end
      warr_gd = -warr_gd/N
      sm_df_do = -sm_df_do/N
      sm_midf_do = -sm_midf_do/N
      --sm_df_do:zero()
      --sm_midf_do:zero()

      -- compute gradients
      local df_do = criterion:backward(outputs, targets)
      local midf_do = micriterion:backward(mioutputs, mitargets)
      model:backward({inputs, input_p1s, input_p2s, indep, indep_p1s, indep_p2s, inputs, input_p2s, input_p1s, indep, indep_p2s, indep_p1s}, {df_do, midf_do, sm_df_do, sm_midf_do})

      if opt.model_type == 'static' then
        -- don't update embeddings for static model
        layers.w2v.gradWeight:zero()
		    layers.p2v1.gradWeight:zero()
		    layers.p2v2.gradWeight:zero()
      elseif opt.model_type == 'multichannel' then
        -- keep one embedding channel static
        layers.chan1.gradWeight:zero()
      --elseif opt.model_type == 'nonstatic' then
        --local addw2v = layers.w2v.gradWeight+layers.miw2v.gradWeight
        --layers.w2v.gradWeight = addw2v
        --layers.miw2v.gradWeight = addw2v
        --local addp2v = layers.p2v1.gradWeight+layers.mip2v1.gradWeight+ layers.p2v2.gradWeight+layers.mip2v2.gradWeight
        --layers.p2v1.gradWeight = addp2v
        --layers.mip2v1.gradWeight = addp2v
        --layers.p2v2.gradWeight = addp2v
        --layers.mip2v2.gradWeight = addp2v
        --local adddw2v = layers.dw2v.gradWeight+layers.midw2v.gradWeight
        --layers.dw2v.gradWeight = adddw2v
        --layers.midw2v.gradWeight = adddw2v
        --local adddp2v = layers.dp2v1.gradWeight+layers.midp2v1.gradWeight+ layers.dp2v2.gradWeight+layers.midp2v2.gradWeight
        --layers.dp2v1.gradWeight = adddp2v
        --layers.midp2v1.gradWeight = adddp2v
        --layers.dp2v2.gradWeight = adddp2v
        --layers.midp2v2.gradWeight = adddp2v

      end
	  --print(grads)
      grad2 = warr_gd
      return err, grads
    end

    local func2 = function(x)
      -- reset gradients
      --print(grad2:size())
      return 1, grad2
    end
    -- gradient descent
    optim_method(func, params, config, state)
    optim_method(func2, final_weight, config2)
    -- reset padding embedding to zero
	  -- print(layers.w2v.weight:size())
	  -- print(layers.p2v.weight:size())
    layers.w2v.weight[1]:zero()
	  layers.p2v1.weight[1]:zero()
	  layers.p2v2.weight[1]:zero()
    layers.dw2v.weight[1]:zero()
    layers.dp2v1.weight[1]:zero()
    layers.dp2v2.weight[1]:zero()
    layers.miw2v.weight[1]:zero()
    layers.mip2v1.weight[1]:zero()
    layers.mip2v2.weight[1]:zero()
    layers.midw2v.weight[1]:zero()
    layers.midp2v1.weight[1]:zero()
    layers.midp2v2.weight[1]:zero()
	  --print(layers.w2v.weight[10])
	
    if opt.skip_kernel > 0 then
      -- keep skip kernel at zero
      layers.skip_conv.weight:select(3,3):zero()
    end

    -- Renorm (Euclidean projection to L2 ball)
    local renorm = function(row)
      local n = row:norm()
      row:mul(opt.L2s):div(1e-7 + n)
    end

    -- renormalize linear row weights
    local w = layers.linear.weight
    for j = 1, w:size(1) do
      renorm(w[j])
    end

    local w2 = layers.milinear.weight
    for j = 1, w2:size(1) do
      renorm(w2[j])
    end
  end

  if opt.debug == 1 then
    print('Total err: ' .. total_err / train_size)
    print(confusion)
  end

  -- time taken
  time = timer:time().real - time
  time = opt.batch_size * time / train_size
  if opt.debug == 1 then
    print("==> time to learn 1 batch = " .. (time*1000) .. 'ms')
  end

  -- return error percent
  confusion:updateValids()
  return final_weight, confusion.totalValid
end

function Trainer:test(test_data, test_e1, test_e2, test_dep, test_dep_e1, test_dep_e2, test_labels, model, criterion, final_weight)
  model:evaluate()

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end
  
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local test_size = test_data:size(1)
  local local_final_weight = torch.sigmoid(final_weight)

  local total_err = 0
  
  local test_predict = {}
  local test_target = {}

  for t = 1, test_size, opt.batch_size do
    -- data samples and labels, in mini batches.
    local batch_size = math.min(opt.batch_size, test_size - t + 1)
    local inputs = test_data:narrow(1, t, batch_size)
	  local input_p1s = test_e1:narrow(1, t, batch_size)
	  local input_p2s = test_e2:narrow(1, t, batch_size)
    local indep = test_dep:narrow(1, t, batch_size)
    local indep_p1s = test_dep_e1:narrow(1, t, batch_size)
    local indep_p2s = test_dep_e2:narrow(1, t, batch_size)
    local targets = test_labels:narrow(1, t, batch_size)
    if opt.cudnn == 1 then
      inputs = inputs:cuda()
      targets = targets:cuda()
    else
      inputs = inputs:double()
	    input_p1s = input_p1s:double()
	    input_p2s = input_p2s:double()
      indep = indep:double()
      indep_p1s = indep_p1s:double()
      indep_p2s = indep_p2s:double()
      targets = targets:double()
    end

    local output = model:forward({inputs, input_p1s, input_p2s, indep, indep_p1s, indep_p2s, inputs, input_p2s, input_p1s, indep, indep_p2s, indep_p1s})
    local outputs, mioutputs, sm_outputs, sm_mioutputs = unpack(output)
    local final_outputs = torch.zeros(batch_size, opt.num_classes)
    local final_index = torch.zeros(batch_size)
    for i = 1, batch_size do
      for j = 1, opt.num_classes do
        if j==1 then
          final_outputs[i][j] = local_final_weight[j]*sm_outputs[i][j] + (1-local_final_weight[j])*sm_mioutputs[i][j]
        elseif j<11 then
          final_outputs[i][j] = local_final_weight[j]*sm_outputs[i][j] + (1-local_final_weight[j])*sm_mioutputs[i][j+9]
        else
          final_outputs[i][j] = local_final_weight[j]*sm_outputs[i][j] + (1-local_final_weight[j])*sm_mioutputs[i][j-9]
        end
      end
    end
    for i = 1, batch_size do
      outputvalue, outputindice = torch.max(final_outputs[i], 1)
      final_index[i] = outputindice[1]
      table.insert(test_predict, outputindice)
    end
    for i = 1, batch_size do
      table.insert(test_target, targets[i])
    end
    local err = criterion:forward(outputs, targets)
    total_err = total_err + err * batch_size

    for i = 1, batch_size do
      confusion:add(final_index[i], targets[i])
    end
  end

  if opt.debug == 1 then
    print(confusion)
    print('Total err: ' .. total_err / test_size)
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid, test_predict, test_target
end

return Trainer
