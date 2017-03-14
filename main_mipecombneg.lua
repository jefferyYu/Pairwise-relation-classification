require 'hdf5'
require 'nn'
require 'optim'
require 'lfs'
require 'sys'
-- Flags
cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Convolutional net for sentence classification')
cmd:text()
cmd:text('Options')
cmd:option('-model_type', 'nonstatic', 'Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')
cmd:option('-data', 'MR.hdf5', 'Training data and word2vec data')
cmd:option('-cudnn', 0, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
cmd:option('-seed', 0, 'random seed, set -1 for actual random')
cmd:option('-folds', 1, 'number of folds to use. If test set provided, folds=1. max 10')
cmd:option('-debug', 0, 'print debugging info including timing, confusions')
cmd:option('-gpuid', 1, 'GPU device id to use.')
cmd:option('-savefile', '', 'Name of output file, which will hold the trained model, model parameters, and training scores. Default filename is TIMESTAMP_results')
cmd:option('-zero_indexing', 0, 'If data is zero indexed')
cmd:text()

-- Preset by preprocessed data
cmd:option('-has_test', 1, 'If data has test, we use it. Otherwise, we use CV on folds')
cmd:option('-has_dev', 1, 'If data has dev, we use it, otherwise we split from train')
cmd:option('-num_classes', 2, 'Number of output classes')
cmd:option('-max_sent', 59, 'maximum sentence length')
cmd:option('-vec_size', 300, 'word2vec vector size')
cmd:option('-vocab_size', 18766, 'Vocab size')
cmd:option('-p_size', 182, 'position size')
cmd:option('-pv_size', 50, 'p2v size')
cmd:text()

-- Training own dataset
cmd:option('-train_only', 0, 'Set to 1 to only train on data. Default is cross-validation')
cmd:option('-test_only', 0, 'Set to 1 to only do testing. Must have a -warm_start_model')
cmd:option('-warm_start_model', '', 'Path to .t7 file with pre-trained model. Should contain a table with key \'model\'')
cmd:text()

-- Training hyperparameters
cmd:option('-num_epochs', 15, 'Number of training epochs')
cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
cmd:option('-L2s', 3, 'L2 normalize weights')
cmd:option('-batch_size', 50, 'Batch size for training')
cmd:text()

-- Model hyperparameters
cmd:option('-num_feat_maps', 150, 'Number of feature maps after 1st convolution')
cmd:option('-kernels', '{2,3,4,5}', 'Kernel sizes of convolutions, table format.')
cmd:option('-dkernels', '{5}', 'Kernel sizes of convolutions, table format.')
cmd:option('-skip_kernel', 0, 'Use skip kernel')
cmd:option('-dropout_p', 0.5, 'p for dropout')
cmd:option('-highway_mlp', 0, 'Number of highway MLP layers')
cmd:option('-highway_conv_layers', 0, 'Number of highway MLP layers')
cmd:text()

function get_layer2(model, name)
  local named_layer = {}
  function get(layer)
    if torch.typename(layer) == name or layer.name == name then
      table.insert(named_layer,layer)
    end
  end

  model:apply(get)
  return named_layer
end

function get_layer(model, name)
  local named_layer
  function get(layer)
    if torch.typename(layer) == name or layer.name == name then
      named_layer = layer
    end
  end

  model:apply(get)
  return named_layer
end

-- build model for training
function build_model(w2v, p2v)
  local ModelBuilder = require 'model.convNN_mipe4gateneg'
  local model_builder = ModelBuilder.new()

  local model
  if opt.warm_start_model == '' then
    model = model_builder:make_net(w2v, p2v)
  else
    require "nngraph"
    if opt.cudnn == 1 then
      require "cudnn"
      require "cunn"
    end
    model = torch.load(opt.warm_start_model).model
  end

  local criterion = nn.ClassNLLCriterion()
  local micriterion = nn.ClassNLLCriterion()
  local finalcriterion = nn.ClassNLLCriterion()

  -- move to GPU
  if opt.cudnn == 1 then
    model = model:cuda()
    criterion = criterion:cuda()
  end

  -- get layers
  local layers = {}
  layer = get_layer2(model, 'nn.Linear')
  layers['linear'] = layer[1]
  layers['milinear'] = layer[2]
  layer_num = get_layer2(model, 'nn.LookupTable')
  layers['w2v'] = layer_num[1] 
  layers['p2v1'] = layer_num[2]
  layers['p2v2'] = layer_num[3]
  layers['dw2v'] = layer_num[4] 
  layers['dp2v1'] = layer_num[5]
  layers['dp2v2'] = layer_num[6]
  layers['miw2v'] = layer_num[7] 
  layers['mip2v1'] = layer_num[8]
  layers['mip2v2'] = layer_num[9]
  layers['midw2v'] = layer_num[10] 
  layers['midp2v1'] = layer_num[11]
  layers['midp2v2'] = layer_num[12]

  if opt.skip_kernel > 0 then
    layers['skip_conv'] = get_layer(model, 'skip_conv')
  end
  if opt.model_type == 'multichannel' then
    layers['chan1'] = get_layer(model, 'channel1')
  end

  return model, criterion, layers, micriterion, finalcriterion
end

function train_loop(all_train, all_train_e1, all_train_e2, all_train_label, all_mi_train_label, test, test_e1, test_e2, test_label, mitest_label, dev, dev_e1, dev_e2, dev_label, midev_label, w2v, p2v, train_dep, train_dep_e1, train_dep_e2, dev_dep, dev_dep_e1, dev_dep_e2,test_dep, test_dep_e1, test_dep_e2)
  -- Initialize objects
  local Trainer = require 'trainer_mipe4gateneg'
  local trainer = Trainer.new()

  local optim_method
  if opt.optim_method == 'adadelta' then
    optim_method = optim.adadelta
  elseif opt.optim_method == 'adam' then
    optim_method = optim.adam
  end

  local best_model -- save best model
  local fold_dev_scores = {}
  local fold_test_scores = {}

  local train, train_e1, train_e2, train_label, mitrain_label -- training set for each fold
  if opt.has_test == 1 then
    train = all_train
	  train_e1 = all_train_e1
	  train_e2 = all_train_e2
    train_dep = train_dep
    train_dep_e1 = train_dep_e1
    train_dep_e2 = train_dep_e2
    train_label = all_train_label
    mitrain_label = all_mi_train_label
  end

  local final_best_weight = torch.ones(opt.num_classes)

  -- Training folds.
  for fold = 1, opt.folds do
    local timer = torch.Timer()
    local fold_time = timer:time().real

    print()
    print('==> fold ', fold)

    if opt.has_dev == 0 then
      -- shuffle train to get dev/train split (10% to dev)
      -- We organize our data in batches at this split before epoch training.
      local J = train:size(1)
      local shuffle = torch.randperm(J):long()
      train = train:index(1, shuffle)
	    train_e1 = train_e1:index(1, shuffle)
	    train_e2 = train_e2:index(1, shuffle)
      train_dep = train_dep:index(1, shuffle)
      train_dep_e1 = train_dep_e1:index(1, shuffle)
      train_dep_e2 = train_dep_e2:index(1, shuffle)
      train_label = train_label:index(1, shuffle)
      mitrain_label = mitrain_label:index(1, shuffle)

      local num_batches = math.floor(J / opt.batch_size)
      local num_train_batches = torch.round(num_batches * 0.9)

      local train_size = num_train_batches * opt.batch_size
      --local train_size = 6 * opt.batch_size
      local dev_size = J - train_size
      --local dev_size = 2 * opt.batch_size
      dev = train:narrow(1, train_size+1, dev_size)
	    dev_e1 = train_e1:narrow(1, train_size+1, dev_size)
	    dev_e2 = train_e2:narrow(1, train_size+1, dev_size)
      dev_dep = train_dep:narrow(1, train_size+1, dev_size)
      dev_dep_e1 = train_dep_e1:narrow(1, train_size+1, dev_size)
      dev_dep_e2 = train_dep_e2:narrow(1, train_size+1, dev_size) 
      dev_label = train_label:narrow(1, train_size+1, dev_size)
      midev_label = mitrain_label:narrow(1, train_size+1, dev_size)
      train = train:narrow(1, 1, train_size)
	    train_e1 = train_e1:narrow(1, 1, train_size)
	    train_e2 = train_e2:narrow(1, 1, train_size)
      train_dep = train_dep:narrow(1, 1, train_size)
      train_dep_e1 = train_dep_e1:narrow(1, 1, train_size)
      train_dep_e2 = train_dep_e2:narrow(1, 1, train_size)
      train_label = train_label:narrow(1, 1, train_size)
      mitrain_label = mitrain_label:narrow(1, 1, train_size)
    end
    
    --neg variable is an indicator variable
    local neg = 1
    local percent = 0.2
    local neg_list = {}
    local pos_list = {}
    for i = 1, train:size(1) do
      if train_label[i] == 1 then 
        table.insert(neg_list, i)
      else 
        table.insert(pos_list, i)
      end
    end

    local neg_tensor = torch.Tensor(neg_list)
    local neg_num = math.floor(#neg_list * percent)
    local pos_num = #pos_list
    local pos_tensor = torch.Tensor(pos_list)

    print('negative samples  ' .. #neg_list)
    print('used mirror negative samples  ' .. neg_num)
    print('positive samples  ' .. pos_num)

    local neg_mirrortrain_tensor = neg_tensor[{{1,neg_num}}]

    local mirror_num = pos_num+neg_num

    local mirrortrain = torch.Tensor(mirror_num, train:size(2))
    local mirrortrain_e1 = torch.Tensor(mirror_num, train:size(2))
    local mirrortrain_e2 = torch.Tensor(mirror_num, train:size(2))
    local mirrortrain_dep = torch.Tensor(mirror_num, train:size(2))
    local mirrortrain_dep_e1 = torch.Tensor(mirror_num, train:size(2))
    local mirrortrain_dep_e2 = torch.Tensor(mirror_num, train:size(2))
    local mirrortrain_label = torch.Tensor(mirror_num)
    local mirrormitrain_label = torch.Tensor(mirror_num)

    if neg == 1 then

      for k =1, neg_num do
        mirrortrain[k] = train[neg_mirrortrain_tensor[k]]
        mirrortrain_e1[k] = train_e1[neg_mirrortrain_tensor[k]]
        mirrortrain_e2[k] = train_e2[neg_mirrortrain_tensor[k]]
        mirrortrain_dep[k] = train_dep[neg_mirrortrain_tensor[k]]
        mirrortrain_dep_e1[k] = train_dep_e1[neg_mirrortrain_tensor[k]]
        mirrortrain_dep_e2[k] = train_dep_e2[neg_mirrortrain_tensor[k]]
        mirrortrain_label[k] = train_label[neg_mirrortrain_tensor[k]]
        mirrormitrain_label[k] = mitrain_label[neg_mirrortrain_tensor[k]]
      end

      for k =1, pos_num do
        mirrortrain[neg_num+k] = train[pos_tensor[k]]
        mirrortrain_e1[neg_num+k] = train_e1[pos_tensor[k]]
        mirrortrain_e2[neg_num+k] = train_e2[pos_tensor[k]]
        mirrortrain_dep[neg_num+k] = train_dep[pos_tensor[k]]
        mirrortrain_dep_e1[neg_num+k] = train_dep_e1[pos_tensor[k]]
        mirrortrain_dep_e2[neg_num+k] = train_dep_e2[pos_tensor[k]]
        mirrortrain_label[neg_num+k] = train_label[pos_tensor[k]]
        mirrormitrain_label[neg_num+k] = mitrain_label[pos_tensor[k]]
      end

      local shuffle = torch.randperm(pos_num+neg_num):long()
      mirrortrain = mirrortrain:index(1, shuffle)
      mirrortrain_e1 = mirrortrain_e1:index(1, shuffle)
      mirrortrain_e2 = mirrortrain_e2:index(1, shuffle)
      mirrortrain_dep = mirrortrain_dep:index(1, shuffle)
      mirrortrain_dep_e1 = mirrortrain_dep_e1:index(1, shuffle)
      mirrortrain_dep_e2 = mirrortrain_dep_e2:index(1, shuffle)
      mirrortrain_label = mirrortrain_label:index(1, shuffle)
      mirrormitrain_label = mirrormitrain_label:index(1, shuffle)
      --print(index)
      --print(na_index-1)  
    end

    -- build model
    local model, criterion, layers, micriterion, finalcriterion = build_model(w2v, p2v)

    -- Call getParameters once
    local params, grads = model:getParameters()
	--print(params:size())

    -- Training loop.
    best_model = model:clone()
    local best_epoch = 1
    local best_err = 0.0

    -- Training.
    -- Gradient descent state should persist over epochs
    local state = {}
    for epoch = 1, opt.num_epochs do
      local epoch_time = timer:time().real

      -- Train
      local final_weight, train_err = trainer:train(mirrortrain, mirrortrain_e1, mirrortrain_e2, mirrortrain_dep, mirrortrain_dep_e1, mirrortrain_dep_e2, mirrortrain_label, mirrormitrain_label, model, criterion, micriterion, optim_method, layers, state, params, grads, train, train_e1, train_e2, train_dep, train_dep_e1, train_dep_e2, train_label, mitrain_label)
      -- Dev
      local dev_err, dev_predict, dev_target = trainer:test(dev, dev_e1, dev_e2, dev_dep, dev_dep_e1, dev_dep_e2, dev_label, model, criterion, final_weight)
      --print(dev_predict)
      if dev_err >= best_err then
        best_model = model:clone()
        best_epoch = epoch
        best_err = dev_err 
        final_best_weight = final_weight
      end

      if opt.debug == 1 then
        print()
        print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
        print('\n')
      end

      --print('epoch:', epoch, 'train perf:', 100*train_err, '%, val perf ', 100*dev_err, '%')

      local f = 'dev_predict'
      local file = io.open(f, "w")
      io.output(file)
      for i = 1, #dev_predict do
        io.write(dev_predict[i][1])
        io.write('\n')
      end
      file.close()
      local f2 = 'dev_target'
      local file2 = io.open(f2, "w")
      io.output(file2)
      for i = 1, #dev_target do
        io.write(dev_target[i])
        io.write('\n')
      end
      file2.close()
      --print('hehe')
      require 'sys'
      os.execute('python torch_eval.py dev_predict dev_target')   

      local test_err, test_predict, test_target = trainer:test(test, test_e1, test_e2, test_dep, test_dep_e1, test_dep_e2, test_label, model, criterion, final_weight)
      table.insert(fold_test_scores, test_err)
      local f = 'test_predict'
      local file = io.open(f, "w")
      io.output(file)
      for i = 1, #test_predict do
        io.write(test_predict[i][1])
        io.write('\n')
      end
      file.close()
      local f2 = 'test_target'
      local file2 = io.open(f2, "w")
      io.output(file2)
      for i = 1, #test_target do
        io.write(test_target[i])
        io.write('\n')
      end
      file2.close()
      require 'sys'
      os.execute('python torch_eval.py test_predict test_target') 
    end

    --print('best dev err:', 100*best_err, '%, epoch ', best_epoch)
    table.insert(fold_dev_scores, best_err)

    -- Testing.
    if opt.train_only == 0 then
      local test_err, test_predict, test_target = trainer:test(test, test_e1, test_e2, test_dep, test_dep_e1, test_dep_e2, test_label, best_model, criterion, final_best_weight)
      table.insert(fold_test_scores, test_err)
      local f = 'test_predict'
      local file = io.open(f, "w")
      io.output(file)
      for i = 1, #test_predict do
        io.write(test_predict[i][1])
        io.write('\n')
      end
      file.close()
      local f2 = 'test_target'
      local file2 = io.open(f2, "w")
      io.output(file2)
      for i = 1, #test_target do
        io.write(test_target[i])
        io.write('\n')
      end
      file2.close()
      require 'sys'
      os.execute('python torch_eval.py test_predict test_target')
      --print('test perf ', 100*test_err, '%')
      
    end

    if opt.debug == 1 then
      print()
      print('time for one fold: ', (timer:time().real - fold_time * 1000), 'ms')
      print('\n')
    end
  end

  return fold_dev_scores, fold_test_scores, best_model
end

function load_data()
  local train, train_label
  local dev, dev_label
  local test, test_label

  print('loading data...')
  local f = hdf5.open(opt.data, 'r')
  local w2v = f:read('w2v'):all()
  local p2v = f:read('p2v'):all()
  train = f:read('train'):all()
  train_e1 = f:read('train_e1'):all()
  train_e2 = f:read('train_e2'):all()
  train_dep = f:read('train_dep'):all()
  train_dep_e1 = f:read('train_dep_e1'):all()
  train_dep_e2 = f:read('train_dep_e2'):all()
  train_label = f:read('train_label'):all()
  mitrain_label = f:read('mitrain_label'):all()
  opt.num_classes = torch.max(train_label)

  if f:read('dev'):dataspaceSize()[1] == 0 then
    opt.has_dev = 0
  else
    opt.has_dev = 1
    dev = f:read('dev'):all()
	  dev_e1 = f:read('dev_e1'):all()
	  dev_e2 = f:read('dev_e2'):all()
    dev_dep = f:read('dev_dep'):all()
    dev_dep_e1 = f:read('dev_dep_e1'):all()
    dev_dep_e2 = f:read('dev_dep_e2'):all()
    dev_label = f:read('dev_label'):all()
    midev_label = f:read('midev_label'):all()
  end
  if f:read('test'):dataspaceSize()[1] == 0 then
    opt.has_test = 0
  else
    opt.has_test = 1
    test = f:read('test'):all()
	  test_e1 = f:read('test_e1'):all()
	  test_e2 = f:read('test_e2'):all()
    test_dep = f:read('test_dep'):all()
    test_dep_e1 = f:read('test_dep_e1'):all()
    test_dep_e2 = f:read('test_dep_e2'):all()    
    test_label = f:read('test_label'):all()
    mitest_label = f:read('mitest_label'):all()
  end
  print('data loaded!')

  return train, train_e1, train_e2, train_label, mitrain_label, test, test_e1, test_e2, test_label, mitest_label, dev, dev_e1, dev_e2, dev_label, midev_label, w2v, p2v, train_dep, train_dep_e1, train_dep_e2, dev_dep, dev_dep_e1, dev_dep_e2,test_dep, test_dep_e1, test_dep_e2
end

function main()
  -- parse arguments
  opt = cmd:parse(arg)

  if opt.seed ~= -1 then
    torch.manualSeed(opt.seed)
  end
  if opt.cudnn == 1 then
    require 'cutorch'
    if opt.seed ~= -1 then
      cutorch.manualSeedAll(opt.seed)
    end
    cutorch.setDevice(opt.gpuid)
  end

  -- Read HDF5 training data
  local train, train_label
  local test, test_label
  local dev, dev_label
  local w2v
  train, train_e1, train_e2, train_label, mitrain_label, test, test_e1, test_e2, test_label, mitest_label, dev, dev_e1, dev_e2, dev_label, midev_label, w2v, p2v, train_dep, train_dep_e1, train_dep_e2, dev_dep, dev_dep_e1, dev_dep_e2,test_dep, test_dep_e1, test_dep_e2 = load_data()

  opt.vocab_size = w2v:size(1)
  opt.vec_size = w2v:size(2)
  opt.p_size = p2v:size(1)
  opt.pv_size = p2v:size(2)
  
  opt.max_sent = train:size(2)
  print('vocab size: ', opt.vocab_size)
  print('vec size: ', opt.vec_size)
  print('position size: ', opt.p_size)
  print('pvec size: ', opt.pv_size)

  -- Retrieve kernels
  loadstring("opt.kernels = " .. opt.kernels)()
  loadstring("opt.dkernels = " .. opt.dkernels)()
  --print(opt.dkernels)
  --opt.dkernels = {5}

  if opt.zero_indexing == 1 then
    train:add(1)
    train_label:add(1)
    if dev ~= nil then
      dev:add(1)
      dev_label:add(1)
    end
    if test ~= nil then
      test:add(1)
      test_label:add(1)
    end
  end

  if opt.test_only == 1 then
    assert(opt.warm_start_model ~= '', 'must have -warm_start_model for testing')
    assert(opt.has_test == 1)
    local Trainer = require "trainer_mipe4gateneg"
    local trainer = Trainer.new()
    print('Testing...')
    local model, criterion = build_model(w2v, p2v)
    local test_err = trainer:test(test, test_label, model, criterion)
    print('Test score:', test_err)
    os.exit()
  end

  if opt.has_test == 1 or opt.train_only == 1 then
    -- don't do CV if we have a test set, or are training only
    opt.folds = 1
  end

  -- training loop
  local fold_dev_scores, fold_test_scores, best_model = train_loop(train, train_e1, train_e2, train_label, mitrain_label, test, test_e1, test_e2, test_label, mitest_label, dev, dev_e1, dev_e2, dev_label, midev_label, w2v, p2v, train_dep, train_dep_e1, train_dep_e2, dev_dep, dev_dep_e1, dev_dep_e2,test_dep, test_dep_e1, test_dep_e2)

  --print('dev scores:')
  --print(fold_dev_scores)
  --print('average dev score: ', torch.Tensor(fold_dev_scores):mean())

  if opt.train_only == 0 then
    --print('test scores:')
    --print(fold_test_scores)
    --print('average test score: ', torch.Tensor(fold_test_scores):mean())
  end

  -- make sure output directory exists
  if not path.exists('results') then lfs.mkdir('results') end

  local savefile
  if opt.savefile ~= '' then
    savefile = opt.savefile
  else
    savefile = string.format('results/%s_relumodel.t7', os.date('%Y%m%d_%H%M'))
  end
  --print('saving results to ', savefile)

  local save = {}
  save['dev_scores'] = fold_dev_scores
  if opt.train_only == 0 then
    save['test_scores'] = fold_test_scores
  end
  save['opt'] = opt
  save['model'] = best_model
  save['embeddings'] = get_layer(best_model, 'nn.LookupTable').weight
  torch.save(savefile, save)
end

main()
