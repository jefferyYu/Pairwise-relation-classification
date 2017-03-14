require 'torch'
require 'nn'
require 'nngraph'

local ModelBuilder = torch.class('ModelBuilder')

function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function ModelBuilder:make_net(w2v, p2v)
  if opt.cudnn == 1 then
    require 'cudnn'
    require 'cunn'
  end

  local input = nn.Identity()()
  local input_p1 = nn.Identity()()
  local input_p2 = nn.Identity()()
  local indep = nn.Identity()()
  local indep_p1s = nn.Identity()()
  local indep_p2s = nn.Identity()()
  local miinput = nn.Identity()()
  local miinput_p1 = nn.Identity()()
  local miinput_p2 = nn.Identity()()
  local miindep = nn.Identity()()
  local miindep_p1s = nn.Identity()()
  local miindep_p2s = nn.Identity()()

  local lookup
  local ps_lookup1
  local ps_lookup2
  local dep_lookup
  local dep_pslookup1
  local dep_pslookup2
  local milookup
  local mips_lookup1
  local mips_lookup2
  local midep_lookup
  local midep_pslookup1
  local midep_pslookup2

  if opt.model_type == 'multichannel' then
    local channels = {}
    for i = 1, 2 do
      local chan = nn.LookupTable(opt.vocab_size, opt.vec_size)
      chan.weight:copy(w2v)
      chan.weight[1]:zero() --padding should always be 0
      chan.name = 'channel' .. i
      table.insert(channels, chan(input))
    end
    lookup = channels
  else
    lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
	  ps_lookup1 = nn.LookupTable(opt.p_size, opt.pv_size)
	  ps_lookup2 = nn.LookupTable(opt.p_size, opt.pv_size)
    dep_lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
    dep_pslookup1 = nn.LookupTable(opt.p_size, opt.pv_size)
    dep_pslookup2 = nn.LookupTable(opt.p_size, opt.pv_size)  
    milookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
    mips_lookup1 = nn.LookupTable(opt.p_size, opt.pv_size)
    mips_lookup2 = nn.LookupTable(opt.p_size, opt.pv_size)
    midep_lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
    midep_pslookup1 = nn.LookupTable(opt.p_size, opt.pv_size)
    midep_pslookup2 = nn.LookupTable(opt.p_size, opt.pv_size) 

    if opt.model_type == 'static' or opt.model_type == 'nonstatic' then
      lookup.weight:copy(w2v)
	    ps_lookup1.weight:copy(p2v)
	    ps_lookup2.weight:copy(p2v)
      dep_lookup.weight:copy(w2v)
      dep_pslookup1.weight:copy(p2v)
      dep_pslookup2.weight:copy(p2v)
      milookup.weight:copy(w2v)
      mips_lookup1.weight:copy(p2v)
      mips_lookup2.weight:copy(p2v)
      midep_lookup.weight:copy(w2v)
      midep_pslookup1.weight:copy(p2v)
      midep_pslookup2.weight:copy(p2v)
    else
      -- rand
      lookup.weight:uniform(-0.25, 0.25)
	    ps_lookup1.weight:copy(p2v)
	    ps_lookup2.weight:copy(p2v)
      dep_lookup.weight:uniform(-0.25, 0.25)
      dep_pslookup1.weight:copy(p2v)
      dep_pslookup2.weight:copy(p2v)
      milookup.weight:uniform(-0.25, 0.25)
      mips_lookup1.weight:copy(p2v)
      mips_lookup2.weight:copy(p2v)
      midep_lookup.weight:uniform(-0.25, 0.25)
      midep_pslookup1.weight:copy(p2v)
      midep_pslookup2.weight:copy(p2v)
    end
    -- padding should always be 0
    lookup.weight[1]:zero()
	  ps_lookup1.weight[1]:zero()
	  ps_lookup2.weight[1]:zero()
    dep_lookup.weight[1]:zero()
    dep_pslookup1.weight[1]:zero()
    dep_pslookup2.weight[1]:zero()
    milookup.weight[1]:zero()
    mips_lookup1.weight[1]:zero()
    mips_lookup2.weight[1]:zero()
    midep_lookup.weight[1]:zero()
    midep_pslookup1.weight[1]:zero()
    midep_pslookup2.weight[1]:zero()

    lookup = lookup(input)
	  ps_lookup1 = ps_lookup1(input_p1)
	  ps_lookup2 = ps_lookup2(input_p2)
    dep_lookup = dep_lookup(indep)
    dep_pslookup1 = dep_pslookup1(indep_p1s)
    dep_pslookup2 = dep_pslookup2(indep_p2s)
    milookup = milookup(miinput)
    mips_lookup1 = mips_lookup1(miinput_p1)
    mips_lookup2 = mips_lookup2(miinput_p2)
    midep_lookup = midep_lookup(miindep)
    midep_pslookup1 = midep_pslookup1(miindep_p1s)
    midep_pslookup2 = midep_pslookup2(miindep_p2s)

  end

  -- kernels is an array of kernel sizes
  local kernels = opt.kernels
  local layer1 = {}
  local milayer1 = {}
  for i = 1, #kernels do
    local conv, miconv
	  local input_layer, input_layer1, miinput_layer, miinput_layer1
    local conv_layer, miconv_layer
    local max_time, mimax_time

    if opt.model_type == 'multichannel' then
      local lookup_conv = {}
      for chan = 1,2 do
        table.insert(lookup_conv, conv(lookup[chan]))
      end
      conv_layer = nn.CAddTable()(lookup_conv)
      max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
    else
      conv = nn.TemporalConvolution(opt.vec_size+opt.pv_size+opt.pv_size, opt.num_feat_maps, kernels[i])
      miconv = nn.TemporalConvolution(opt.vec_size+opt.pv_size+opt.pv_size, opt.num_feat_maps, kernels[i])
	    --share_params(conv, miconv)
      input_layer1 = nn.JoinTable(3)({lookup, ps_lookup1})
	    input_layer = nn.JoinTable(3)({input_layer1, ps_lookup2})
      miinput_layer1 = nn.JoinTable(3)({milookup, mips_lookup1})
      miinput_layer = nn.JoinTable(3)({miinput_layer1, mips_lookup2})
      conv_layer = conv(input_layer)
      miconv_layer = miconv(miinput_layer)
      max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
      mimax_time = nn.Max(2)(nn.ReLU()(miconv_layer))
    end

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    miconv.weight:uniform(-0.01, 0.01)
    miconv.bias:zero()
    table.insert(layer1, max_time)
    table.insert(milayer1, mimax_time)
  end

  local dkernels = opt.dkernels
  local dlayer1 = {}
  local midlayer1 = {}
  for j = 1, #dkernels do
    local conv, miconv
    local input_layer, input_layer1, miinput_layer, miinput_layer1
    local conv_layer, miconv_layer
    local max_time, mimax_time
    conv = nn.TemporalConvolution(opt.vec_size+opt.pv_size+opt.pv_size, opt.num_feat_maps, dkernels[j])
    miconv = nn.TemporalConvolution(opt.vec_size+opt.pv_size+opt.pv_size, opt.num_feat_maps, dkernels[j])    
    --share_params(conv, miconv)
    input_layer1 = nn.JoinTable(3)({dep_lookup, dep_pslookup1})
    input_layer = nn.JoinTable(3)({input_layer1, dep_pslookup2})
    miinput_layer1 = nn.JoinTable(3)({midep_lookup, midep_pslookup1})
    miinput_layer = nn.JoinTable(3)({miinput_layer1, midep_pslookup2})
    conv_layer = conv(input_layer)
    miconv_layer = miconv(miinput_layer)
    max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
    mimax_time = nn.Max(2)(nn.ReLU()(miconv_layer))

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    miconv.weight:uniform(-0.01, 0.01)
    miconv.bias:zero()
    table.insert(dlayer1, max_time)
    table.insert(midlayer1, mimax_time)
  end


  local conv_layer_concat, dconv_layer_concat
  local miconv_layer_concat, midconv_layer_concat
  if #layer1 > 1 then
    conv_layer_concat = nn.JoinTable(2)(layer1)
    miconv_layer_concat = nn.JoinTable(2)(milayer1)
  else
    conv_layer_concat = layer1[1]
    miconv_layer_concat = milayer1[1]
  end
  dconv_layer_concat = dlayer1[1]
  midconv_layer_concat = midlayer1[1]

  local last_layer = nn.JoinTable(2)({conv_layer_concat,dconv_layer_concat})

  local milast_layer = nn.JoinTable(2)({miconv_layer_concat,midconv_layer_concat})

  -- simple MLP layer
  local linear = nn.Linear((#layer1) * opt.num_feat_maps+(#dlayer1) * opt.num_feat_maps, opt.num_classes)
  local milinear = nn.Linear((#layer1) * opt.num_feat_maps+(#dlayer1) * opt.num_feat_maps, opt.num_classes)
  --share_params(linear, milinear)
  linear.weight:normal():mul(0.01)
  linear.bias:zero()
  milinear.weight:normal():mul(0.01)
  milinear.bias:zero()
  --local linear = nn.Linear(opt.num_feat_maps, opt.num_classes)
  --linear.weight:normal():mul(0.01)
  --linear.bias:zero()

  local softmax
  local misoftmax
  if opt.cudnn == 1 then
    softmax = cudnn.LogSoftMax()
  else
    softmax = nn.LogSoftMax()
    misoftmax = nn.LogSoftMax()
  end

  local output = softmax(linear(nn.Dropout(opt.dropout_p)(last_layer))) 
  local mioutput = misoftmax(milinear(nn.Dropout(opt.dropout_p)(milast_layer))) 

  local sm_output = nn.SoftMax()(linear(nn.Dropout(opt.dropout_p)(last_layer)))
  local sm_mioutput = nn.SoftMax()(milinear(nn.Dropout(opt.dropout_p)(milast_layer)))
  model = nn.gModule({input, input_p1, input_p2, indep, indep_p1s, indep_p2s, miinput, miinput_p1, miinput_p2, miindep, miindep_p1s, miindep_p2s}, {output, mioutput, sm_output, sm_mioutput})
  return model
end

return ModelBuilder
