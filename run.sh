for i in 'comb' 'combneg'
do
  echo main_mipe${i}.lua
  th main_mipe${i}.lua -data Semeval_pedep.hdf5 -cudnn 0
done