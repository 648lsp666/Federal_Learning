import pickle

with open('result/evalinfo.pkl','rb') as f:
  info=pickle.load(f)

print(info[0]['comm_size_list'])