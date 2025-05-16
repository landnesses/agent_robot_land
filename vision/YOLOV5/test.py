mu_distDZ = [2,2,2,2,500,2,2,200,2]
mu_distDZ.sort()
new_mu_distDZ=[]
for i in range(0,len(mu_distDZ)-3):
    new_mu_distDZ.append(mu_distDZ[i])
Average = sum(new_mu_distDZ) / len(new_mu_distDZ)
distDZ = Average
print(distDZ)
# 使用9个点的深度去除显著偏差取平均来表示深度 END