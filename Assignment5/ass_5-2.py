import numpy as np
from sklearn import preprocessing
# Categorical variables
colors = np.array(["Red", "Green", "Blue", "Green", "Red", "Blue"])
sizes = np.array(["Small", "Medium", "Large", "Small", "Large", "Medium"])
brands = np.array(["Nike", "Adidas", "Puma", "Nike", "Puma", "Adidas"])


# Encoding brands #
unique, idx=np.unique(brands, return_index=True)
unique=unique[idx]
brands_dic={b:i for i,b in enumerate(unique)}
result_brands=np.array([brands_dic[brand] for brand in brands])
print(result_brands)

# Encoding sizes #
size_order = {"Small": 1, "Medium": 2, "Large": 3}
result_sizes=np.array([size_order[sizes[0]],
                      size_order[sizes[1]],
                      size_order[sizes[2]],
                      size_order[sizes[3]],
                      size_order[sizes[4]],
                      size_order[sizes[5]]]
                      )
print(result_sizes)

# Encoding colors #

unique=np.unique(colors)

# fit one-hot encoding
encoding_colors=np.zeros((unique.size,unique.size))
flag=0
for color in unique:
    encoding_colors[flag][flag]=1
    flag+=1

# transform colors using one-hot encoding
result_colors=np.arange(colors.size*unique.size).reshape(colors.size,unique.size)
for cIdx,color in enumerate(colors):
    for uIdx,uni in enumerate(unique):
        if(color==uni):
            result_colors[cIdx]=encoding_colors[uIdx]

print(result_colors)

result=np.hstack((result_sizes.reshape(-1,1),result_brands.reshape(-1,1),result_colors))

print(result)