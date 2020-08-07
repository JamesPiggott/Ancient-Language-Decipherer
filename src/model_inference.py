import numpy as np


from tensorflow import keras
model = keras.models.load_model('../models/short_model')

model.summary()



#     min_dist=100

#     for (name,db_enc) in dico_hiero.items():
#         for encoding in db_enc:
#             dist=np.linalg.norm(image-encoding[0])


#             if dist<min_dist:
#                 min_dist = dist
#                 identity=name

#     #if min_dist>0.5:
#         #identity='UNKNOWN'

#     return min_dist, identity

# #####TEST RECOGNITION#####

# correct_label=0
# ntest=test_hiero.shape[0]

# for i in range(ntest):
#     dist, hieroglyph = which_hiero(test_hiero[i], dico_hiero)
#     if labels_true[ntrain+i] == hieroglyph:
#         correct_label += 1

# accuracy=float(correct_label/ntest)*100
# print("Accuracy : {:2.2f}%".format(accuracy))

# import matplotlib.pyplot as plt
# from matplotlib import gridspec


# fig = plt.figure(figsize=(8, 8))
# plt.ioff()

# # gridspec inside gridspec
# outer_grid = gridspec.GridSpec(3, 3, wspace=0.05, hspace=0.05)

# for i in range(9):
#     dist, hieroglyph = which_hiero(test_hiero[i+5], dico_hiero)
#     print("True Hieroglyph : " ,labels_true[ntrain+i+5],"// Predicted : " ,hieroglyph, "dist : ", dist)

#     inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)

#     ax = plt.Subplot(fig, inner_grid[0])
#     ax.imshow(test_data[i+5].reshape(img_height, img_width),cmap='gray')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     fig.add_subplot(ax)
#     ax = plt.Subplot(fig, inner_grid[1])
#     index=dico_hiero[hieroglyph][1][1]
#     ax.imshow(train_data[0][index].reshape(img_height, img_width),cmap='gray')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.text(-32,-8, 'Dissimilarity : {:.2f}'.format(dist), style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
#     fig.add_subplot(ax)

# plt.suptitle("Left : Input Hieroglyph // Right : Predicted class Accuracy : {:2.2f}%".format(accuracy))
# #plt.show()

# fig.savefig('screenshots/results2.png')

# plt.close(fig)