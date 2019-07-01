
from dcca_tensorflow.deepcca import DCCA
from dcca_tensorflow.utils import load_data, svm_classify


model_name = "deepcca"
outfile_prefix = "mnist_test"
apply_linear_cca = True
num_epochs = 10
output_dim = 100
dropout = 0.00
batch_size = 200



(X_train_1, Y_train_1), (X_val_1, Y_val_1), (X_test_1, Y_test_1) = \
            load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
(X_train_2, Y_train_2), (X_val_2, Y_val_2), (X_test_2, Y_test_2) = \
            load_data('noisymnist_view2.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz')



dcca = DCCA(apply_linear_cca=apply_linear_cca, output_dim=output_dim, dropout=dropout)

# Pack Data
train_view_1 = (X_train_1, Y_train_1)
val_view_1 = (X_val_1, Y_val_1)
test_view_1 = (X_test_1, Y_test_1)
data_view_1 = (train_view_1, val_view_1, test_view_1)

train_view_2 = (X_train_2, Y_train_2)
val_view_2 = (X_val_2, Y_val_2)
test_view_2 = (X_test_2, Y_test_2)
data_view_2 = (train_view_2, val_view_2, test_view_2)

projected_feats = dcca.train_dcca(data_view_1=data_view_1, data_view_2=data_view_2,
                                  outfile_prefix=outfile_prefix, num_epochs=num_epochs,
                                  batch_size=batch_size)


print("Projected Image summary", projected_feats[0][0].shape, projected_feats[1][0].shape, projected_feats[2][0].shape)
print("Projected Text summary", projected_feats[0][1].shape, projected_feats[1][1].shape, projected_feats[2][1].shape)


# Training and testing of SVM with linear kernel on the view 1 with new features
[test_acc, valid_acc] = svm_classify(projected_feats, C=0.01)
print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
print("Accuracy on view 1 (test data) is:", test_acc * 100.0)
