import numpy as np
import logging

from dcca_tensorflow.utils import load_data, svm_classify
from dcca_tensorflow.linear_cca import linear_cca
from dcca_tensorflow.models import create_model
from keras.callbacks import ModelCheckpoint


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DCCA:

    outdim_size = 10
    learning_rate = 1e-3
    epoch_num = 100
    batch_size = 64

    # the regularization parameter of the network
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = False


    def __init__(self, apply_linear_cca=True, output_dim=outdim_size, dropout=None):
        self.apply_linear_cca = apply_linear_cca
        self.output_dim = output_dim
        self.dropout = dropout

        # number of layers with nodes in each one
        self.layer_sizes1 = [1024, 1024, 1024, self.output_dim]
        self.layer_sizes2 = [1024, 1024, 1024, self.output_dim]


    def train_dcca(self, data_view_1=None, data_view_2=None, outfile_prefix="",
                   num_epochs=epoch_num, batch_size=batch_size,
                   models_path="./"):

        logger.info("[Deep CCA] - Training DCCA. Apply Linear CCA: {}".format(self.apply_linear_cca))

        weights_file = models_path + "{}-weights_dcca_linear-cca-{}_bs-{}_outdim-{}_epochs-{}_dropout-{}_lr-{}_reg-{}.h5".format(
                                                                                     outfile_prefix,
                                                                                     self.apply_linear_cca,
                                                                                     batch_size, self.output_dim,
                                                                                     self.epoch_num,
                                                                                     self.dropout,
                                                                                     self.learning_rate,
                                                                                     self.reg_par
                                                                                     )
        input_shape1 = data_view_1[0][0].shape[1]
        input_shape2 = data_view_2[0][0].shape[1]

        print("Input shapes View 1: {} - View 2: {}".format(input_shape1, input_shape2))

        model = create_model(self.layer_sizes1, self.layer_sizes2, input_shape1, input_shape2,
                             DCCA.learning_rate, DCCA.reg_par, self.output_dim, DCCA.use_all_singular_values,
                             dropout=self.dropout)

        model = self.train_model(model, data_view_1, data_view_2, num_epochs, batch_size,
                                 weights_file=weights_file)
        new_data = self.test_model(model, data_view_1, data_view_2, self.output_dim, self.apply_linear_cca)

        return new_data



    def train_model(self, model, data1, data2, epoch_num, batch_size, weights_file=None):

        # Unpacking the data
        train_set_x1, train_set_y1 = data1[0]
        valid_set_x1, valid_set_y1 = data1[1]
        test_set_x1, test_set_y1 = data1[2]

        train_set_x2, train_set_y2 = data2[0]
        valid_set_x2, valid_set_y2 = data2[1]
        test_set_x2, test_set_y2 = data2[2]

        print("View 1 summary", train_set_x1.shape, valid_set_x1.shape, test_set_x1.shape)
        print("View 2 summary", train_set_x2.shape, valid_set_x2.shape, test_set_x2.shape)


        checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True,
                                       save_weights_only=True)

        # used dummy Y because labels are not used in the loss function
        model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)),
                  batch_size=batch_size, epochs=epoch_num, shuffle=True,
                  validation_data=([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))),
                  callbacks=[checkpointer])

        model.load_weights(weights_file)

        results = model.evaluate([test_set_x1, test_set_x2], np.zeros(len(test_set_x1)), batch_size=batch_size,
                                 verbose=1)

        print('loss on test data: ', results)

        results = model.evaluate([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1)), batch_size=batch_size,
                                 verbose=1)
        print('loss on validation data: ', results)
        return model



    def test_model(self, model, data1, data2, outdim_size, apply_linear_cca):

        new_data = []
        for k in range(3):
            pred_out = model.predict([data1[k][0], data2[k][0]])
            r = int(pred_out.shape[1] / 2)
            new_data.append([pred_out[:, :r], pred_out[:, r:], data1[k][1]])

        # based on the DCCA paper, a linear CCA should be applied on the output of the networks because
        # the loss function actually estimates the correlation when a linear CCA is applied to the output of the networks
        # however it does not improve the performance significantly
        if apply_linear_cca:
            w = [None, None]
            m = [None, None]
            print("Linear CCA started!")
            w[0], w[1], m[0], m[1] = linear_cca(new_data[0][0], new_data[0][1], outdim_size)
            print("Linear CCA ended!")

            # Something done in the original MATLAB implementation of DCCA, do not know exactly why;)
            # it did not affect the performance significantly on the noisy MNIST dataset
            # s = np.sign(w[0][0,:])
            # s = s.reshape([1, -1]).repeat(w[0].shape[0], axis=0)
            # w[0] = w[0] * s
            # w[1] = w[1] * s
            ###

            for k in range(3):
                data_num = len(new_data[k][0])
                for v in range(2):
                    new_data[k][v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)
                    new_data[k][v] = np.dot(new_data[k][v], w[v])

        return new_data


if __name__ == '__main__':
    dcca = DCCA()
    dcca.train_dcca()
