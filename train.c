#include <stdlib.h>
#include <stdio.h>
#include "nn_layers.h"

int main(){
    srand(13);

    printf("\n==== LOADING TRAINING DATA ====\n");
    dataset ds = read_data("archive/bin_data/train/train-images.idx3-ubyte", "archive/bin_data/train/train-labels.idx1-ubyte");


    printf("\n==== LOADING TEST DATA ====\n");
    dataset test_data = read_data("archive/bin_data/test/t10k-images.idx3-ubyte", "archive/bin_data/test/t10k-labels.idx1-ubyte");


    int nr_of_neurons = 200;
    int nr_of_input = ds.data_size;

    layer hidden_layer = create_ran_layer(nr_of_neurons, nr_of_input);
    layer output_layer = create_ran_layer(10, hidden_layer.nr_of_neurons);

    float eta = 0.02;


    // KEEP BATCH AT 1 BECAUSE update_hidden_weights_and_bias DOSENT WORK WITH THE i INPUT PROPERRLY
    
    int epochs = 20;

    printf("\n==== TRAINING AND TESTING====\n");
    for(int run = 0; run < epochs; run++){
        float accuracy_train = train_NN(ds, &hidden_layer, &output_layer, eta);
        float accuracy = test_on_data(test_data, &hidden_layer, &output_layer);
        // printf("Accuracy: %.2f%%\n", accuracy_train);
        // printf("accyracy test: %.2f%%\n\n", accuracy);
        printf("epoch %d: train=%.2f%%, test=%.2f%%, diff=%.2f, eta=%.5f\n", run + 1, accuracy_train, accuracy, accuracy_train - accuracy, eta);
        eta *= 0.95;
    }
    
    save_data(ds, &hidden_layer, &output_layer);
    
    free_layer(hidden_layer);
    free_layer(output_layer);
    free_dataset(test_data);
    free_dataset(ds);

    return 0;
}
