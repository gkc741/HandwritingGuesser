#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include <stdint.h>

typedef struct layer{
    float* weights; // Long flattended list of weights for every neruon in the layer
    float* bias; // the bias
    int nr_of_neurons;  // the size of the out list basically how many neurons
    int in_size; // the size of the input list
}layer;
/*
The weight list is a flattended list so that means its a 2d array in flattended into row-major 1d list
this means i acces it instead of [i][j] i do, [i * (weight pr neuron / in_size) + j]

The bias is just a long list of all the biases of the neurons
*/

typedef struct dataset{
    float* data;
    int data_size;
    uint8_t* labels;
    int nr_of_images;
} dataset;

typedef struct {
    float* output_list;
    int size;
} output;

typedef struct{
    output hidden_output;
    output output_output;
    output probs;
    float* error_output;
    float* error_hidden;
} workspace;

typedef struct{
    output hidden_output;
    output output_output;
} test_workspace;

typedef struct {
    int input;
    int hidden;
    int output;
} Header;

// Layer creation and cleanup
struct layer create_ran_layer(int nr_of_neurons, int nr_of_inputs);
void free_layer(struct layer layer);

// Dataset reading
struct dataset read_images(const char* file);
uint8_t* read_labels(const char* file);
struct dataset read_data(const char* image_file, const char* label_file);
void free_dataset(struct dataset dataset);

// Activation and math
float relu(float x);
void calc_hidden_layer_output(struct dataset dataset, struct layer hidden_layer, int img_indx, output* input);
void calc_output_layer_output(float* output_of_hidden_layer, struct layer output_layer, output* input);
void calc_softmax_prob(output* output_of_output_layer, output* input);
float calc_loss(dataset dataset, float* predictioin_probabilities, int img_indx);
int argmax(output prob_prediction_list);

// Backpropagation
void output_layer_errors(dataset dataset, output prob_prediction, int img_indx, float* input);
void hidden_layer_errors(output hidden_layer_output, layer output_layer, float* error_list_of_output, float* input);
void update_output_weights_and_bias(layer* output_layer, float* error_list_of_output, float eta, output hidden_layer_output);
void update_hidden_weights_and_bias(layer* hidden_layer, float* error_list_of_hidden, float eta, float* avg_pixl_values);

// Training and testing
void free_workspace(workspace* ws);
float train_NN(dataset dataset, layer* hidden_layer, layer* output_layer, float eta);
float test_on_data(dataset dataset, layer* hidden_layer, layer* output_layer);

// I/O
void save_data(dataset ds, layer* hidden_layer, layer* output_layer);
void predict_drawing(float* input, layer* hidden_layer, layer* output_layer, float* out, int* digit);

#endif
